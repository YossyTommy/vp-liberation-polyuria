#!/usr/bin/env python3
"""
07_switch_closed.py - eICU-CRD v2.0, switch sensitivity and closed cohort.
Extension of 06_vpstop_timing.py (post hoc, exploratory, no causal claims):
 (A) NE->VP "switch" stays inside NE-first (and VP->NE switch inside VP-first): counts and refits
     without them.
 (B) Closed cohort: alive and in the ICU from -12 to +8 h around the anchor with urine recorded in
     >= 6 of 12 bins before and >= 4 of 8 bins after (strict: >= 10/12 and >= 7/8).
 (C) Number-at-risk table (VP-stop new-onset population, Aalen-Johansen setup of eicu_vpstop.py).
 (D) Size of the basic population and number of stays with urine recorded at selected hours.

The usage-group cohort is rebuilt in memory with eicu_cohort.build_usage_cohort and checked
against cohort_usage_groups.parquet; anchor population, hourly series, cumulative incidence and
the overlap-weighted stop-order estimate come from eicu_vpstop.py (the same functions as in
06_vpstop_timing.py, so the estimates of that script are reproduced here, not read from its output).

Switch definitions (first VP episode = idx_start..t0a_vp)
  NE->VP switch (any NE+VP stay; in practice NE-first): the first VP episode starts strictly after
     the end of the last NE episode that starts at or before T0b (idx_start > last_ne_end; for
     NE-first last_ne_end = T0a), i.e. NE had completely stopped before VP was started, with no
     NE/VP overlap.
  VP->NE switch (VP-first): no NE episode (start <= T0b) overlaps the first VP episode, i.e. every
     NE episode either starts strictly after the VP stop (t0a_vp) or ends strictly before the VP
     start; at least one NE episode starts after the VP stop (guaranteed by the VP-first
     definition). Stays with an NE episode that ended before VP started (NE -> VP -> NE) are
     counted inside this flag and reported separately.

Optional environment variables: MICE_M (default 10), NBOOT (200).
Outputs (OUT_DIR/eicu_07/)
  switch_sensitivity.csv, closed_cohort.csv, closed_cohort_hourly.csv, risk_table.csv,
  basic_population_n.csv, reproduction_checks.csv, run_log.txt

Run:  python eicu/07_switch_closed.py   (after 01_primary_analysis.py and 04_usage_groups.py)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from scipy.stats import t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # eicu_* modules; needed when Python runs in safe-path mode
from eicu_cohort import build_usage_cohort, columns_differing_from_stored  # noqa: E402
from eicu_common import COHORT_PRIMARY, COHORT_USAGE, DB, RUN_LOG, RunLog, cohort_path, env_int, out_dir  # noqa: E402
from eicu_extract import load_source  # noqa: E402
from eicu_stats import COV, cr_times, fit_groups, fit_or  # noqa: E402
from eicu_vpstop import (ORDER_GROUPS, anchor_population, cif_block, competing_risk_frame, cov_at, hourly_table,  # noqa: E402
                         order_estimate, order_frame, order_sources, sodium_frame)

NBOOT = env_int("NBOOT", 200)
G5 = ["VP-only", "NE-first", "Simultaneous", "VP-first"]
OUTS2 = (("polyuria_24h", "24 h"), ("poly8", "0-8 h"), ("poly_8_24", ">8-24 h"))
OUT5 = (("poly8", "0-8 h"), ("poly_8_24", ">8-24 h"), ("polyuria_24h", "24 h"))
T0_TWO_GROUP = "T0 (two-group primary analysis)"
ANCHOR_LABEL = {"VP_stop": "VP stop", "NE_stop": "NE stop"}


class Rows:
    """Result rows of one output table."""

    def __init__(self):
        self.rows = []

    def add(self, analysis, group, anchor, n, events, estimate, lo=np.nan, hi=np.nan, note="", **kw):
        anchor = ANCHOR_LABEL.get(anchor, anchor)
        r = dict(db=DB, analysis=analysis, group=group, anchor=anchor, n=n, events=events, estimate=estimate, lo=lo, hi=hi, note=note)
        r.update(kw)
        self.rows.append(r)

    def frame(self):
        return pd.DataFrame(self.rows)


def switch_flags(C, EP):
    """Stay-level switch flags, added to C."""
    EPne = EP["ne"][["patientunitstayid", "start", "end"]].merge(C[["patientunitstayid", "t0", "idx_start", "t0a_vp"]], on="patientunitstayid")
    EPne = EPne[EPne.start <= EPne.t0]                                     # NE episodes used before T0b
    ov = EPne[(EPne.start <= EPne.t0a_vp) & (EPne.end >= EPne.idx_start)]  # NE episode overlapping the first VP episode
    pre = EPne[EPne.end < EPne.idx_start]                                  # NE episode finished before VP start
    C["sw_ne_vp"] = (C.last_ne_end.notna() & (C.grp != "NE-only") & (C.idx_start > C.last_ne_end)).astype(int)
    C["sw_vp_ne"] = ((C.grp.isin(["VP-first", "VP-first+other"])) & ~C.patientunitstayid.isin(set(ov.patientunitstayid))).astype(int)
    C["sw_vp_ne_prior_ne"] = ((C.sw_vp_ne == 1) & C.patientunitstayid.isin(set(pre.patientunitstayid))).astype(int)
    C["switch"] = ((C.grp == "NE-first") & (C.sw_ne_vp == 1)) | ((C.grp == "VP-first") & (C.sw_vp_ne == 1))
    C["switch"] = C.switch.astype(int)
    nf = C.grp == "NE-first"
    assert (C.loc[nf, "sw_ne_vp"] == (C.loc[nf, "idx_start"] > C.loc[nf, "t0a"]).astype(int)).all()


def two_group(P, tag, NA):
    """Adjusted odds ratios of the two-group comparison for the polyuria outcomes and hypernatraemia."""
    out = []
    for y, yl in OUTS2:
        out.append((yl, fit_or(P, y, "vp", COV, label=tag)))
    S, _ = sodium_frame(P, NA)
    out.append(("Na>=150 72 h", fit_or(S, "na150_72h", "vp", COV, label=tag)))
    return out


def five_group(P, tag):
    """Group odds ratios of the 5-group model at T0b for the polyuria outcomes and hypernatraemia."""
    res = []
    for y, yl in OUT5:
        for r in fit_groups(P, y, "grp", G5, covs=COV, label=tag):
            r["outcome_label"] = yl
            res.append(r)
    S = P[(P.na_pre_max < 150) & (P.na_post72_n >= 1)].copy()
    S["na150_72h"] = (S.na_post72_max >= 150).astype(int)
    for r in fit_groups(S, "na150_72h", "grp", G5, covs=COV, label=tag):
        r["outcome_label"] = "Na>=150 72 h"
        res.append(r)
    return res


def window(Hh, a, b):
    """Recorded stay-hours, stay-hours in a run and mean urine output over the bins a..b of an hourly table."""
    w = Hh[(Hh.hour >= a) & (Hh.hour <= b)]
    rec, run = int(w.n_recorded.sum()), int(w.n_in_run.sum())
    uo_pooled = (w.uo_mean * w.n_recorded).sum() / rec if rec else np.nan        # mean over all recorded stay-hours
    uo_hourmean = w.uo_mean.mean()                                                 # unweighted mean of hourly means
    return rec, run, uo_pooled, uo_hourmean


def main():
    warnings.filterwarnings("ignore")
    t_start = time.time()
    log = RunLog()
    say = log.say
    out = out_dir("07")
    REPRO = []

    def rep(item, val, target):
        ok = str(val) == str(target)
        REPRO.append(dict(item=item, value=val, target=target, ok=ok))
        say("REPRO", item, val, "target", target, "OK" if ok else "MISMATCH")
        return ok

    # ---- cohort, anchor population, populations of the stop-order comparison -------------------
    src = load_source(say)
    C, tabs = build_usage_cohort(src, say)
    bad = columns_differing_from_stored(C, pd.read_parquet(cohort_path(COHORT_USAGE)))
    assert not bad, f"rebuilt cohort differs from {COHORT_USAGE}: {bad}"
    anc = anchor_population(C, tabs.ur)
    A = anc.A
    NA = tabs.na[["patientunitstayid", "h", "v"]].copy()
    cohort2 = pd.read_parquet(cohort_path(COHORT_PRIMARY))
    P2 = cohort2[cohort2.primary == 1].copy()
    chk = P2[["patientunitstayid", "t0"]].merge(C[["patientunitstayid", "t0"]], on="patientunitstayid", suffixes=("_two", "_usage"))
    assert len(chk) == len(P2) and (chk.t0_two == chk.t0_usage).all()
    PR = C[C.primary == 1].copy()
    ORD = C[C.grp.isin(ORDER_GROUPS)].copy()
    XA = cov_at(ORD, ORD.t0a.values, order_sources(C, tabs, src))
    POP1 = A[A.grp.isin(ORDER_GROUPS) & (A.new_onset_pop == 1)].assign(_y=lambda d: (d.onset_anc <= 8).astype(int))
    POP2 = C[(C.primary == 1) & C.grp.isin(ORDER_GROUPS)].assign(_y=lambda d: d.poly8.astype(int))
    POPS = {"1_VPstop_0_8h": POP1, "2_T0b_0_8h": POP2}

    # =====================================================================================
    # Switch flags and counts
    # =====================================================================================
    switch_flags(C, tabs.EP)
    SW = dict(zip(C.patientunitstayid, C.switch))
    SWNV = dict(zip(C.patientunitstayid, C.sw_ne_vp))
    for D_ in (A, PR):
        D_["switch"] = D_.patientunitstayid.map(SW).astype(int)
        D_["sw_ne_vp"] = D_.patientunitstayid.map(SWNV).astype(int)
    x_nf = PR[PR.grp == "NE-first"]
    rep("NE-first switch at T0b (5-group population, expected 45/263)", f"{int(x_nf.switch.sum())}/{len(x_nf)}", "45/263")

    sw = Rows()
    add = sw.add
    P2["usage_grp"] = P2.patientunitstayid.map(dict(zip(C.patientunitstayid, C.grp)))
    P2["switch"] = P2.patientunitstayid.map(dict(zip(C.patientunitstayid, C.sw_ne_vp))).fillna(0).astype(int) * (P2.grp == "VP")
    v = P2[P2.grp == "VP"]
    add("count: NE->VP switch in 2-group primary VP group", "2grp VP", T0_TWO_GROUP, len(v), int(v.switch.sum()), 100 * v.switch.mean(),
        note="stop-order of the 370 VP stays: " + ", ".join(f"{k} {n}" for k, n in v.usage_grp.value_counts().items()) +
             "; switch stays by stop-order group: " + ", ".join(f"{k} {n}" for k, n in v[v.switch == 1].usage_grp.value_counts().items()))
    add("count: NE->VP switch in NE-first (5-group analysis population)", "NE-first", "T0b", len(x_nf), int(x_nf.switch.sum()), 100 * x_nf.switch.mean(),
        note=f"median h NE stop -> VP start among switches {((x_nf.idx_start - x_nf.t0a)[x_nf.switch == 1] / 60).median():.1f}")
    a = A[(A.grp == "NE-first") & (A.new_onset_pop == 1)]
    add("count: NE->VP switch in NE-first (VP-stop new-onset population)", "NE-first", "VP stop", len(a), int(a.switch.sum()), 100 * a.switch.mean())
    a0 = A[(A.grp == "NE-first")]
    add("count: NE->VP switch in NE-first (VP-stop basic population)", "NE-first", "VP stop", len(a0), int(a0.switch.sum()), 100 * a0.switch.mean())
    for lab_, X_, an in (("5-group analysis population", PR, "T0b"), ("VP-stop new-onset population", A[A.new_onset_pop == 1], "VP stop"),
                         ("VP-stop basic population", A, "VP stop")):
        x = X_[X_.grp == "VP-first"]
        xc = C.set_index("patientunitstayid").loc[x.patientunitstayid]
        add(f"count: VP->NE switch in VP-first ({lab_})", "VP-first", an, len(x), int(x.switch.sum()), 100 * x.switch.mean(),
            note=f"of these, NE episode also before VP start (NE->VP->NE): {int(xc.sw_vp_ne_prior_ne.sum())}")
    x = PR[PR.grp == "VP-first+other"]
    add("count: VP->NE switch in VP-first+other (5-group population; not in 5 main groups)", "VP-first+other", "T0b", len(x),
        int(C.set_index("patientunitstayid").loc[x.patientunitstayid].sw_vp_ne.sum()), 100 * C.set_index("patientunitstayid").loc[x.patientunitstayid].sw_vp_ne.mean(),
        note="other vasopressor running after the VP stop; reported for information only")
    x = PR[PR.grp == "Simultaneous"]
    add("count: NE->VP no-overlap in Simultaneous (5-group population; for information)", "Simultaneous", "T0b", len(x), int(x.sw_ne_vp.sum()), 100 * x.sw_ne_vp.mean())

    # =====================================================================================
    # 2-group primary refit without switch stays in the VP group
    # =====================================================================================
    for tag, P in (("2-group primary (reproduction, all VP)", P2), ("2-group primary, VP group excluding NE->VP switch", P2[P2.switch == 0])):
        for yl, r in two_group(P, tag, NA):
            add(tag, "VP vs NE without VP", T0_TWO_GROUP, r["n"], r["events"], r["OR"], r["lo"], r["hi"],
                note=f"outcome {yl}; VP {r['events_exposed']}/{r['n_exposed']}, NE {r['events'] - r['events_exposed']}/{r['n'] - r['n_exposed']}; {r['method']}, hospital-clustered SE",
                outcome=yl)
    r2 = pd.DataFrame([r for r in sw.rows if r["analysis"].startswith("2-group primary (reproduction")])
    rep("2-group aOR 24 h", f"{r2.loc[r2.outcome == '24 h', 'estimate'].iloc[0]:.2f} ({r2.loc[r2.outcome == '24 h', 'lo'].iloc[0]:.2f}-{r2.loc[r2.outcome == '24 h', 'hi'].iloc[0]:.2f})", "1.34 (0.97-1.86)")
    rep("2-group aOR 0-8 h", f"{r2.loc[r2.outcome == '0-8 h', 'estimate'].iloc[0]:.2f} ({r2.loc[r2.outcome == '0-8 h', 'lo'].iloc[0]:.2f}-{r2.loc[r2.outcome == '0-8 h', 'hi'].iloc[0]:.2f})", "1.94 (1.33-2.83)")
    rep("2-group aOR Na>=150", f"{r2.loc[r2.outcome == 'Na>=150 72 h', 'estimate'].iloc[0]:.2f}", "2.38")

    # =====================================================================================
    # 5-group T0b model with NE-first switch stays removed
    # =====================================================================================
    R5a = five_group(PR, "5-group T0b (reproduction)")
    PRx = PR[~((PR.grp == "NE-first") & (PR.switch == 1))].copy()
    R5b = five_group(PRx, "5-group T0b, NE-first excluding NE->VP switch (switch stays dropped)")
    for R in (R5a, R5b):
        for r in R:
            if r["group"] != "NE-first":
                continue
            k, n = r["events_group"].split(" ")[0].split("/")
            add(r["analysis"], "NE-first" if "excluding" not in r["analysis"] else "NE-first (no switch)", "T0b", int(n), int(k), r["OR"], r["lo"], r["hi"],
                note=f"outcome {r['outcome_label']}; vs NE without VP {r['events_ref']}; {r['method']}; model n={r['n_model']}", outcome=r["outcome_label"])
    m5 = [r for r in R5a if r["group"] == "NE-first" and r["outcome_label"] == "0-8 h"][0]
    rep("5-group NE-first 0-8 h aOR", f"{m5['OR']:.2f} ({m5['lo']:.2f}-{m5['hi']:.2f})", "2.02 (1.32-3.08)")
    xs = PR[(PR.grp == "NE-first") & (PR.switch == 1)]
    for y, yl in OUT5:
        add("NE-first NE->VP switch subgroup (descriptive)", "NE-first switch", "T0b", len(xs), int(xs[y].sum()), 100 * xs[y].mean(), outcome=yl, note=f"outcome {yl}; crude n/N")
    s = xs[(xs.na_pre_max < 150) & (xs.na_post72_n >= 1)]
    add("NE-first NE->VP switch subgroup (descriptive)", "NE-first switch", "T0b", len(s), int((s.na_post72_max >= 150).sum()),
        100 * (s.na_post72_max >= 150).mean() if len(s) else np.nan, outcome="Na>=150 72 h", note="outcome Na>=150 72 h; Na denominator")
    xn = PR[(PR.grp == "NE-first") & (PR.switch == 0)]
    for y, yl in OUT5:
        add("NE-first without switch (descriptive)", "NE-first (no switch)", "T0b", len(xn), int(xn[y].sum()), 100 * xn[y].mean(), outcome=yl, note=f"outcome {yl}; crude n/N")

    # stay-hour summary of NE-first around the VP stop (log only)
    hb = hourly_table(A[A.grp == "NE-first"], "NE-first", "VP_stop", anc)
    pre_, post_ = window(hb, -12, -1), window(hb, 0, 7)
    say(f"NE-first in-run pre {pre_[1]}/{pre_[0]} = {100*pre_[1]/pre_[0]:.1f}%, post {post_[1]}/{post_[0]} = {100*post_[1]/post_[0]:.1f}%; "
        f"UO pooled {pre_[2]:.2f}->{post_[2]:.2f}; UO mean-of-hourly-means {pre_[3]:.2f}->{post_[3]:.2f}")

    # =====================================================================================
    # VP-stop anchor: NE-first without switch / switch alone
    # =====================================================================================
    SUBS = {"NE-first": A.grp == "NE-first", "NE-first (no switch)": (A.grp == "NE-first") & (A.switch == 0),
            "NE-first switch": (A.grp == "NE-first") & (A.switch == 1), "VP-first": A.grp == "VP-first", "NE without VP": A.grp == "NE-only"}
    assert int(A.loc[A.grp == "VP-first", "switch"].sum()) == 0   # no VP->NE switch in VP-first -> 'VP-first (no switch)' = VP-first
    CIFX, HOURX = {}, {}
    for g, m in SUBS.items():
        X = A[m & (A.new_onset_pop == 1)]
        an = X.anc_name.iloc[0]
        rr, _ = cif_block(X, g, an, "(ii) new-onset", NBOOT)
        cc = pd.DataFrame(rr)
        CIFX[g] = cc
        k8 = int((X.onset_anc <= 8).sum())
        add("VP-stop anchor: crude new-onset polyuria 0-8 h", g, an, len(X), k8, 100 * k8 / len(X), note="ignores competing events (definition of the VP-stop anchored table of the usage-group analysis)")
        for hh in (8, 24):
            r8 = cc[cc.time_h == hh].iloc[0]
            add(f"VP-stop anchor: AJ CIF at {hh} h (%)", g, an, len(X), int(r8.n_events), 100 * r8.cif, 100 * r8.lo, 100 * r8.hi,
                note=f"Aalen-Johansen; competing death/discharge/restart; hospital-cluster bootstrap B={NBOOT}")
        Hh = hourly_table(A[m], g, an, anc)
        HOURX[g] = Hh
        for (a_, b_, lab_) in ((-12, -1, "pre -12..-1 h"), (0, 7, "post 0..7 h")):
            rec, run, uop, uoh = window(Hh, a_, b_)
            add(f"VP-stop anchor: stay-hour in-run % ({lab_})", g, an, rec, run, 100 * run / rec if rec else np.nan,
                note=f"basic population (alive in ICU at anchor, n={int(m.sum())}); denominator = stay-hours with urine recorded")
            add(f"VP-stop anchor: mean hourly urine output mL/kg/h ({lab_})", g, an, rec, np.nan, uop,
                note=f"mean over recorded stay-hours (pooled); unweighted mean of hourly means = {uoh:.2f}; basic population n={int(m.sum())}")
    rep("VP-stop 0-8 h NE-first (expected 34/289)", f"{int((A[SUBS['NE-first'] & (A.new_onset_pop == 1)].onset_anc <= 8).sum())}/{int((SUBS['NE-first'] & (A.new_onset_pop == 1)).sum())}", "34/289")
    rep("AJ CIF 8 h NE-first/VP-first/NE w/o VP", "/".join(f"{100 * CIFX[g].loc[CIFX[g].time_h == 8, 'cif'].iloc[0]:.1f}" for g in ("NE-first", "VP-first", "NE without VP")), "10.4/5.0/6.0")
    rep("NE without VP new-onset n", int((SUBS["NE without VP"] & (A.new_onset_pop == 1)).sum()), 3422)

    # =====================================================================================
    # Overlap weighting NE-first vs VP-first, all stays and without switch stays in both groups
    # =====================================================================================
    ato_all = {}
    for key, P in POPS.items():
        an = "VP stop" if key.startswith("1") else "T0b"
        e = order_estimate(order_frame(P, XA), NBOOT)
        ato_all[key] = (np.exp(e.qb[0]), np.exp(e.qb[0] - 1.96 * e.se[0]), np.exp(e.qb[0] + 1.96 * e.se[0]))
        add("ATO NE-first vs VP-first 0-8 h (reproduction, all stays)", "NE-first vs VP-first", an, e.n1 + e.n0, e.k1 + e.k0,
            *ato_all[key], note=f"NE-first {e.k1}/{e.n1}, VP-first {e.k0}/{e.n0}; OR>1 = more polyuria in NE-first")
        e = order_estimate(order_frame(P[P.patientunitstayid.map(SW) == 0], XA), NBOOT)
        crude = (e.cor, np.exp(np.log(e.cor) - 1.96 * e.cse), np.exp(np.log(e.cor) + 1.96 * e.cse))
        add("ATO NE-first vs VP-first 0-8 h, switch stays excluded from both groups", "NE-first vs VP-first (no switch)", an, e.n1 + e.n0, e.k1 + e.k0,
            np.exp(e.qb[0]), np.exp(e.qb[0] - 1.96 * e.se[0]), np.exp(e.qb[0] + 1.96 * e.se[0]),
            note=f"NE-first {e.k1}/{e.n1}, VP-first {e.k0}/{e.n0}; crude OR {crude[0]:.2f} ({crude[1]:.2f}-{crude[2]:.2f}); 10-var PS at T0a, MICE m={e.m}, hospital-cluster bootstrap B={NBOOT} ({min(e.nboot)}-{max(e.nboot)} ok); OR>1 = more polyuria in NE-first")
    rep("ATO VP stop", "%.2f (%.2f-%.2f)" % ato_all["1_VPstop_0_8h"], "2.23 (0.97-5.12)")
    rep("ATO T0b", "%.2f (%.2f-%.2f)" % ato_all["2_T0b_0_8h"], "1.29 (0.58-2.87)")
    sw.frame().to_csv(os.path.join(out, "switch_sensitivity.csv"), index=False)

    # =====================================================================================
    # Closed cohort
    # =====================================================================================
    closed = Rows()
    add = closed.add
    HRA, SER, RUN = anc.HRA, anc.SER, anc.RUN
    GB = {g: SUBS[g] for g in ("NE-first", "VP-first", "NE without VP", "NE-first (no switch)")}
    npre = HRA[(HRA.h >= -12) & (HRA.h <= -1)].groupby("patientunitstayid").h.nunique()
    npost = HRA[(HRA.h >= 0) & (HRA.h <= 7)].groupby("patientunitstayid").h.nunique()
    A["n_rec_pre12"] = A.patientunitstayid.map(npre).fillna(0).astype(int)
    A["n_rec_post8"] = A.patientunitstayid.map(npost).fillna(0).astype(int)
    A["in_icu_m12_p8"] = ((A.anc - 720 >= 0) & (A.unitdischargeoffset >= A.anc + 480)).astype(int)   # ICU admission <= -12 h; ICU discharge/death >= +8 h

    def stay_win(sid, a, b):
        d = SER.get(sid, {})
        hs = [h for h in range(a, b + 1) if h in d]
        run = RUN.get(sid, set())
        return len(hs), sum(1 for h in hs if h in run), (np.mean([d[h] for h in hs]) if hs else np.nan)

    hourly_rows = []
    for ver, (kp, kq) in (("closed (>=6/12 pre, >=4/8 post)", (6, 4)), ("strict (>=10/12 pre, >=7/8 post)", (10, 7))):
        for g, m in GB.items():
            X = A[m & (A.in_icu_m12_p8 == 1) & (A.n_rec_pre12 >= kp) & (A.n_rec_post8 >= kq)]
            an = X.anc_name.iloc[0] if len(X) else ""
            basic = int(m.sum())
            st = np.array([stay_win(s, -12, -1) + stay_win(s, 0, 7) for s in X.patientunitstayid])   # rec_pre, run_pre, uo_pre, rec_post, run_post, uo_post
            n = len(X)
            add(f"{ver}: n", g, an, basic, n, 100 * n / basic, note="events = closed-cohort n; n = basic population (alive in ICU at anchor); estimate = %")
            for j, lab_ in ((0, "pre -12..-1 h"), (3, "post 0..7 h")):
                rec, run = int(st[:, j].sum()), int(st[:, j + 1].sum())
                add(f"{ver}: stay-hour in-run % ({lab_})", g, an, rec, run, 100 * run / rec, note=f"closed cohort n={n}; denominator = recorded stay-hours")
            anypre, anypost = st[:, 1] > 0, st[:, 4] > 0
            b_, c_ = int((anypre & ~anypost).sum()), int((~anypre & anypost).sum())
            p = binomtest(min(b_, c_), b_ + c_, 0.5).pvalue if b_ + c_ else np.nan
            add(f"{ver}: stays with >=1 in-run hour (pre -12..-1 h)", g, an, n, int(anypre.sum()), 100 * anypre.mean(), note="per-stay proportion")
            add(f"{ver}: stays with >=1 in-run hour (post 0..7 h)", g, an, n, int(anypost.sum()), 100 * anypost.mean(),
                note=f"paired vs pre: McNemar exact p={p:.3g}; discordant pre-only {b_}, post-only {c_}")
            add(f"{ver}: McNemar exact p (>=1 in-run hour, post vs pre)", g, an, n, b_ + c_, p, note=f"discordant pairs: pre-only {b_}, post-only {c_}")
            d = st[:, 5] - st[:, 2]
            mu = d.mean()
            se = d.std(ddof=1) / np.sqrt(n)
            q = tdist.ppf(.975, n - 1)
            add(f"{ver}: per-stay mean urine output mL/kg/h (pre -12..-1 h)", g, an, n, np.nan, np.nanmean(st[:, 2]), note="mean of per-stay means (recorded hours)")
            add(f"{ver}: per-stay mean urine output mL/kg/h (post 0..7 h)", g, an, n, np.nan, np.nanmean(st[:, 5]), note="mean of per-stay means (recorded hours)")
            add(f"{ver}: paired difference in per-stay mean urine output (post - pre), mL/kg/h", g, an, n, np.nan, mu, mu - q * se, mu + q * se,
                note=f"paired t 95% CI (not cluster-adjusted); median diff {np.median(d):.2f}; stays with increase {int((d > 0).sum())}/{n}")
            hourly_rows.append(hourly_table(X, g, an, anc, extra={"cohort": ver}))
    closed.frame().to_csv(os.path.join(out, "closed_cohort.csv"), index=False)
    pd.concat(hourly_rows, ignore_index=True).to_csv(os.path.join(out, "closed_cohort_hourly.csv"), index=False)

    # =====================================================================================
    # Number-at-risk table (VP-stop new-onset population; horizon 24 h)
    # =====================================================================================
    rk = []
    for g in ("NE-first", "VP-first", "NE without VP", "NE-first (no switch)", "NE-first switch"):
        X = A[SUBS[g] & (A.new_onset_pop == 1)]
        an = X.anc_name.iloc[0]
        t_, s_ = cr_times(competing_risk_frame(X), horizon=24.0)
        cc = CIFX[g]
        for h in [0, 2, 4, 6, 8, 12, 16, 20, 24]:
            rk.append(dict(db=DB, analysis="risk table, VP-stop new-onset population", group=g, anchor=an, time_h=h, n=len(X),
                           n_at_risk=int((t_ >= h).sum()) if h > 0 else len(t_),
                           cum_polyuria=int(((s_ == 1) & (t_ <= h)).sum()), cum_death=int(((s_ == 2) & (t_ <= h)).sum()),
                           cum_icu_discharge=int(((s_ == 3) & (t_ <= h)).sum()), cum_vasopressor_restart=int(((s_ == 4) & (t_ <= h)).sum()),
                           censored=int(((s_ == 0) & (t_ <= h)).sum()),
                           cif_pct=100 * cc.loc[cc.time_h == h, "cif"].iloc[0], cif_lo=100 * cc.loc[cc.time_h == h, "lo"].iloc[0], cif_hi=100 * cc.loc[cc.time_h == h, "hi"].iloc[0],
                           note="n_at_risk(t) = #(time >= t) (as vpstop_cif); events at t counted as cumulative by t; censored = administrative at 24 h only"))
    RISK = pd.DataFrame(rk)
    RISK["anchor"] = RISK.anchor.replace(ANCHOR_LABEL)
    RISK.to_csv(os.path.join(out, "risk_table.csv"), index=False)
    at8 = RISK[(RISK.time_h == 8)]
    for g in ("NE-first", "VP-first", "NE without VP"):
        b = CIFX[g][CIFX[g].time_h == 8].iloc[0]
        assert int(at8.loc[at8.group == g, "n_at_risk"].iloc[0]) == int(b.n_at_risk) and int(at8.loc[at8.group == g, "cum_polyuria"].iloc[0]) == int(b.n_events)

    # =====================================================================================
    # Basic population and stays with urine recorded at selected hours
    # =====================================================================================
    fn_ = []
    for g in ("NE-first", "VP-first", "NE without VP", "NE-first (no switch)", "NE-first switch"):
        Hh = HOURX.get(g)
        m = SUBS[g]
        an = A[m].anc_name.iloc[0]
        fn_.append(dict(db=DB, analysis="basic population (alive in ICU at anchor)", group=g, anchor=an, hour=np.nan, n=int(m.sum()),
                        events=np.nan, estimate=np.nan, lo=np.nan, hi=np.nan, note="n = stays alive and in ICU at the anchor (anchor population)"))
        for h in [-24, -16, -8, -1, 0, 7, 8, 16, 23]:
            r = Hh[Hh.hour == h].iloc[0]
            fn_.append(dict(db=DB, analysis="n_recorded at hour", group=g, anchor=an, hour=h, n=int(r.n_alive_in_icu), events=int(r.n_recorded),
                            estimate=100 * r.n_in_run / r.n_recorded if r.n_recorded else np.nan, lo=np.nan, hi=np.nan,
                            note="n = alive in ICU at bin start; events = n_recorded; estimate = in-run % (n_in_run / n_recorded)"))
    BASIC = pd.DataFrame(fn_)
    BASIC["anchor"] = BASIC.anchor.replace(ANCHOR_LABEL)
    BASIC.to_csv(os.path.join(out, "basic_population_n.csv"), index=False)
    pd.DataFrame(REPRO).to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
    log.write(os.path.join(out, RUN_LOG))
    assert all(r["ok"] for r in REPRO), "reproduction failed"
    say("done in %.0f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
