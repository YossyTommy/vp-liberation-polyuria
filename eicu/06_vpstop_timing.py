#!/usr/bin/env python3
"""
06_vpstop_timing.py - eICU-CRD v2.0, timing of polyuria and hypernatremia relative to the
vasopressin stop, by stop order. Post hoc, exploratory.

  Part 1  timing of polyuria around the VP stop by stop order
  Part 2  timing of hypernatremia relative to early polyuria
  Part 3  NE-first vs VP-first: characteristics at T0a and one overlap-weighted estimate

The usage-group cohort is rebuilt in memory with eicu_cohort.build_usage_cohort (the source
tables are needed for the anchored series) and checked against cohort_usage_groups.parquet in
every column. Anchors, populations and the competing-risk setup are described in
eicu_vpstop.py; groups: NE-only, VP-only, NE-first, Simultaneous, VP-first, VP-first+other
(kept separate), VP restarted (not analysed).

Optional environment variables: MICE_M (default 10), NBOOT (200).
Outputs (OUT_DIR/eicu_06/)
  eICU_vpstop_timing_results.xlsx, vpstop_hourly.csv, vpstop_cif.csv,
  vpstop_cif_sens_incl_pre_polyuria.csv, vpstop_t0b_hist.csv, na_timing.csv, na_by_poly.csv,
  order_estimates.csv, order_smd.csv, order_smd_pop_VPstop.csv, run_log.txt

Run:  python eicu/06_vpstop_timing.py   (after 01_primary_analysis.py, 04_usage_groups.py and 05_fluid_balance.py)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # eicu_* modules; needed when Python runs in safe-path mode
from eicu_cohort import build_usage_cohort, columns_differing_from_stored  # noqa: E402
from eicu_common import (COHORT_FLUID, COHORT_PRIMARY, COHORT_USAGE, DB, RUN_LOG, RunLog, cohort_path, env_int,  # noqa: E402
                         out_dir)
from eicu_extract import load_source  # noqa: E402
from eicu_stats import COV, fit_or, fmt, smd, smd2  # noqa: E402
from eicu_vpstop import (ANCHOR_GROUPS, ORDER_GROUPS, PSV, anchor_population, cif_block, cov_at, hourly_table,  # noqa: E402
                         n_pct, order_estimate, order_frame, order_sources, sodium_frame)

NBOOT = env_int("NBOOT", 200)
USAGE_GROUPS = ["NE-only", "VP-only", "NE-first", "Simultaneous", "VP-first", "VP-first+other"]
TWO_GROUPS = ["2grp_VP", "2grp_NE"]
VARS = [("age_n", "Age, y"), ("male", "Male"), ("cvicu", "Cardiac ICU"), ("card_surg", "Cardiac surgery (APACHE dx)"), ("sepsis", "Sepsis"),
        ("apache4", "APACHE IV (ICU day 1)"), ("press_h", "Vasopressor time to anchor, h"), ("ne_running", "NE episode running at anchor (incl. stopping at anchor)"),
        ("ne_rate", "NE rate just before anchor, mcg/kg/min (convertible)"), ("ne_rate_unknown", "NE rate not convertible"),
        ("vp_running", "VP episode running at anchor (incl. stopping at anchor)"), ("vp_rate", "VP rate just before anchor, units/min"), ("vp_rate_unknown", "VP rate not convertible"),
        ("other_press", "Other vasopressor (PE/epi/dopa/AT-II) before anchor"), ("map6", "MAP median 6 h before, mm Hg"),
        ("lactate_max24", "Max lactate 24 h before, mmol/L"), ("na_last", "Last Na 24 h before, mmol/L"),
        ("kdigo_0", "KDIGO 0"), ("kdigo_1", "KDIGO 1"), ("kdigo_2", "KDIGO 2"), ("kdigo_3", "KDIGO 3"), ("kdigo_unk", "KDIGO unknown"),
        ("uo_pre6", "Urine output 6 h before, mL/kg/h"), ("fo_pct", "Cumulative FO% ICU admission to anchor"),
        ("mv", "Mechanical ventilation 24 h before"), ("steroid", "Corticosteroid 24 h before"), ("inotrope", "Inotrope at anchor")]
SMD_VARS = VARS + [("kdigo_ge2", "KDIGO >= 2"), ("log_press_h", "log(1+vasopressor h)")]


def check_cohort(C, say):
    """The cohort rebuilt in memory must equal the stored usage-group cohort in every column."""
    stored = pd.read_parquet(cohort_path(COHORT_USAGE))
    bad = columns_differing_from_stored(C, stored)
    assert not bad, f"rebuilt cohort differs from {COHORT_USAGE}: {bad}"
    say(f"cohort rebuilt; identical to {COHORT_USAGE} ({len(C)} stays, {len(stored.columns)} columns)")


def vpstop_flow(C, A):
    """Flow of the anchor population per group."""
    fl = []
    for g in ANCHOR_GROUPS:
        x0 = C[C.grp == g]
        x1 = A[A.grp == g]
        fl.append(dict(group=g, anchor=x1.anc_name.iloc[0], classified=len(x0),
                       adult_noRRT=int(((x0.adult == 1) & (x0.rrt == 0)).sum()), anchor_population_alive_in_ICU_no_pit=len(x1),
                       urine_ge12_bins_0_24=int(x1.urine24_pop.sum()), new_onset_pop=int(x1.new_onset_pop.sum()),
                       excluded_by_pre12_polyuria=int(((x1.urine24_pop == 1) & (x1.pre12_anc == 1)).sum()),
                       excluded_other_vasopressor_at_NE_stop=int(((x0.riskset == 1) & (x0.excl_pit == 0) & (x0.press_t0 == 1)).sum()) if g == "NE-only" else ""))
    return pd.DataFrame(fl)


def t0b_position(C, A, rep):
    """Position of T0b relative to the VP stop in VP-first stays (new-onset population)."""
    VF = A[(A.grp == "VP-first") & (A.new_onset_pop == 1)]
    hb = np.floor(VF.t0b_from_anc_h).astype(int)
    hist = hb.value_counts().sort_index().rename_axis("hour_from_vpstop").reset_index(name="n")
    hist.insert(0, "db", DB)
    pr = C[(C.primary == 1) & (C.grp == "VP-first")]
    rep("VP-first VP stop -> T0b median, 5-group analysis population (expected 11.0 h)", round(pr.gap_vpstop_t0b_h.median(), 1), 11.0)
    rows = []
    for lab_, x in (("VP-stop new-onset population (figure)", VF.t0b_from_anc_h), ("5-group analysis population (T0b)", pr.gap_vpstop_t0b_h),
                    ("anchor population", A.loc[A.grp == "VP-first", "t0b_from_anc_h"])):
        rows.append(dict(population=lab_, n=len(x), median=round(x.median(), 2), q1=round(x.quantile(.25), 2), q3=round(x.quantile(.75), 2),
                         le8h=n_pct(int((x <= 8).sum()), len(x)), le24h=n_pct(int((x <= 24).sum()), len(x))))
    # polyuria between the VP stop and T0b, and onset relative to T0b
    k = VF[(VF.onset_anc <= 24)]
    rows.append(dict(population="VP-stop new-onset population: VP-first with onset 0-24 h: onset before T0b (NE still running)", n=len(k),
                     median=np.nan, q1=np.nan, q3=np.nan, le8h=n_pct(int((k.onset_anc <= k.t0b_from_anc_h).sum()), len(k)), le24h=""))
    k8 = VF[(VF.onset_anc <= 8)]
    rows.append(dict(population="VP-stop new-onset population: VP-first with onset 0-8 h: onset before T0b (NE still running)", n=len(k8),
                     median=np.nan, q1=np.nan, q3=np.nan, le8h=n_pct(int((k8.onset_anc <= k8.t0b_from_anc_h).sum()), len(k8)), le24h=""))
    return hist, pd.DataFrame(rows)


def sodium_timing(S2, X2, S5, X5):
    """First sodium >= 150 mmol/L and sodium measurement by 12-h bin after T0b (and after the
    VP stop for the VP-first groups)."""
    rows, rows_x = [], []

    def timing(S, X, gcol, groups, anchor, shift=None, maxh=72):
        edges = np.arange(0, maxh + 12, 12)
        for g in groups:
            s = S[S[gcol] == g]
            ids = set(s.patientunitstayid)
            x = X[X.patientunitstayid.isin(ids)].copy()
            tna = s.t_na150.copy()
            if shift is not None:
                sh = s.set_index("patientunitstayid")[shift]
                x["h"] = x.h + x.patientunitstayid.map(sh).values
                tna = tna + s[shift]
            for a, b in zip(edges[:-1], edges[1:]):
                xb = x[(x.h > a) & (x.h <= b)]
                rows.append(dict(db=DB, group=g, anchor=anchor, bin_start_h=int(a), bin_end_h=int(b), n_measured=int(xb.patientunitstayid.nunique()),
                                 n_first_na150=int(((tna > a) & (tna <= b)).sum())))
                rows_x.append(dict(group=g, anchor=anchor, bin=f"({a},{b}]", denominator=len(s), stays_measured=int(xb.patientunitstayid.nunique()),
                                   Na_tests=len(xb), first_Na150=int(((tna > a) & (tna <= b)).sum())))

    timing(S2, X2, "sgrp", TWO_GROUPS, "T0b")
    timing(S5, X5, "sgrp", USAGE_GROUPS, "T0b")
    S5["vpstop_to_t0b_h"] = S5.gap_vpstop_t0b_h
    mx = int(np.ceil((S5.loc[S5.grp.isin(["VP-first", "VP-first+other"]), "vpstop_to_t0b_h"].max() + 72) / 12) * 12)
    timing(S5, X5, "sgrp", ["VP-first", "VP-first+other"], "VP_stop", shift="vpstop_to_t0b_h", maxh=mx)
    return pd.DataFrame(rows), pd.DataFrame(rows_x)


def sodium_by_polyuria(S2, S5):
    """Sodium >= 150 mmol/L by presence of polyuria in 0-8 h, relative to the onset, and the sodium rise."""
    rows, rows_x = [], []
    for S, groups in ((S2, TWO_GROUPS), (S5, USAGE_GROUPS)):
        for g in groups:
            for ep in (0, 1, "all"):
                s = S[S.sgrp == g] if ep == "all" else S[(S.sgrp == g) & (S.poly8 == ep)]
                k = int(s.na150_72h.sum())
                n = len(s)
                d = s.dNa.dropna()
                r = dict(db=DB, group=g, early_polyuria=ep, n=n, n_na150=k, pct=round(100 * k / n, 1) if n else np.nan,
                         median_dNa=d.median() if len(d) else np.nan, q1_dNa=d.quantile(.25) if len(d) else np.nan, q3_dNa=d.quantile(.75) if len(d) else np.nan)
                if ep != "all":
                    rows.append(r)
                p = s[(s.poly8 == 1) & (s.na150_72h == 1)]
                rows_x.append(dict(r, n_dNa=len(d), Na150_before_onset=int((p.t_na150 < p.uo_onset_h).sum()) if ep != 0 else "",
                                   Na150_at_or_after_onset=int((p.t_na150 >= p.uo_onset_h).sum()) if ep != 0 else "",
                                   median_h_onset_to_Na150=(p.t_na150 - p.uo_onset_h).median() if len(p) else np.nan,
                                   median_h_T0b_to_first_Na150=s.loc[s.na150_72h == 1, "t_na150"].median() if k else np.nan,
                                   Na_tests_0_72_median=s.na_post72_n.median()))
    return pd.DataFrame(rows), pd.DataFrame(rows_x)


def sodium_max_6h(S2, X2, S5, X5):
    """Maximum sodium per 6-h bin after T0b."""
    rows = []
    for S, X, groups in ((S2, X2, TWO_GROUPS), (S5, X5, USAGE_GROUPS)):
        for g in groups:
            ids = set(S.loc[S.sgrp == g, "patientunitstayid"])
            x = X[X.patientunitstayid.isin(ids)]
            for a in range(0, 72, 6):
                mx = x[(x.h > a) & (x.h <= a + 6)].groupby("patientunitstayid").v.max()
                rows.append(dict(group=g, bin=f"({a},{a + 6}]", denominator=len(ids), stays_measured=len(mx), median_maxNa=mx.median() if len(mx) else np.nan,
                                 q1=mx.quantile(.25) if len(mx) else np.nan, q3=mx.quantile(.75) if len(mx) else np.nan, n_ge150=int((mx >= 150).sum())))
    return pd.DataFrame(rows)


def sodium_model(S2, rep):
    """Sodium >= 150 mmol/L ~ group (+ polyuria in 0-8 h) + covariates in the two-group primary population."""
    mres = []
    for lab_, covs in (("Na>=150 ~ VP + COV (reproduction of 2-group adjusted model)", COV), ("Na>=150 ~ VP + 0-8 h polyuria + COV", COV + ["poly8"])):
        row, full = fit_or(S2, "na150_72h", "vp", covs, label=lab_, full=True)
        row["poly8_OR_95CI"] = fmt(*full.loc["poly8", ["OR", "lo", "hi"]]) if "poly8" in full.index else ""
        mres.append(row)
    mres.insert(0, fit_or(S2, "na150_72h", "vp", [], mice=False, label="crude"))
    rep("2-group Na>=150 adjusted OR (expected 2.38)", f"{mres[1]['OR']:.2f}", "2.38")
    tab = pd.DataFrame(mres)
    b0, b1 = np.log(mres[1]["OR"]), np.log(mres[2]["OR"])
    tab["pct_change_logOR_vs_COV_model"] = ["", "", f"{100 * (b1 - b0) / b0:.1f}%"]
    return tab


def order_comparison(POPS, XA, XB, TAB):
    """NE-first vs VP-first: overlap-weighted estimate with the propensity score measured at
    T0a, diagnostics, balance and descriptive tables at T0a and T0b. The descriptive tables are
    added to TAB; returns the estimates, the balance tables per population and the diagnostics."""
    est_rows, smd_out, diag = [], {}, []
    for key, (P, lab_) in POPS.items():
        DA = order_frame(P, XA)
        DB_ = order_frame(P, XB)
        e = order_estimate(DA, NBOOT)
        smds, ess, psr = [], [], []
        for k, imp in enumerate(e.imps):
            w = imp.w
            ess.append([(w[imp.vp == g].sum() ** 2) / (w[imp.vp == g] ** 2).sum() for g in (1, 0)])
            psr.append([imp.ps.min(), imp.ps.max()])
            for c, _l in SMD_VARS:
                x = DA[c].astype(float).values
                ok = ~np.isnan(x)
                smds.append(dict(variable=c, imp=k, w=smd(x[ok], DA.vp.values[ok], w[ok])))
        qb, se, m = e.qb, e.se, e.m
        est_rows.append(dict(db=DB, outcome=key, events_NEfirst=e.k1, n_NEfirst=e.n1, events_VPfirst=e.k0, n_VPfirst=e.n0,
                             crude_or=e.cor, crude_lo=np.exp(np.log(e.cor) - 1.96 * e.cse), crude_hi=np.exp(np.log(e.cor) + 1.96 * e.cse),
                             ato_or=np.exp(qb[0]), ato_lo=np.exp(qb[0] - 1.96 * se[0]), ato_hi=np.exp(qb[0] + 1.96 * se[0]),
                             ato_rd_pct=100 * qb[1], ato_rd_lo=100 * (qb[1] - 1.96 * se[1]), ato_rd_hi=100 * (qb[1] + 1.96 * se[1])))
        diag.append(dict(outcome=key, description=lab_, n_hospitals=int(DA.hospitalid.nunique()), ps_variables=", ".join(PSV), ps_parameters=len(PSV),
                         smaller_group_per_ps_parameter=round(min(e.n1, e.n0) / len(PSV), 1), outcome_events_total=e.k1 + e.k0,
                         outcome_events_per_ps_parameter=round((e.k1 + e.k0) / len(PSV), 1),
                         ps_range=f"{min(r[0] for r in psr):.3f}–{max(r[1] for r in psr):.3f}",
                         ESS_NEfirst_VPfirst=f"{np.mean([x[0] for x in ess]):.0f} / {np.mean([x[1] for x in ess]):.0f}",
                         method=f"MICE m={m} (IterativeImputer; outcome+exposure as auxiliaries); PS logistic; ATO weights (ato_fit); hospital-cluster bootstrap B={NBOOT} per imputation; Rubin (bootstrap within-variance)",
                         missing_at_T0a="; ".join(f"{c} {int(DA[c].isna().sum())}/{len(DA)}" for c in PSV if DA[c].isna().any()),
                         crude_method="logistic, hospital-clustered SE",
                         bootstrap_replicates_ok=f"{min(e.nboot)}–{max(e.nboot)} per imputation",
                         replicates_dropped_nonfinite_logOR=f"{int(np.sum([imp.nonfinite[0] for imp in e.imps]))} of {int(np.sum(e.nboot))} (a resample with 0 events in one group)"))
        S_ = pd.DataFrame(smds).groupby("variable", sort=False).w.mean()
        rows = []
        for c, l_ in SMD_VARS:
            rows.append(dict(db=DB, variable=c, smd_T0a=smd2(DA.loc[DA.vp == 1, c], DA.loc[DA.vp == 0, c]),
                             smd_T0b=smd2(DB_.loc[DB_.vp == 1, c], DB_.loc[DB_.vp == 0, c]), smd_T0a_weighted=round(S_[c], 2)))
        smd_out[key] = pd.DataFrame(rows)
        # descriptive table at T0a and T0b
        drow = []
        for c, l_ in VARS:
            r = {"variable": l_, "key": c}
            for tag, D_ in (("T0a", DA), ("T0b", DB_)):
                for g, gv in (("NE-first", 1), ("VP-first", 0)):
                    x = D_.loc[D_.vp == gv, c]
                    s_ = x.dropna()
                    if set(s_.unique()) <= {0, 1}:
                        r[f"{tag} {g} (n={len(x)})"] = f"{int(s_.sum())}/{len(x)} ({100 * s_.sum() / len(x):.1f}%)"
                    else:
                        r[f"{tag} {g} (n={len(x)})"] = (f"{s_.median():.2f} [{s_.quantile(.25):.2f}–{s_.quantile(.75):.2f}]" if len(s_) else "—") + f"; n={len(s_)}/{len(x)}"
            drow.append(r)
        TAB[f"Order_table_{key}"] = pd.DataFrame(drow)
    return pd.DataFrame(est_rows), smd_out, pd.DataFrame(diag)


def sequence_check(POPS, XA):
    """NE-first: was VP started only after the NE stop (switch)? VP-first: was NE running at the VP stop?"""
    sq = []
    for key, (P, lab_) in POPS.items():
        for g in ORDER_GROUPS:
            x = P[P.grp == g]
            if g == "NE-first":
                late = x.idx_start > x.t0a
                sq.append(dict(population=key, group=g, n=len(x), item="first VP episode started AFTER the NE stop (switch NE->VP, no overlap at T0a)",
                               value=n_pct(int(late.sum()), len(x)), median_h_NEstop_to_VPstart=round(((x.idx_start - x.t0a)[late] / 60).median(), 1)))
            else:
                Xa = XA.set_index("patientunitstayid").loc[x.patientunitstayid]
                sq.append(dict(population=key, group=g, n=len(x), item="NE episode running at the VP stop", value=n_pct(int(Xa.ne_running.sum()), len(x)),
                               median_h_NEstop_to_VPstart=np.nan))
    return pd.DataFrame(sq)


def main():
    warnings.filterwarnings("ignore")
    t_start = time.time()
    log = RunLog()
    say = log.say
    out = out_dir("06")
    TAB = {}
    REPRO = []

    def rep(item, val, target=None):
        REPRO.append(dict(item=item, value=val, target=target, ok=(None if target is None else str(val) == str(target))))
        say("REPRO", item, val, "target", target)

    # ---- cohort ------------------------------------------------------------------------------
    src = load_source(say)
    C, tabs = build_usage_cohort(src, say)
    check_cohort(C, say)

    # =====================================================================================
    # Part 1. Polyuria timing around the VP stop (NE-only: NE stop)
    # =====================================================================================
    anc = anchor_population(C, tabs.ur)
    A = anc.A
    # the anchored series must reproduce the anchored variables of the cohort table
    v = A.grp != "NE-only"
    for mine, theirs_vp, theirs_ne in (("onset_anc", "a_onset_h", "uo_onset_h"), ("pre12_anc", "a_poly_pre12", "poly_pre12"), ("obs24_anc", "a_obs_h_24", "obs_h_24")):
        ref = np.where(v, A[theirs_vp], A[theirs_ne]).astype(float)
        ok = np.isclose(A[mine].astype(float), ref, equal_nan=True).all()
        rep(f"anchored series: {mine} equals usage-group cohort {theirs_vp}/{theirs_ne} in all {len(A)} anchor-population stays", bool(ok), True)

    # hourly prevalence and trajectory, -24..23 h
    VPS_HOURLY = pd.concat([hourly_table(A[A.grp == g], g, A.loc[A.grp == g, "anc_name"].iloc[0], anc) for g in ANCHOR_GROUPS], ignore_index=True)
    VPS_HOURLY.to_csv(os.path.join(out, "vpstop_hourly.csv"), index=False)
    say("vpstop_hourly rows", len(VPS_HOURLY))

    # Aalen-Johansen cumulative incidence 0-24 h with hospital-cluster bootstrap CI
    cif_rows, cif_sens, cnts = [], [], []
    for g in ANCHOR_GROUPS:
        X = A[(A.grp == g) & (A.new_onset_pop == 1)]
        rr, cnt = cif_block(X, g, X.anc_name.iloc[0], "(ii) new-onset (no polyuria -12..-1 h), >=12/24 bins", NBOOT)
        cif_rows += rr
        cnts.append(cnt)
        X = A[(A.grp == g) & (A.urine24_pop == 1)]
        rr, cnt = cif_block(X, g, X.anc_name.iloc[0], "(v) sensitivity: pre-anchor polyuria included, >=12/24 bins", NBOOT)
        cif_sens += rr
        cnts.append(cnt)
    VPS_CIF = pd.DataFrame(cif_rows)
    VPS_CIF.to_csv(os.path.join(out, "vpstop_cif.csv"), index=False)
    VPS_CIF_SENS = pd.DataFrame(cif_sens)
    VPS_CIF_SENS.to_csv(os.path.join(out, "vpstop_cif_sens_incl_pre_polyuria.csv"), index=False)
    TAB["VPstop_counts"] = pd.DataFrame(cnts)
    say(TAB["VPstop_counts"][["population", "group", "n", "crude_onset_0_8", "AJ_events_0_8", "CIF8_pct", "CIF24_pct"]].to_string(index=False))
    for g, k, n in (("VP-first", 9, 181), ("NE-first", 34, 289)):
        X = A[(A.grp == g) & (A.new_onset_pop == 1)]
        rep(f"VP-stop 0-8 h new-onset polyuria {g} (expected {k}/{n})", f"{int((X.onset_anc <= 8).sum())}/{len(X)}", f"{k}/{n}")
    x = A[(A.grp == "NE-only") & (A.new_onset_pop == 1)]
    rep("NE-stop anchor: NE-only new-onset population = 5-group NE-only analysis population (3,422; IDs identical)",
        f"{len(x)}; IDs identical {set(x.patientunitstayid) == set(C.loc[(C.primary == 1) & (C.grp == 'NE-only'), 'patientunitstayid'])}", "3422; IDs identical True")
    TAB["VPstop_flow"] = vpstop_flow(C, A)
    VPS_T0B, TAB["VPstop_T0b_position_VPfirst"] = t0b_position(C, A, rep)
    VPS_T0B.to_csv(os.path.join(out, "vpstop_t0b_hist.csv"), index=False)
    say(TAB["VPstop_T0b_position_VPfirst"].to_string(index=False))

    # =====================================================================================
    # Part 2. Hypernatraemia timing vs early polyuria (T0b anchor)
    # =====================================================================================
    NA = tabs.na[["patientunitstayid", "h", "v"]].copy()          # sodium 100-200 mmol/L, h from T0b
    cohort2 = pd.read_parquet(cohort_path(COHORT_PRIMARY))
    P2 = cohort2[cohort2.primary == 1].copy()
    chk = P2[["patientunitstayid", "t0"]].merge(C[["patientunitstayid", "t0"]], on="patientunitstayid", suffixes=("_two", "_usage"))
    assert len(chk) == len(P2) and (chk.t0_two == chk.t0_usage).all()
    rep("2-group primary n (VP/NE)", f"{int((P2.grp == 'VP').sum())}/{int((P2.grp == 'NE').sum())}", "370/3422")
    P2["sgrp"] = P2.grp.map({"VP": "2grp_VP", "NE": "2grp_NE"})
    PR = C[C.primary == 1].copy()
    PR["sgrp"] = PR.grp
    S2, X2 = sodium_frame(P2, NA)
    S5, X5 = sodium_frame(PR[PR.grp.isin(USAGE_GROUPS)], NA)
    rep("2-group Na denominator & events VP", f"{int(S2.loc[S2.grp == 'VP', 'na150_72h'].sum())}/{int((S2.grp == 'VP').sum())}")
    rep("2-group Na denominator & events NE", f"{int(S2.loc[S2.grp == 'NE', 'na150_72h'].sum())}/{int((S2.grp == 'NE').sum())}")
    assert (S2.t_na150.notna() == (S2.na150_72h == 1)).all() and (S5.t_na150.notna() == (S5.na150_72h == 1)).all()
    NA_TIMING, TAB["Na_i_timing_detail"] = sodium_timing(S2, X2, S5, X5)
    NA_TIMING.to_csv(os.path.join(out, "na_timing.csv"), index=False)
    NA_BY_POLY, TAB["Na_ii_na_by_poly_detail"] = sodium_by_polyuria(S2, S5)
    NA_BY_POLY.to_csv(os.path.join(out, "na_by_poly.csv"), index=False)
    say(TAB["Na_ii_na_by_poly_detail"][["group", "early_polyuria", "n", "n_na150", "pct", "median_dNa", "Na150_before_onset", "Na150_at_or_after_onset"]].to_string(index=False))
    TAB["Na_iii_maxNa_6h"] = sodium_max_6h(S2, X2, S5, X5)
    TAB["Na_iv_model"] = sodium_model(S2, rep)
    say(TAB["Na_iv_model"][["analysis", "n", "events", "events_exposed", "n_exposed", "aOR_95CI", "poly8_OR_95CI"]].to_string(index=False))

    # =====================================================================================
    # Part 3. NE-first vs VP-first: background at T0a (the decision point) and one overlap-weighted estimate
    # =====================================================================================
    ORD = C[C.grp.isin(ORDER_GROUPS)].copy()
    S = order_sources(C, tabs, src)
    XA = cov_at(ORD, ORD.t0a.values, S)
    XB = cov_at(ORD, ORD.t0.values, S)
    # the re-measurement at T0b must equal the variables of the usage-group and fluid-balance cohorts
    v = XB.merge(ORD, on="patientunitstayid", suffixes=("", "_cohort"))
    for mine, theirs in (("press_h", "press_h_pre"), ("other_press", "other_press_pre"), ("inotrope", "inotrope_t0"), ("map6", "map_base"),
                         ("lactate_max24", "lactate_pre24"), ("na_last", "na_pre"), ("kdigo_cat", "kdigo_cat_cohort"), ("uo_pre6", "uo_pre6_rate"),
                         ("mv", "mv_t0"), ("steroid", "steroid_pre24")):
        a, b = v[mine], v[theirs]
        ok = (a.astype(str) == b.astype(str)).values if a.dtype == object else np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        rep(f"cov_at(T0b) {mine} == usage-group cohort {theirs.removesuffix('_cohort')}: stays agreeing (NE-first+VP-first)", f"{int(ok.sum())}/{len(v)}", f"{len(v)}/{len(v)}")
    f = XB[["patientunitstayid", "fo_pct"]].merge(pd.read_parquet(cohort_path(COHORT_FLUID))[["patientunitstayid", "fo_pct"]], on="patientunitstayid", suffixes=("", "_fluid"))
    rep(f"cov_at(T0b) fo_pct == fluid-balance cohort fo_pct (n={len(f)})", bool(np.isclose(f.fo_pct, f.fo_pct_fluid, equal_nan=True).all()), True)

    POP1 = A[A.grp.isin(ORDER_GROUPS) & (A.new_onset_pop == 1)].assign(_y=lambda d: (d.onset_anc <= 8).astype(int))
    POP2 = C[(C.primary == 1) & C.grp.isin(ORDER_GROUPS)].assign(_y=lambda d: d.poly8.astype(int))
    POPS = {"1_VPstop_0_8h": (POP1, "new-onset polyuria 0-8 h after the VP stop (VP-stop new-onset population)"),
            "2_T0b_0_8h": (POP2, "new-onset polyuria 0-8 h after T0b (5-group analysis population)")}
    ORDER_EST, smd_out, DIAG = order_comparison(POPS, XA, XB, TAB)
    ORDER_EST.to_csv(os.path.join(out, "order_estimates.csv"), index=False)
    smd_out["2_T0b_0_8h"].to_csv(os.path.join(out, "order_smd.csv"), index=False)
    smd_out["1_VPstop_0_8h"].to_csv(os.path.join(out, "order_smd_pop_VPstop.csv"), index=False)
    TAB["Order_estimates"] = ORDER_EST
    TAB["Order_diagnostics"] = DIAG
    TAB["Order_smd_pop2_T0b"] = smd_out["2_T0b_0_8h"]
    TAB["Order_smd_pop1_VPstop"] = smd_out["1_VPstop_0_8h"]
    say(ORDER_EST.round(3).to_string(index=False))
    say(TAB["Order_diagnostics"].to_string(index=False))
    TAB["Order_sequence_check"] = sequence_check(POPS, XA)
    say(TAB["Order_sequence_check"].to_string(index=False))

    # =====================================================================================
    # write
    # =====================================================================================
    NOTE = pd.DataFrame({"note": [
        f"06_vpstop_timing.py, eICU-CRD v2.0; cohort rebuilt with eicu_cohort.build_usage_cohort, identical to {COHORT_USAGE}; FO% = functions of eicu_fluid.py.",
        "All analyses post hoc / exploratory. Every count is shown as n/N in the tables/CSVs.",
        "Anchor: VP stop (first VP episode end) for VP groups; NE stop (= T0) for NE-only. Anchor population: adult, no RRT, no pituitary/DI/brain-death code, alive & in ICU at the anchor. NE-only additionally requires no other vasopressor running/recorded within 60 min before the NE stop (press_t0 == 0 of the usage-group cohort, i.e. the NE stop is liberation); its new-onset population is then identical to the 5-group NE-only analysis population (3,422). VP groups at the VP stop are not restricted (another vasopressor, e.g. NE in VP-first, may continue), as in the VP-stop anchored table of the usage-group analysis.",
        "Hourly table: bin h = floor((charting offset - anchor)/60); n_alive_in_icu = ICU admission <= bin start < unit discharge (ICU death = unit discharge); n_in_run = bins >3 mL/kg/h adjacent to another recorded >3 bin (part of a >=2-bin run); no new-onset rule.",
        "CIF: Aalen-Johansen (aj_cif), competing = ICU death, ICU discharge alive, vasopressor restart after liberation from all vasopressors (time from anchor = T0b-anchor + restart_h); ties -> competing (cr_times). n_at_risk(h) = #(t >= h); n_events = cumulative polyuria events by h; 95% CI = hospital-cluster bootstrap percentile.",
        "'Crude' 0-8 h counts ignore competing events (= definition of the VP-stop anchored table of the usage-group analysis, used for the expected values); 'AJ events' count only polyuria before a competing event.",
        "Na denominator: all Na < 150 in the 24 h before T0b and >= 1 Na in (0, 72] h. n_measured = stays with >= 1 Na in the bin (test counts in xlsx). Early polyuria = new-onset polyuria 0-8 h after T0b (poly8). dNa = max Na (0,72] - last Na (-24,0].",
        "Na model: logistic, hospital-clustered SE, MICE m=10 (fit_or), COV of the two-group primary analysis (21 parameters) +/- poly8.",
        "Stop-order comparison: exposure coded NE-first = 1 vs VP-first = 0 (OR > 1 = more polyuria in NE-first). PS = 10 pre-specified variables measured at T0a; NE/VP infusion rates at T0a are not in the PS (they describe the drug being stopped vs the drug continuing, i.e. part of the exposure definition).",
        "Stop-order comparison: covariates at T0a re-use the rules of the usage-group cohort with the anchor replaced; reproduced at T0b against the usage-group and fluid-balance cohort values (na_last: 1308/1310, 2 same-timestamp ties)."]})
    with pd.ExcelWriter(os.path.join(out, "eICU_vpstop_timing_results.xlsx")) as xw:
        NOTE.to_excel(xw, sheet_name="00_notes", index=False)
        pd.DataFrame(REPRO).to_excel(xw, sheet_name="01_reproduction", index=False)
        for k in TAB:
            TAB[k].to_excel(xw, sheet_name=k[:31], index=False)
        for nm, df in (("csv_vpstop_hourly", VPS_HOURLY), ("csv_vpstop_cif", VPS_CIF), ("csv_vpstop_cif_sens", VPS_CIF_SENS), ("csv_vpstop_t0b_hist", VPS_T0B), ("csv_na_timing", NA_TIMING),
                       ("csv_na_by_poly", NA_BY_POLY), ("csv_order_smd", smd_out["2_T0b_0_8h"]), ("csv_order_smd_pop_VPstop", smd_out["1_VPstop_0_8h"]), ("csv_order_estimates", ORDER_EST)):
            df.to_excel(xw, sheet_name=nm[:31], index=False)
    log.write(os.path.join(out, RUN_LOG))
    assert all(r["ok"] in (True, None) for r in REPRO if not r["item"].startswith("cov_at(T0b) na_last")), "reproduction failed"
    say("done in %.0f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
