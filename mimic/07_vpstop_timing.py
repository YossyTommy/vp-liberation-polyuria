#!/usr/bin/env python3
"""
07_vpstop_timing.py - MIMIC-IV v3.1, timing of polyuria and hypernatremia relative to the
vasopressin stop, by stop order; switch sensitivity and closed cohort.

Exploratory, post hoc, mainly descriptive. Stop order: NE-first (norepinephrine stopped first,
vasopressin last) vs VP-first (vasopressin stopped first, norepinephrine last).

Part 1 - timing_analyses()
  (1) polyuria around the VP stop (NE-only: NE stop): hourly prevalence of stays in a polyuria
      run, Aalen-Johansen cumulative incidence of new-onset polyuria, position of T0b
  (2) timing of Na >= 150 relative to early polyuria (anchor T0b)
  (3) NE-first vs VP-first with covariates re-measured at T0a: overlap-weighted (ATO) estimate
Part 2 - main()  (runs part 1 and re-uses its objects; no group, population or outcome is
  re-defined)
  (A) NE->VP "switch" stays: counts and estimates without them
  (B) closed cohort: alive and in the ICU from -12 to +8 h around the anchor with urine recorded
      in >= 6 of 12 bins before and >= 4 of 8 bins after (strict: >= 10/12 and >= 7/8)
  (C) number-at-risk table of the VP-stop new-onset population
  (D) size of the basic population and number of stays with urine recorded at selected hours

Shared code (definitions are not re-implemented)
  mimic_common.py   extraction, episodes (1 h), run_onset, build() (covariates at T0)
  mimic_models.py   MICE, glm_fit (subject-clustered SE), or_mi, rubin, event_data, aj
  mimic_groups.py   classify() (ugrp, T0a, T0a(VP), T0b), anchor_cols(), shifted_anchors(),
                    extra_exposure(), group_models()
  mimic_fluid.py    prep_io(), fluid_vars() (cumulative FO%), ato_analysis() (overlap weighting,
                    subject-level cluster bootstrap), wsmd()
  Models on imputed data use with_imp of mimic_models.py; only the overlap-weighted stop-order
  comparison uses with_imp8 of this script (imputed FO% and urine output of the last 6 h added).

Inputs
  WORK_DIR/mimic/cohort/   analysis-ready tables of 02_primary_analysis.py (T0 = end of the
                           first index episode = VP stop in the VP group), 05_usage_groups.py
                           (T0 = T0b; ugrp, t0a, t0a_vp, t0b, pop_main) and 06_fluid_balance.py
                           (FO% at T0b)
  WORK_DIR/mimic/data, ext   csv extracts and caches; ext/oe_pressorstays.parquet (outputevents
                           of vasopressor-treated stays) is created here if missing

Definitions (unchanged)
  Polyuria   > 3 mL/kg/h in >= 2 consecutive recorded hourly bins (an unrecorded bin resets the
             run; onset = end of the second bin). Hourly bin = floor((charttime - anchor) / 1 h),
             summed volume / body weight.
  Basic exclusions   adult, anchor inside the ICU stay, alive at the anchor, no renal replacement
             therapy, no pituitary/hypothalamic/diabetes insipidus/brain-death/pituitary-surgery code.
  NE-only    as in the primary analysis (anchor = end of the first NE episode, no other
             vasopressor running).
  New-onset population   basic exclusions + urine recorded in >= 12 of 24 bins after the anchor +
             no polyuria in the 12 h before the anchor.
  Switch     NE->VP: NE-first and the first VP episode starts after T0a (= end of the last NE
             episode before T0b), i.e. VP was started after NE had been stopped, with no overlap.
             VP->NE: VP-first and the first NE episode starts after T0a (= end of the first VP
             episode).
  Competing risks   death, ICU discharge, vasopressor restart (event_data; ties go to polyuria).
             CI = percentiles of a subject-level cluster bootstrap.

Paths come from config.py. Optional environment variables: NBOOT (default 200), M_IMP (10).
Outputs (OUT_DIR/mimic_07/)
  results_mimic_vpstop_timing.xlsx, vpstop_hourly.csv, vpstop_cif.csv, vpstop_t0b_hist.csv,
  na_timing.csv, na_by_poly.csv, order_smd.csv, order_estimates.csv, switch_sensitivity.csv,
  closed_cohort.csv, closed_cohort_hourly.csv, risk_table.csv, basic_population_n.csv,
  reproduction_checks.csv, key_results.json, run_log_part1.txt, run_log.txt

Run:  python mimic/07_vpstop_timing.py   (after scripts 02, 05 and 06)
"""
import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # folder of the shared modules
import mimic_common as mc  # noqa: E402
import mimic_models as mm  # noqa: E402
from mimic_common import config, VP, NE, hrs  # noqa: E402
from mimic_groups import G4, classify, shifted_anchors, extra_exposure, group_models  # noqa: E402
from mimic_fluid import prep_io, fluid_vars, ato_analysis, wsmd  # noqa: E402

warnings.filterwarnings("ignore")

NBOOT = int(os.environ.get("NBOOT", 200))
DBNAME = "MIMIC-IV"
say = mc.RunLog()


# ===========================================================================
# Formatting helpers
def nN(k, n):
    return "%d/%d (%.1f%%)" % (k, n, 100 * k / n if n else np.nan)


def miqr(x, d=1):
    x = pd.Series(x).dropna()
    if not len(x):
        return ""
    f = "%%.%df [%%.%df–%%.%df]" % (d, d, d)
    return f % (x.median(), x.quantile(.25), x.quantile(.75))


# ===========================================================================
# Hourly urine bins (rule of mimic_common.hourly_bins with a wider window)
def bins_window(u, anc, lo=-26, hi=72):
    m = u.merge(anc[["stay_id", "t0", "weight"]], on="stay_id", how="inner")
    m["h"] = np.floor((m.charttime - m.t0).dt.total_seconds() / 3600)
    m = m[(m.h >= lo) & (m.h < hi) & (m.weight > 0)]
    hr = m.groupby(["stay_id", "h"]).agg(ml=("value", "sum"), wt=("weight", "first")).reset_index()
    hr["rate"] = hr.ml / hr.wt
    hr["h"] = hr.h.astype(int)
    return hr


def recheck_outcomes(hr, ref):
    """Recompute obs24 / pre_poly / onset from bins_window with run_onset and compare with the cohort table."""
    rows = []
    for sid, g in hr.groupby("stay_id"):
        rr = dict(zip(g.h, g.rate))
        on = mc.run_onset(rr, 0, 72, 3.0)
        rows.append((sid, np.nan if on is None else on, int(mc.run_onset(rr, -12, 0, 3.0) is not None),
                     sum(1 for k in rr if 0 <= k < 24)))
    r = pd.DataFrame(rows, columns=["stay_id", "onset_chk", "pre_poly_chk", "obs24_chk"]).set_index("stay_id")
    x = ref.set_index("stay_id")[["onset", "pre_poly", "obs24"]].join(r, how="left")
    x["obs24_chk"] = x.obs24_chk.fillna(0)
    x["pre_poly_chk"] = x.pre_poly_chk.fillna(0)
    mis = dict(onset=int((x.onset.fillna(-1) != x.onset_chk.fillna(-1)).sum()),
              pre_poly=int((x.pre_poly != x.pre_poly_chk).sum()), obs24=int((x.obs24 != x.obs24_chk).sum()), n=len(x))
    return mis


def in_run_flags(hr):
    """Whether a bin belongs to a run of >= 2 consecutive recorded bins > 3 mL/kg/h."""
    hr = hr.sort_values(["stay_id", "h"]).copy()
    hr["gt3"] = hr.rate > 3
    prev_h = hr.groupby("stay_id").h.shift(1)
    prev_g = hr.groupby("stay_id").gt3.shift(1)
    new_run = ~(hr.gt3 & (prev_h == hr.h - 1) & (prev_g == True))
    hr["run_id"] = new_run.cumsum()
    ln = hr[hr.gt3].groupby("run_id").size()
    hr["in_run"] = hr.gt3 & hr.run_id.map(ln).fillna(0).ge(2)
    return hr


def hourly_table(pop, hr, group, anchor, hours=range(-24, 24)):
    """Per hourly bin: stays in the ICU, stays with urine recorded, bins > 3 mL/kg/h, bins in a run, urine output."""
    ids = set(pop.stay_id)
    h = hr[hr.stay_id.isin(ids)]
    p = pop.set_index("stay_id")
    rows = []
    for k in hours:
        bs = p.t0 + pd.Timedelta(hours=k)
        alive = (p.intime <= bs) & (p.outtime > bs) & (p.deathtime.isna() | (p.deathtime > bs))
        x = h[h.h == k]
        rows.append(dict(db=DBNAME, group=group, anchor=anchor, hour=k, n_alive_in_icu=int(alive.sum()),
                         n_recorded=int(x.stay_id.nunique()), n_uo_gt3=int((x.rate > 3).sum()),
                         n_in_run=int(x.in_run.sum()),
                         uo_median=x.rate.median() if len(x) else np.nan,
                         uo_q1=x.rate.quantile(.25) if len(x) else np.nan,
                         uo_q3=x.rate.quantile(.75) if len(x) else np.nan,
                         uo_mean=x.rate.mean() if len(x) else np.nan))
    return rows


# ===========================================================================
# Aalen-Johansen (event_data / aj of mimic_models) with a subject-level cluster bootstrap
GRID = np.arange(0, 25)


def cif_table(x, group, anchor, nboot=NBOOT, seed=2026):
    T, C = mm.event_data(x, horizon=72.0)
    o = mm.aj(T, C, grid=GRID, causes=(1, 2, 3, 4))
    subj = x.subject_id.values
    pos = pd.Series(np.arange(len(x))).groupby(subj).apply(np.array).to_dict()
    us = np.unique(subj)
    rng = np.random.RandomState(seed)
    B = []
    for b in range(nboot):
        idx = np.concatenate([pos[s] for s in rng.choice(us, len(us), replace=True)])
        B.append(mm.aj(T[idx], C[idx], grid=GRID, causes=(1,))[1])
    B = np.array(B)
    lo, hi = np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)
    rows = []
    for i, g in enumerate(GRID):
        ev = int(((C == 1) & (T > g - 1) & (T <= g)).sum()) if g > 0 else int(((C == 1) & (T <= 0)).sum())
        rows.append(dict(db=DBNAME, group=group, anchor=anchor, time_h=int(g), n_at_risk=int(o["nrisk"][i]),
                         n_events=ev, cif=o[1][i], lo=lo[i], hi=hi[i]))
    summ = dict(db=DBNAME, group=group, anchor=anchor, N=len(x), patients=int(x.subject_id.nunique()),
                aj_events_0_8=int(((C == 1) & (T <= 8)).sum()), aj_events_8_24=int(((C == 1) & (T > 8) & (T <= 24)).sum()),
                at_risk_8h=int((T > 8).sum()),
                crude_poly8=nN(int(x.poly8.sum()), len(x)), crude_poly_8_24=nN(int(x.poly_8_24.sum()), len(x)),
                crude_poly24=nN(int(x.poly24.sum()), len(x)),
                cif8="%.1f%% (%.1f–%.1f)" % (100 * o[1][8], 100 * lo[8], 100 * hi[8]),
                cif24="%.1f%% (%.1f–%.1f)" % (100 * o[1][24], 100 * lo[24], 100 * hi[24]),
                death_by24=int(((C == 2) & (T <= 24)).sum()), icuout_by24=int(((C == 3) & (T <= 24)).sum()),
                restart_by24=int(((C == 4) & (T <= 24)).sum()), nboot=nboot)
    return rows, summ


# ===========================================================================
# Covariates at T0a
def rate_before(v, item, t0, weight):
    """Infusion rate just before t0 (last rate > 0 row with starttime < t0). NE: mcg/kg/min, VP: U/min;
    unconvertible units give NaN plus an indicator."""
    x = v[(v.itemid == item) & (v.rate > 0) & v.stay_id.isin(t0.index)].copy()
    x["t0"] = x.stay_id.map(t0)
    x = x[x.starttime < x.t0].sort_values(["stay_id", "starttime", "endtime"])
    last = x.groupby("stay_id").tail(1).set_index("stay_id")
    if item == NE:
        r = np.where(last.rateuom == "mcg/kg/min", last.rate, np.nan)
    else:
        r = np.where(last.rateuom == "units/hour", last.rate / 60, np.where(last.rateuom == "units/min", last.rate, np.nan))
    out = pd.Series(r, index=last.index).reindex(t0.index)
    unconv = pd.Series(np.isnan(r), index=last.index).reindex(t0.index).fillna(False).astype(int)
    running = last.endtime.reindex(t0.index) > t0      # row continues beyond t0 = running at t0
    return out, unconv, running.fillna(False).astype(int)


def build_at_t0a(data, c, ep):
    """build() of mimic_common with T0 of VP-containing stays replaced by T0a (NE-only unchanged)."""
    v = data["v"]
    d_orig = mc.anchors(v, 1.0).set_index("stay_id")
    dA = shifted_anchors(v, ep, d_orig, c.set_index("stay_id").t0a)
    DA, _, _ = mc.build(data, 1.0, anchor_table=dA)
    return DA


IMPV8 = ["kdigo_pre", "na_pre", "map_base", "log_lact", "fo_pct", "uo_pre6_mlkgh"]
AUX8 = ["z", "age10", "male", "cvicu", "diur_pre24", "fluid_pre24_l", "log_press_h", "ne_peak_c", "other_vaso_pre",
        "vent_t0", "steroid_pre24", "inotrope_t0", "sepsis", "card_surg", "emergency", "y_vpstop", "y_t0b"]


def with_imp8(d, imp):
    """with_imp of mimic_models plus the imputed KDIGO stage as a number, FO% and log urine output
    of the 6 h before the anchor (variables of the propensity model PS8)."""
    x = mm.with_imp(d, imp)
    x["kdigo_n"] = x.kdigo_i.astype(float)
    if "fo_pct" in imp.columns:
        x["fo_i"] = imp.loc[d.index, "fo_pct"]
    if "uo_pre6_mlkgh" in imp.columns:
        x["luo6_i"] = np.log1p(imp.loc[d.index, "uo_pre6_mlkgh"].clip(lower=0))
    return x


def impute8(d):
    """impute() of mimic_models with the imputed and auxiliary variables of the stop-order comparison."""
    return mm.impute(d, impvars=IMPV8, auxv=AUX8)


PS8 = "age10+card_surg+sepsis+log_press_h+ne_peak_c+map_i+lact_i+kdigo_n+luo6_i+fo_i"
SMD8 = [("age10", "age, /10 y"), ("male", "male sex"), ("cvicu", "CVICU"), ("card_surg", "cardiac surgery"),
        ("sepsis", "sepsis (ICD)"), ("press_h", "vasopressor hours from first start to anchor"),
        ("log_press_h", "log(1+vasopressor h) [PS]"),
        ("ne_rate", "NE rate just before anchor, mcg/kg/min"), ("vp_rate", "VP rate just before anchor, U/min"),
        ("ne_peak_c", "NE peak before anchor, /0.1 mcg/kg/min [PS]"), ("vp_dur_h", "VP infusion hours before anchor"),
        ("other_vaso_pre", "other vasopressor before anchor"), ("map_base", "MAP median 6 h before anchor (observed)"),
        ("map_i", "MAP (imputed) [PS]"), ("lactate_pre24", "lactate max 24 h before anchor (observed)"),
        ("lact_i", "lactate (imputed) [PS]"), ("na_pre", "Na last 24 h before anchor (observed)"),
        ("kdigo_pre", "KDIGO pre (observed, 0–3)"), ("kdigo_n", "KDIGO pre (imputed, 0–3) [PS]"), ("uo_pre6_mlkgh", "urine output 6 h before anchor, mL/kg/h (observed)"),
        ("luo6_i", "log(1+UO 6 h) (imputed) [PS]"), ("fo_pct", "FO% ICU admission→anchor (observed)"),
        ("fo_i", "FO% (imputed) [PS]"), ("vent_t0", "invasive ventilation at anchor"), ("steroid_pre24", "systemic steroid 24 h before anchor"),
        ("inotrope_t0", "inotrope (dobutamine/milrinone) at anchor"), ("diur_pre24", "loop diuretic 24 h before anchor"),
        ("fluid_pre24_l", "IV fluid 24 h before anchor, L")]


def smd_un(a, b):
    a, b = pd.Series(a, dtype=float).dropna(), pd.Series(b, dtype=float).dropna()
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sd if sd > 0 else 0.0


# ===========================================================================
# Part 1: timing of polyuria and hypernatremia, stop-order comparison
# ===========================================================================
def timing_analyses(outdir):
    """Part 1. Returns the objects that part 2 re-uses."""
    t_all = time.time()
    SH, KEY = {}, {}
    data = mc.load_all(say)
    v, u = data["v"], data["u"]
    D0 = pd.read_parquet(mc.cohort_file(mc.PRIMARY_TABLE))
    DB = pd.read_parquet(mc.cohort_file(mc.USAGE_TABLE))
    say("parquets", D0.shape, DB.shape)
    gmap = DB.set_index("stay_id").ugrp
    D0["ugrp"] = D0.stay_id.map(gmap)
    D0.loc[D0.grp == "NE", "ugrp"] = "NE-only"
    for col in ("t0a", "t0a_vp", "t0b", "gap_vp_t0b_h", "gap_t0a_t0b_h", "other_after_vp"):
        D0[col] = D0.stay_id.map(DB.set_index("stay_id")[col])

    # ---- classification recomputed (classify of mimic_groups) and compared with the cohort table ----
    c, ep = classify(v)
    cc = c.set_index("stay_id")
    dbv = DB[DB.grp == "VP"].set_index("stay_id")
    chk_cls = dict(n=len(dbv), ugrp_mismatch=int((cc.ugrp.reindex(dbv.index) != dbv.ugrp).sum()),
                   t0a_mismatch=int((cc.t0a.reindex(dbv.index) != dbv.t0a).sum()),
                   t0b_mismatch=int((cc.t0b.reindex(dbv.index) != dbv.t0b).sum()))
    say("classification recheck vs parquet:", chk_cls)

    # ---- populations ----------------------------------------------------------------
    base0 = (D0.ex_age == 0) & (D0.ex_notinicu == 0) & (D0.ex_deadpre == 0) & (D0.rrt_any == 0) & (D0.ex_neuroendo == 0)
    base0 &= (D0.grp == "VP") | (D0.vaso_at_t0 == 0)
    D0["vpstop_base"] = base0.astype(int)
    D0["vpstop_eval"] = (base0 & (D0.obs24 >= 12)).astype(int)
    D0["vpstop_main"] = (base0 & (D0.obs24 >= 12) & (D0.pre_poly == 0)).astype(int)
    VPS_GROUPS = [("NE-first", "VP_stop"), ("VP-first", "VP_stop"), ("VP-only", "VP_stop"), ("NE-only", "NE_stop"),
           ("Simultaneous", "VP_stop")]
    flow = []
    for g, a in VPS_GROUPS:
        s = D0[D0.ugrp == g]
        flow.append(dict(group=g, anchor=a, classified=len(s), base_alive_in_icu=int(s.vpstop_base.sum()),
                         urine_ge12_of_24=int(s.vpstop_eval.sum()), new_onset_rule=int(s.vpstop_main.sum()),
                         patients_main=int(s[s.vpstop_main == 1].subject_id.nunique())))
    # T0b reference (five-group analysis population)
    MAINB = DB[(DB.pop_main == 1) & DB.ugrp.isin(G4)].copy()
    for g in ("NE-first", "VP-first"):
        s = DB[DB.ugrp == g]
        flow.append(dict(group=g, anchor="T0b", classified=len(s), base_alive_in_icu=int(s.pop_base.sum()),
                         urine_ge12_of_24=int(s.pop_eval.sum()), new_onset_rule=int(s.pop_main.sum()),
                         patients_main=int(s[s.pop_main == 1].subject_id.nunique())))
    SH["VPstop_00_flow"] = pd.DataFrame(flow)
    say(SH["VPstop_00_flow"].to_string())

    # ---- reproduction checks --------------------------------------------------------
    rep = []
    exp = {"VP-first": (71, 576), "NE-first": (85, 570)}
    for g, (k, n) in exp.items():
        s = D0[(D0.ugrp == g) & (D0.vpstop_main == 1)]
        rep.append(dict(check=f"VP-stop anchor 0–8 h polyuria, {g} (script 05, sheet 09)", value="%d/%d" % (s.poly8.sum(), len(s)),
                        expected="%d/%d" % (k, n), match=bool((s.poly8.sum(), len(s)) == (k, n))))
    vfm = MAINB[MAINB.ugrp == "VP-first"]
    rep.append(dict(check="VP-first: VP stop → T0b median (T0b analysis population)", value="%.1f h" % vfm.gap_vp_t0b_h.median(),
                    expected="15.1 h", match=bool(round(vfm.gap_vp_t0b_h.median(), 1) == 15.1)))
    rep.append(dict(check="classify() rerun vs cohort table of script 05 (ugrp/t0a/t0b mismatches among VP stays)",
                    value=json.dumps(chk_cls), expected="0/0/0",
                    match=bool(chk_cls["ugrp_mismatch"] == 0 and chk_cls["t0a_mismatch"] == 0 and chk_cls["t0b_mismatch"] == 0)))
    n5 = MAINB.ugrp.value_counts().to_dict()
    rep.append(dict(check="5-group T0b analysis N", value=json.dumps(n5),
                    expected='{"NE-only": 4667, "NE-first": 560, "VP-first": 456, "VP-only": 203, "Simultaneous": 190}',
                    match=n5 == {"NE-only": 4667, "NE-first": 560, "VP-first": 456, "VP-only": 203, "Simultaneous": 190}))

    # ---- (1) hourly urine bins ------------------------------------------------------
    pop0 = D0[D0.vpstop_base == 1]
    hr0 = bins_window(u, pop0[["stay_id", "t0", "weight"]], lo=-26, hi=72)
    mis0 = recheck_outcomes(hr0, pop0[["stay_id", "onset", "pre_poly", "obs24"]])
    rep.append(dict(check="hourly bins rebuilt (−26..72 h) → onset/pre_poly/obs24 vs cohort table of script 02 (mismatching stays)",
                    value=json.dumps(mis0), expected="0/0/0", match=bool(mis0["onset"] == 0 and mis0["pre_poly"] == 0 and mis0["obs24"] == 0)))
    hr0 = in_run_flags(hr0)
    popB = DB[(DB.pop_base == 1) & DB.ugrp.isin(["NE-first", "VP-first"])]
    hrB = in_run_flags(bins_window(u, popB[["stay_id", "t0", "weight"]], lo=-26, hi=72))
    misB = recheck_outcomes(hrB, popB[["stay_id", "onset", "pre_poly", "obs24"]])
    rep.append(dict(check="hourly bins at T0b (NE-first/VP-first) vs cohort table of script 05", value=json.dumps(misB), expected="0/0/0",
                    match=bool(misB["onset"] == 0 and misB["pre_poly"] == 0 and misB["obs24"] == 0)))
    SH["00_reproduction"] = pd.DataFrame(rep)
    say(SH["00_reproduction"].to_string())

    hourly, hourly_main = [], []
    for g, a in VPS_GROUPS:
        hourly += hourly_table(pop0[pop0.ugrp == g], hr0, g, a)
        hourly_main += hourly_table(D0[(D0.vpstop_main == 1) & (D0.ugrp == g)], hr0, g, a + "|newonset_population")
    for g in ("NE-first", "VP-first"):
        hourly += hourly_table(popB[popB.ugrp == g], hrB, g, "T0b")
    H = pd.DataFrame(hourly)
    H.to_csv(os.path.join(outdir, "vpstop_hourly.csv"), index=False)
    SH["VPstop_i_iv_hourly"] = H
    SH["VPstop_i_iv_hourly_newonset_pop"] = pd.DataFrame(hourly_main)

    # ---- (1) cumulative incidence ---------------------------------------------------
    cif_rows, summ = [], []
    for g, a in VPS_GROUPS:
        x = D0[(D0.vpstop_main == 1) & (D0.ugrp == g)]
        r, s = cif_table(x, g, a)
        cif_rows += r
        summ.append(s)
    for g in ("NE-first", "VP-first"):
        x = MAINB[MAINB.ugrp == g]
        r, s = cif_table(x, g, "T0b")
        cif_rows += r
        summ.append(s)
    for g, a in VPS_GROUPS:   # sensitivity: polyuria in the 12 h before the anchor not excluded
        x = D0[(D0.vpstop_eval == 1) & (D0.ugrp == g)]
        r, s = cif_table(x, g, a + "_sens_incl_prepoly")
        s["pre_poly_included"] = int(x.pre_poly.sum())
        cif_rows += r
        summ.append(s)
    CIF = pd.DataFrame(cif_rows)
    CIF.to_csv(os.path.join(outdir, "vpstop_cif.csv"), index=False)
    SH["VPstop_ii_v_cif_hourly"] = CIF
    SH["VPstop_ii_v_cif_summary"] = pd.DataFrame(summ)
    say(SH["VPstop_ii_v_cif_summary"].to_string())

    # ---- (1) VP-first: position of T0b -----------------------------------------------
    vf = D0[(D0.vpstop_main == 1) & (D0.ugrp == "VP-first")]
    hh = np.floor(vf.gap_vp_t0b_h).astype(int)
    T0H = hh.value_counts().sort_index().rename_axis("hour_from_vpstop").reset_index(name="n")
    T0H.insert(0, "db", DBNAME)
    T0H.to_csv(os.path.join(outdir, "vpstop_t0b_hist.csv"), index=False)
    vfb = D0[(D0.vpstop_base == 1) & (D0.ugrp == "VP-first")]
    q = vf.gap_vp_t0b_h
    SH["VPstop_iii_T0b_position"] = pd.DataFrame([
        dict(population="VP-first, VP-stop new-onset population (VP-stop anchor)", N=len(vf), median_IQR_h=miqr(q),
             le8h=nN(int((q <= 8).sum()), len(vf)), le24h=nN(int((q <= 24).sum()), len(vf)), gt72h=nN(int((q > 72).sum()), len(vf)),
             polyuria_onset_before_T0b=nN(int((vf.onset <= q).sum()), len(vf)),
             poly8_onset_before_T0b=nN(int(((vf.poly8 == 1) & (vf.onset <= q)).sum()), int(vf.poly8.sum()))),
        dict(population="VP-first, base population (VP-stop anchor)", N=len(vfb), median_IQR_h=miqr(vfb.gap_vp_t0b_h),
             le8h=nN(int((vfb.gap_vp_t0b_h <= 8).sum()), len(vfb)), le24h=nN(int((vfb.gap_vp_t0b_h <= 24).sum()), len(vfb)),
             gt72h=nN(int((vfb.gap_vp_t0b_h > 72).sum()), len(vfb)))])
    T0H_b = np.floor(vfb.gap_vp_t0b_h).astype(int).value_counts().sort_index().rename_axis("hour_from_vpstop").reset_index(name="n")
    SH["VPstop_iii_T0b_hist_basepop"] = T0H_b
    KEY["VPstop"] = dict(summary=summ, t0b=SH["VPstop_iii_T0b_position"].to_dict("records"))
    say("VPstop done %.0fs" % (time.time() - t_all))

    # ===========================================================================
    # (2) Sodium
    L = data["lb"]
    L = L[L.itemid.isin(mc.NA_ITEMS)]

    def na_long(d):
        x = L.merge(d[["hadm_id", "stay_id", "t0"]], on="hadm_id")
        x["h"] = hrs(x.charttime, x.t0)
        return x

    P0 = D0[D0.pop_primary == 1].copy()
    P0["g2"] = np.where(P0.grp == "VP", "VP (2-group primary)", "NE-only (2-group primary)")
    MAINB["g2"] = MAINB.ugrp
    sets = [("2-group", P0, "T0_2grp"), ("5-group", MAINB, "T0b")]
    timing, bypoly, naflow, nabins, naord = [], [], [], [], []
    for lab, d, anc in sets:
        nl = na_long(d)
        post = nl[(nl.h > 0) & (nl.h <= 72)]
        first150 = post[post.v >= 150].groupby("stay_id").h.min()
        d = d.copy()
        d["t_na150"] = d.stay_id.map(first150)
        d["na_n72_chk"] = d.stay_id.map(post.groupby("stay_id").size()).fillna(0)
        d["dNa"] = d.na_peak_72 - d.na_pre
        for g in d.g2.unique():
            s = d[d.g2 == g]
            den_old = s[s.na_max_pre24 < 150]
            den = den_old[den_old.na_n72 >= 1]
            naflow.append(dict(set=lab, group=g, analysis_N=len(s), pre24_Na_missing=int(s.na_max_pre24.isna().sum()),
                               pre24_Na_ge150=int((s.na_max_pre24 >= 150).sum()), existing_denominator=len(den_old),
                               no_Na_0_72h=int((den_old.na_n72 == 0).sum()), na_denominator=len(den),
                               na150_72=nN(int(den.na150_72.sum()), len(den)),
                               na150_72_existing_denominator=nN(int(den_old.na150_72.sum()), len(den_old)),
                               count_check=int((den.na_n72 != den.na_n72_chk).sum())))
            ids = set(den.stay_id)
            pn = post[post.stay_id.isin(ids)]
            for b0 in range(0, 72, 12):
                inb = pn[(pn.h > b0) & (pn.h <= b0 + 12)]
                timing.append(dict(db=DBNAME, group=f"{g} [{lab}]", anchor=anc, bin_start_h=b0, bin_end_h=b0 + 12,
                                   n_measured=int(inb.stay_id.nunique()),
                                   n_first_na150=int(((den.t_na150 > b0) & (den.t_na150 <= b0 + 12)).sum()),
                                   n_tests=len(inb), N_denominator=len(den)))
            for b0 in range(0, 72, 6):
                inb = pn[(pn.h > b0) & (pn.h <= b0 + 6)]
                mx = inb.groupby("stay_id").v.max()
                nabins.append(dict(set=lab, group=g, bin=f"({b0},{b0 + 6}] h", N_denominator=len(den),
                                   n_stays_measured=len(mx), n_tests=len(inb), median_max_Na=mx.median() if len(mx) else np.nan,
                                   q1=mx.quantile(.25) if len(mx) else np.nan, q3=mx.quantile(.75) if len(mx) else np.nan))
            for ep8 in (0, 1):
                z = den[den.poly8 == ep8]
                dn = z.dNa.dropna()
                bypoly.append(dict(db=DBNAME, group=f"{g} [{lab}]", early_polyuria=ep8, n=len(z), n_na150=int(z.na150_72.sum()),
                                   pct=100 * z.na150_72.mean() if len(z) else np.nan,
                                   median_dNa=dn.median() if len(dn) else np.nan, q1_dNa=dn.quantile(.25) if len(dn) else np.nan,
                                   q3_dNa=dn.quantile(.75) if len(dn) else np.nan, n_dNa=len(dn)))
            z = den[(den.poly8 == 1) & (den.na150_72 == 1)]
            naord.append(dict(set=lab, group=g, poly8_and_Na150=len(z),
                              Na150_before_polyuria_start=int((z.t_na150 <= z.onset - 2).sum()),
                              Na150_during_qualifying_run=int(((z.t_na150 > z.onset - 2) & (z.t_na150 < z.onset)).sum()),
                              Na150_at_or_after_onset=int((z.t_na150 >= z.onset).sum()),
                              median_hours_onset_to_Na150=miqr(z.t_na150 - z.onset)))
            if lab == "5-group" and g == "VP-first":
                # time of the same events measured from the VP stop
                tv = den.t_na150 + den.gap_vp_t0b_h
                for b0 in range(0, int(np.ceil(tv.max() / 12) * 12) if tv.notna().any() else 0, 12):
                    timing.append(dict(db=DBNAME, group=f"{g} [{lab}]", anchor="VP_stop", bin_start_h=b0, bin_end_h=b0 + 12,
                                       n_measured=np.nan, n_first_na150=int(((tv > b0) & (tv <= b0 + 12)).sum()),
                                       n_tests=np.nan, N_denominator=len(den)))
                # Na >= 150 between the VP stop and T0b (earlier than 24 h before T0b: not covered by the denominator exclusion)
                nv = L.merge(den[["hadm_id", "stay_id", "t0a_vp", "t0"]], on="hadm_id")
                nv = nv[(nv.charttime > nv.t0a_vp) & (nv.charttime <= nv.t0)]
                naflow.append(dict(set=lab, group="VP-first: Na>=150 between VP stop and T0b (Na denominator)",
                                   na_denominator=len(den), na150_72=nN(int(nv[nv.v >= 150].stay_id.nunique()), len(den))))
        if lab == "2-group":
            P0 = d
        else:
            MAINB = d
    TM = pd.DataFrame(timing)
    TM[["db", "group", "anchor", "bin_start_h", "bin_end_h", "n_measured", "n_first_na150"]].to_csv(
        os.path.join(outdir, "na_timing.csv"), index=False)
    BP = pd.DataFrame(bypoly)
    BP[["db", "group", "early_polyuria", "n", "n_na150", "pct", "median_dNa", "q1_dNa", "q3_dNa"]].to_csv(
        os.path.join(outdir, "na_by_poly.csv"), index=False)
    SH["Na_00_denominators"] = pd.DataFrame(naflow)
    SH["Na_i_first_Na150_timing"] = TM
    SH["Na_ii_by_early_polyuria"] = BP
    SH["Na_ii_order_Na_vs_onset"] = pd.DataFrame(naord)
    SH["Na_iii_maxNa_6h_bins"] = pd.DataFrame(nabins)
    say(SH["Na_00_denominators"].to_string())

    # one model (two-group primary; MICE on all stays as in the primary analysis)
    IMPS0 = mm.impute(D0.drop(columns=["ugrp"]))
    NA_old = P0[P0.na_max_pre24 < 150]
    NA_new = NA_old[NA_old.na_n72 >= 1]
    mrows = []
    for lab, dd in (("existing denominator (pre-24 h Na <150)", NA_old), ("Na denominator (+ >=1 Na 0–72 h)", NA_new)):
        o1 = mm.or_mi(dd, IMPS0, "na150_72")["vp"]
        o2 = mm.or_mi(dd, IMPS0, "na150_72", expo="vp+poly8", terms=["vp", "poly8"])
        mrows.append(dict(denominator=lab, N=len(dd), N_VP=int(dd.vp.sum()), events_VP=int(dd[dd.vp == 1].na150_72.sum()),
                          events_NE=int(dd[dd.vp == 0].na150_72.sum()),
                          poly8_VP=int(dd[dd.vp == 1].poly8.sum()), poly8_NE=int(dd[dd.vp == 0].poly8.sum()),
                          aOR_VP_without_poly8=mm.fmt(o1), aOR_VP_with_poly8=mm.fmt(o2["vp"]), aOR_poly8=mm.fmt(o2["poly8"]),
                          pct_change_logOR=100 * (np.log(o2["vp"]["OR"]) - np.log(o1["OR"])) / np.log(o1["OR"])))
    SH["Na_iv_model"] = pd.DataFrame(mrows)
    say(SH["Na_iv_model"].to_string())
    KEY["Na"] = dict(model=mrows, denominators=naflow, order=naord)
    say("Na done %.0fs" % (time.time() - t_all))

    # ===========================================================================
    # (3) T0a
    DA = build_at_t0a(data, c, ep)
    DA = DA.set_index("stay_id")
    say("built at T0a", DA.shape)
    # additional variables at T0a
    ids8 = set(D0.loc[D0.ugrp.isin(["NE-first", "VP-first"]), "stay_id"])
    A = DA.loc[DA.index.isin(ids8)].copy()
    t0a = A.t0
    w = A.weight
    A["ne_rate"], A["ne_rate_unconv"], A["ne_running"] = rate_before(v, NE, t0a, w)
    A["vp_rate"], A["vp_rate_unconv"], A["vp_running"] = rate_before(v, VP, t0a, w)
    # stays in which the continuing drug was started after T0a (switch without overlap: NE->VP / VP->NE)
    fs = ep[ep.itemid.isin([NE, VP])].groupby(["stay_id", "itemid"]).start.min().unstack()
    g8 = A.index.map(gmap)
    cont_start = np.where(g8 == "NE-first", fs[VP].reindex(A.index), fs[NE].reindex(A.index))
    A["continuing_drug_started_after_t0a"] = (pd.to_datetime(cont_start) > A.t0).astype(int)
    ex = extra_exposure(v, u, A.reset_index()[["stay_id", "t0", "weight"]]).set_index("stay_id")
    for k in ex.columns:
        A[k] = ex[k]
    ie = data["ie"]
    oe = pd.read_parquet(os.path.join(mc.EXT, "oe_pressorstays.parquet"))
    ie8 = ie[ie.stay_id.isin(ids8)]
    oe8 = oe[oe.stay_id.isin(ids8)]
    I, O, io_counts = prep_io(ie8, oe8)
    FA = fluid_vars(I, O, A.reset_index()[["stay_id", "t0", "intime", "weight"]]).set_index("stay_id")
    A["fo_pct"] = FA.fo_pct
    A["icu_pre_h"] = FA.icu_pre_h
    # the same variables at T0b
    Bt = DB.set_index("stay_id").loc[sorted(ids8)].copy()
    Bt["ne_rate"], Bt["ne_rate_unconv"], Bt["ne_running"] = rate_before(v, NE, Bt.t0, Bt.weight)
    Bt["vp_rate"], Bt["vp_rate_unconv"], Bt["vp_running"] = rate_before(v, VP, Bt.t0, Bt.weight)
    FB = fluid_vars(I, O, Bt.reset_index()[["stay_id", "t0", "intime", "weight"]]).set_index("stay_id")
    Bt["fo_pct"] = FB.fo_pct
    fo_ref = pd.read_parquet(mc.cohort_file(mc.FLUID_TABLE_T0B)).set_index("stay_id").fo_pct.reindex(Bt.index)
    fo_mis = int((~np.isclose(Bt.fo_pct.fillna(-999), fo_ref.fillna(-999), atol=1e-9)).sum())
    say("FO% at T0b vs cohort table of script 06, mismatches:", fo_mis, "of", len(Bt))
    # check: T0 of the T0a rebuild equals t0a of the cohort table
    chk_t0a = int((A.t0 != DB.set_index("stay_id").t0a.reindex(A.index)).sum())
    say("T0a rebuild t0 mismatches vs parquet t0a:", chk_t0a)
    SH["00_reproduction"] = pd.concat([SH["00_reproduction"], pd.DataFrame([dict(
        check="T0a build: anchor time vs t0a in the cohort table of script 05 (mismatches among NE-first+VP-first)", value=str(chk_t0a), expected="0",
        match=chk_t0a == 0), dict(check="FO% at T0b recomputed with fluid_vars vs cohort table of script 06 (mismatches, NE-first+VP-first)",
        value="%d/%d" % (fo_mis, len(Bt)), expected="0", match=fo_mis == 0)])], ignore_index=True)

    pops = {"popA_VPstop": D0[(D0.vpstop_main == 1) & D0.ugrp.isin(["NE-first", "VP-first"])][["stay_id", "subject_id", "ugrp", "poly8"]],
            "popB_T0b": MAINB[MAINB.ugrp.isin(["NE-first", "VP-first"])][["stay_id", "subject_id", "ugrp", "poly8"]]}
    U8 = sorted(set(pops["popA_VPstop"].stay_id) | set(pops["popB_T0b"].stay_id))
    X = A.loc[U8].copy()
    X["z"] = (X.index.map(gmap) == "NE-first").astype(int)
    X["y_vpstop"] = X.index.map(D0.set_index("stay_id").poly8).astype(float)
    X["y_t0b"] = X.index.map(DB.set_index("stay_id").poly8).astype(float)
    X = X.reset_index()
    IMP8 = impute8(X)
    say("Order imputation done", len(X))
    X = X.set_index("stay_id", drop=False)
    for i in range(len(IMP8)):
        IMP8[i].index = X.index
    # description (T0a / T0b)
    desc = []
    for lab, src in (("T0a", A), ("T0b", Bt)):
        for g in ("NE-first", "VP-first"):
            for pn, pp in pops.items():
                s = src.loc[pp[pp.ugrp == g].stay_id]
                r = dict(anchor=lab, population=pn, group=g, N=len(s))
                for col, nm in SMD8:
                    if col in s.columns and not col.endswith("_i") and col not in ("log_press_h",):
                        xx = pd.to_numeric(s[col], errors="coerce")
                        if set(xx.dropna().unique()) <= {0, 1}:
                            r[nm] = nN(int(xx.sum()), int(xx.notna().sum()))
                        else:
                            r[nm] = miqr(xx, 2) + " (n=%d)" % xx.notna().sum()
                r["NE rate not convertible to mcg/kg/min"] = int(s.ne_rate_unconv.sum())
                r["NE running at anchor"] = nN(int(s.ne_running.sum()), len(s))
                r["VP running at anchor"] = nN(int(s.vp_running.sum()), len(s))
                if lab == "T0a":
                    r["continuing drug (VP for NE-first / NE for VP-first) first started after T0a"] = nN(
                        int(s.continuing_drug_started_after_t0a.sum()), len(s))
                desc.append(r)
    SH["Order_i_background"] = pd.DataFrame(desc)

    # SMD: unweighted (T0a, T0b) and ATO-weighted (T0a)
    smd_rows, est_rows, ato_detail = [], [], []
    for pn, pp in pops.items():
        ids = pp.stay_id.values
        zz = (pp.ugrp.values == "NE-first").astype(int)
        y = pp.poly8.values.astype(float)
        d8 = X.loc[ids].copy()
        d8.index.name = None
        d8["vp"] = zz                     # ato_analysis uses d.vp as the treatment indicator (here NE-first = 1)
        d8["yy"] = y
        imps = [im.loc[ids] for im in IMP8]
        res, pts, _ = ato_analysis(d8, imps, ["yy"], PS8, NBOOT, with_imp8, seed=2026)
        r = res["yy"]
        # weights (mean over imputations) and weighted SMD (per imputation, averaged)
        Wl = [p[1] for p in pts]
        ess = [(w_[zz == 1].sum() ** 2 / (w_[zz == 1] ** 2).sum(), w_[zz == 0].sum() ** 2 / (w_[zz == 0] ** 2).sum()) for w_ in Wl]
        bt = Bt.loc[ids]
        for col, nm in SMD8:
            un_a, wt_a = [], []
            for im, w_ in zip(imps, Wl):
                xi = with_imp8(d8, im)
                if col not in xi.columns:
                    continue
                a_, b_ = wsmd(xi, zz, w_, col)
                un_a.append(a_)
                wt_a.append(b_)
            if col in bt.columns:
                sb = smd_un(pd.to_numeric(bt[col], errors="coerce")[zz == 1], pd.to_numeric(bt[col], errors="coerce")[zz == 0])
            else:
                sb = np.nan
            smd_rows.append(dict(db=DBNAME, variable=f"{pn}|{nm}", smd_T0a=np.mean(un_a) if un_a else np.nan, smd_T0b=sb,
                                 smd_T0a_weighted=np.mean(wt_a) if wt_a else np.nan))
        # crude OR (clustered SE)
        dfc = pd.DataFrame({"y": y, "z": zz, "subject_id": pp.subject_id.values})
        fr = mm.glm_fit("y~z", dfc)
        lo_, hi_ = fr.conf_int().loc["z"]
        est_rows.append(dict(db=DBNAME, outcome="poly8_VPstop_anchor" if pn == "popA_VPstop" else "poly8_T0b_anchor",
                             events_NEfirst=int(y[zz == 1].sum()), n_NEfirst=int((zz == 1).sum()),
                             events_VPfirst=int(y[zz == 0].sum()), n_VPfirst=int((zz == 0).sum()),
                             crude_or=np.exp(fr.params["z"]), crude_lo=np.exp(lo_), crude_hi=np.exp(hi_),
                             ato_or=np.exp(r["lor"]), ato_lo=np.exp(r["lor_lo"]), ato_hi=np.exp(r["lor_hi"]),
                             ato_rd_pct=100 * r["rd"], ato_rd_lo=100 * r["rd_lo"], ato_rd_hi=100 * r["rd_hi"]))
        ato_detail.append(dict(population=pn, N=len(ids), weighted_risk_NEfirst=100 * r["p1"], weighted_risk_VPfirst=100 * r["p0"],
                               nboot=r["nboot"], ps_covariates=PS8, n_ps_parameters=len(PS8.split("+")),
                               events_min_group=int(min(y[zz == 1].sum(), y[zz == 0].sum())), events_total=int(y.sum()),
                               EPV_outcome_total=y.sum() / len(PS8.split("+")),
                               treatment_EPV=min((zz == 1).sum(), (zz == 0).sum()) / len(PS8.split("+")),
                               ESS_NEfirst=np.mean([e[0] for e in ess]), ESS_VPfirst=np.mean([e[1] for e in ess]),
                               ps_median_NEfirst=np.mean([np.median(p[2][zz == 1]) for p in pts]),
                               ps_median_VPfirst=np.mean([np.median(p[2][zz == 0]) for p in pts])))
    SMD = pd.DataFrame(smd_rows)
    SMD.to_csv(os.path.join(outdir, "order_smd.csv"), index=False)
    EST = pd.DataFrame(est_rows)
    EST.to_csv(os.path.join(outdir, "order_estimates.csv"), index=False)
    SH["Order_ii_estimates"] = EST
    SH["Order_ii_ato_detail"] = pd.DataFrame(ato_detail)
    SH["Order_i_smd"] = SMD
    SH["Order_io_rows"] = pd.DataFrame([io_counts]).T.reset_index().rename(columns={"index": "item", 0: "value"})
    say(EST.to_string())
    say(SMD.to_string())
    KEY["Order"] = dict(estimates=est_rows, ato=ato_detail)

    SH = {"README": pd.DataFrame([
        ("script", "07_vpstop_timing.py; same functions as scripts 02, 05 and 06 (modules mimic_common, mimic_models, mimic_groups, mimic_fluid)"),
        ("VPstop anchor", "VP-containing groups: VP stop = end of first VP episode (= T0 of script 02). NE-only: NE stop (= T0 of script 02). Reference rows anchor=T0b for NE-first/VP-first (analysis population of script 05)."),
        ("VPstop populations", "(i)(iv) base: adult, anchor in ICU, alive at anchor, no RRT, no pituitary/DI/brain death; NE-only also no other vasopressor at T0. (ii) + >=12/24 urine bins after anchor + no polyuria −12..−1 h. (v) (ii) without the new-onset rule."),
        ("VPstop hourly", "bin k = [anchor+k h, anchor+k+1 h). n_alive_in_icu = in ICU and not dead at bin start. n_in_run = bin >3 mL/kg/h within a run of >=2 consecutive recorded bins >3 (missing bin breaks a run)."),
        ("VPstop CIF", "Aalen–Johansen; competing death, ICU discharge, vasopressor restart (event_data of mimic_models; tie → polyuria). n_events = events in (t−1, t]. 95%% CI = subject cluster bootstrap percentile, B=%d." % NBOOT),
        ("Na", "Denominator = pre-T0b 24 h Na all <150 (max <150; missing pre-Na excluded as in scripts 05/06) and >=1 Na in (0,72] h. Na items 50983/52623/50824 (labs.csv). 2-group anchor = T0 of script 02; 5-group anchor = T0b."),
        ("Na model", "na150_72 ~ vp (+ poly8) + COV_MI, MICE of script 02 (m=10) on all 16,344 stays, subject-clustered SE (or_mi)."),
        ("Order", "Covariates re-measured at T0a = earlier of last NE end and VP stop, by running build() with T0 := T0a (anchor_cols). PS (logit) = " + PS8 + "; overlap weights (ATO, ato_est); exposure NE-first=1 vs VP-first=0; subject cluster bootstrap B=%d; MICE m=10 (impute with IMPVARS + FO%%, UO 6 h)." % NBOOT),
        ("status", "post hoc, exploratory, descriptive"),
    ], columns=["item", "description"]), **SH}
    xl = os.path.join(outdir, "results_mimic_vpstop_timing.xlsx")
    with pd.ExcelWriter(xl) as xw:
        for k, t in SH.items():
            t.to_excel(xw, sheet_name=k[:31], index=False)
    with open(os.path.join(outdir, "key_results.json"), "w") as f:
        json.dump(KEY, f, default=str, indent=1, ensure_ascii=False)
    say("total %.0f s" % (time.time() - t_all))
    say.write(os.path.join(outdir, "run_log_part1.txt"))
    return dict(D0=D0, DB=DB, MAINB=MAINB, c=c, ep=ep, pop0=pop0, hr0=hr0, P0=P0, IMPS0=IMPS0, X=X, IMP8=IMP8, A=A,
                pops=pops, reproduction=SH["00_reproduction"], summ=summ, est_rows=est_rows, CIF=CIF, H=H)


# ===========================================================================
# Part 2: switch sensitivity, closed cohort, number at risk
COLS = ["db", "analysis", "group", "anchor", "n", "events", "estimate", "lo", "hi", "note"]
TL = [0, 2, 4, 6, 8, 12, 16, 20, 24]
HL = [-24, -16, -8, -1, 0, 7, 8, 16, 23]


def R(analysis, group, anchor, n=np.nan, events=np.nan, estimate=np.nan, lo=np.nan, hi=np.nan, note=""):
    return dict(db=DBNAME, analysis=analysis, group=group, anchor=anchor, n=n, events=events, estimate=estimate,
                lo=lo, hi=hi, note=note)


def ensure_oe():
    """Create WORK_DIR/mimic/ext/oe_pressorstays.parquet if missing: outputevents restricted to the
    vasopressor-treated stays of vaso2.csv (same columns and rule as extract() of 06_fluid_balance.py)."""
    os.makedirs(mc.EXT, exist_ok=True)
    f = os.path.join(mc.EXT, "oe_pressorstays.parquet")
    if os.path.exists(f):
        return "exists"
    vv = pd.read_csv(os.path.join(mc.DATA, "vaso2.csv"), header=None, usecols=[1], names=["stay_id"])
    sids = set(vv.stay_id.dropna().astype("int64"))
    oe = pd.read_csv(os.path.join(mc.MIMIC, "icu", "outputevents.csv.gz"),
                     usecols=["subject_id", "stay_id", "charttime", "itemid", "value", "valueuom"])
    oe = oe[oe.stay_id.isin(sids)].copy()
    oe["charttime"] = pd.to_datetime(oe.charttime)
    oe.to_parquet(f)
    return "created %d rows" % len(oe)


def switch_flags(c, ep):
    fs = ep[ep.itemid.isin([NE, VP])].groupby(["stay_id", "itemid"]).start.min().unstack()
    s = c.set_index("stay_id")[["ugrp", "t0a", "t0a_vp", "ne_last_end", "t0b"]].copy()
    s["vp_first_start"] = fs[VP].reindex(s.index)
    s["ne_first_start"] = fs[NE].reindex(s.index)
    s["sw_ne_vp"] = ((s.ugrp == "NE-first") & (s.vp_first_start > s.t0a)).astype(int)
    s["sw_vp_ne"] = ((s.ugrp == "VP-first") & (s.ne_first_start > s.t0a)).astype(int)
    s["gap_sw_h"] = np.where(s.sw_ne_vp == 1, hrs(s.vp_first_start, s.ne_last_end),
                             np.where(s.sw_vp_ne == 1, hrs(s.ne_first_start, s.t0a_vp), np.nan))
    s["sw_any"] = s.sw_ne_vp | s.sw_vp_ne
    return s


def window_stats(pop, hr, lo, hi):
    """Stay-hour summary (n_in_run / n_recorded of hourly_table summed over the window) and pooled mean urine
    output of the recorded bins."""
    h = hr[hr.stay_id.isin(set(pop.stay_id)) & (hr.h >= lo) & (hr.h <= hi)]
    return dict(bins=len(h), in_run=int(h.in_run.sum()), uo_mean=h.rate.mean() if len(h) else np.nan,
                uo_sum=h.rate.sum())


def per_stay(pop, hr, lo, hi):
    h = hr[hr.stay_id.isin(set(pop.stay_id)) & (hr.h >= lo) & (hr.h <= hi)]
    g = h.groupby("stay_id").agg(nb=("h", "size"), anyrun=("in_run", "max"), uo=("rate", "mean"))
    return g.reindex(pop.stay_id.values)


def mcnemar_exact(pre, post):
    pre, post = np.asarray(pre, int), np.asarray(post, int)
    b, c_ = int(((pre == 1) & (post == 0)).sum()), int(((pre == 0) & (post == 1)).sum())
    p = 1.0 if b + c_ == 0 else stats.binomtest(min(b, c_), b + c_, 0.5).pvalue
    return b, c_, p


def tci(x):
    x = pd.Series(x).dropna()
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    return m, m - q * se, m + q * se, n


def ato_rows(pops, X, IMP8, keepfn, label):
    out = []
    for pn, pp in pops.items():
        pp = pp[keepfn(pp)]
        ids = pp.stay_id.values
        zz = (pp.ugrp.values == "NE-first").astype(int)
        y = pp.poly8.values.astype(float)
        d8 = X.loc[ids].copy()
        d8.index.name = None
        d8["vp"] = zz
        d8["yy"] = y
        imps = [im.loc[ids] for im in IMP8]
        res, pts, _ = ato_analysis(d8, imps, ["yy"], PS8, NBOOT, with_imp8, seed=2026)
        r = res["yy"]
        Wl = [p[1] for p in pts]
        ess = [(w_[zz == 1].sum() ** 2 / (w_[zz == 1] ** 2).sum(), w_[zz == 0].sum() ** 2 / (w_[zz == 0] ** 2).sum()) for w_ in Wl]
        dfc = pd.DataFrame({"y": y, "z": zz, "subject_id": pp.subject_id.values})
        fr = mm.glm_fit("y~z", dfc)
        clo, chi = np.exp(fr.conf_int().loc["z"])
        anc = "VP_stop" if pn == "popA_VPstop" else "T0b"
        note = ("NE-first %d/%d vs VP-first %d/%d; crude OR %.2f (%.2f–%.2f); ATO RD %+.1f pp (%.1f to %.1f); weighted risk %.1f%% vs %.1f%%; "
                "ESS %.0f/%.0f; PS = %s; MICE m=10 (imputation of part 1 reused, subset rows); subject cluster bootstrap B=%d"
                % (y[zz == 1].sum(), (zz == 1).sum(), y[zz == 0].sum(), (zz == 0).sum(), np.exp(fr.params["z"]), clo, chi,
                   100 * r["rd"], 100 * r["rd_lo"], 100 * r["rd_hi"], 100 * r["p1"], 100 * r["p0"],
                   np.mean([e[0] for e in ess]), np.mean([e[1] for e in ess]), PS8, r["nboot"]))
        out.append(R("ATO_OR_poly8_NEfirst_vs_VPfirst", label, anc, n=len(ids), events=int(y.sum()),
                     estimate=np.exp(r["lor"]), lo=np.exp(r["lor_lo"]), hi=np.exp(r["lor_hi"]), note=note))
    return out


def main():
    t1 = time.time()
    outdir = config.out_dir("mimic_07")
    mc.check_extracts()
    mc.build_caches(say)
    say("oe_pressorstays:", ensure_oe())
    L = timing_analyses(outdir)                  # part 1
    D0, DB, MAINB, c, ep, pop0, hr0 = L["D0"], L["DB"], L["MAINB"], L["c"], L["ep"], L["pop0"], L["hr0"]
    P0, IMPS0, X, IMP8, A, pops = L["P0"], L["IMPS0"], L["X"], L["IMP8"], L["A"], L["pops"]
    REP, AR, BR, CR, DR, BH = [], [], [], [], [], []

    # ---- 0. reproduction (checks of part 1 and expected values) -------------------
    rep1 = pd.DataFrame(L["reproduction"])
    for _, r in rep1.iterrows():
        REP.append(dict(check="part 1: " + r["check"], value=r["value"], expected=r["expected"], match=bool(r["match"])))
    sm = {(s["group"], s["anchor"]): s for s in L["summ"]}
    for g, k in (("NE-first", "13.0%"), ("VP-first", "11.1%"), ("NE-only", "5.6%")):
        a_ = "NE_stop" if g == "NE-only" else "VP_stop"
        REP.append(dict(check=f"AJ CIF 8 h {g} ({a_})", value=sm[(g, a_)]["cif8"] + " N=%d" % sm[(g, a_)]["N"],
                        expected=k + (" N=4667" if g == "NE-only" else ""), match=sm[(g, a_)]["cif8"].startswith(k)))
    est = {e["outcome"]: e for e in L["est_rows"]}
    for o_, k in (("poly8_VPstop_anchor", "1.17 (0.83–1.70)"), ("poly8_T0b_anchor", "1.61 (1.09–2.34)")):
        e = est[o_]
        v_ = "%.2f (%.2f–%.2f)" % (e["ato_or"], e["ato_lo"], e["ato_hi"])
        REP.append(dict(check="Order ATO OR " + o_, value=v_, expected=k, match=v_ == k))
    SW = switch_flags(c, ep)
    chk = (A.continuing_drug_started_after_t0a != A.index.map(SW.sw_any)).sum()
    REP.append(dict(check="switch flag (switch_flags) vs continuing_drug_started_after_t0a of part 1 (NE-first+VP-first, mismatches)",
                    value=str(int(chk)), expected="0", match=int(chk) == 0))
    for d_ in (D0, DB, P0, MAINB, pop0):
        d_["sw_ne_vp"] = d_.stay_id.map(SW.sw_ne_vp).fillna(0).astype(int)
        d_["sw_vp_ne"] = d_.stay_id.map(SW.sw_vp_ne).fillna(0).astype(int)
        d_["gap_sw_h"] = d_.stay_id.map(SW.gap_sw_h)
    popA_nf = D0[(D0.vpstop_main == 1) & (D0.ugrp == "NE-first")]
    REP.append(dict(check="NE→VP switch among NE-first, VP-stop new-onset population", value="%d/%d" % (popA_nf.sw_ne_vp.sum(), len(popA_nf)),
                    expected="46/570", match=(int(popA_nf.sw_ne_vp.sum()), len(popA_nf)) == (46, 570)))
    # two-group primary analysis
    o_all = {y: mm.or_mi(P0, IMPS0, y)["vp"] for y in mm.YS}
    NA0 = P0[P0.na_max_pre24 < 150]
    o_all["na150_72"] = mm.or_mi(NA0, IMPS0, "na150_72")["vp"]
    nvp, nne = int((P0.grp == "VP").sum()), int((P0.grp == "NE").sum())
    for y, k in (("poly24", "1.51 (1.24–1.84)"), ("poly8", "1.70 (1.29–2.24)"), ("na150_72", "2.19")):
        v_ = mm.fmt(o_all[y])
        REP.append(dict(check=f"2-group primary aOR {y} (VP {nvp} vs NE {nne})", value=v_, expected=k + " (743 vs 4667)",
                        match=v_.startswith(k) and (nvp, nne) == (743, 4667)))
    # five groups (T0b)
    IMPSB = mm.impute(DB)
    gm_all = {}
    for y in mm.YS:
        gm_all[y] = {r["group"]: r for r in group_models(MAINB, IMPSB, y, "ugrp", G4)}
    NAB = MAINB[MAINB.na_max_pre24 < 150]
    gm_all["na150_72"] = {r["group"]: r for r in group_models(NAB, IMPSB, "na150_72", "ugrp", G4)}
    r_ = gm_all["poly8"]["NE-first"]
    nnf = int((MAINB.ugrp == "NE-first").sum())
    REP.append(dict(check="5-group T0b NE-first 0–8 h aOR (n NE-first)", value=r_["adjusted_OR"] + " n=%d" % nnf,
                    expected="1.72 (1.25–2.36) n=560", match=r_["adjusted_OR"] == "1.72 (1.25–2.36)" and nnf == 560))
    REPd = pd.DataFrame(REP)
    say(REPd.to_string())
    REPd.to_csv(os.path.join(outdir, "reproduction_checks.csv"), index=False)
    if not REPd.match.all():
        say("REPRODUCTION FAILED — stopping before extensions")
        return dict(reproduced=False)

    # ---- switch counts ---------------------------------------------------------------
    def cnt(lab, d, flag, grp_label, anc, note=""):
        k = int(d[flag].sum())
        gaps = d.loc[d[flag] == 1, "gap_sw_h"]
        AR.append(R("switch_count", grp_label, anc, n=len(d), events=k, estimate=100 * k / len(d) if len(d) else np.nan,
                    note=f"{lab}; estimate = % of n; gap stop→start of other drug median [IQR] {miqr(gaps)} h; " + note))

    p0vp = P0[P0.grp == "VP"]
    brk = p0vp.ugrp.value_counts().to_dict()
    cnt("(i) 2-group primary VP group: NE→VP switch (NE-first)", p0vp, "sw_ne_vp", "VP (2-group primary)", "T0_2grp (VP stop)",
        "ugrp of the 743: " + json.dumps(brk, ensure_ascii=False))
    cnt("(i) 2-group primary VP group: VP→NE switch (VP-first)", p0vp, "sw_vp_ne", "VP (2-group primary)", "T0_2grp (VP stop)")
    cnt("(ii) NE-first, 5-group T0b analysis population", MAINB[MAINB.ugrp == "NE-first"], "sw_ne_vp", "NE-first", "T0b")
    cnt("VP-first, 5-group T0b analysis population: VP→NE", MAINB[MAINB.ugrp == "VP-first"], "sw_vp_ne", "VP-first", "T0b")
    cnt("(iii) NE-first, VP-stop new-onset population", popA_nf, "sw_ne_vp", "NE-first", "VP_stop")
    cnt("VP-first, VP-stop new-onset population: VP→NE", D0[(D0.vpstop_main == 1) & (D0.ugrp == "VP-first")], "sw_vp_ne", "VP-first", "VP_stop")
    cnt("NE-first, VP-stop basic population", pop0[pop0.ugrp == "NE-first"], "sw_ne_vp", "NE-first", "VP_stop")
    cnt("VP-first, VP-stop basic population: VP→NE", pop0[pop0.ugrp == "VP-first"], "sw_vp_ne", "VP-first", "VP_stop")

    # ---- two-group primary, NE->VP switch stays removed from the VP group ----------
    P0x = P0[~((P0.grp == "VP") & (P0.sw_ne_vp == 1))]
    for lab, d, oo in (("all (reproduction)", P0, o_all), ("excl. NE→VP switch", P0x, None)):
        gl = "VP (2-group primary)" if oo is not None else "VP excl. switch"
        for y in mm.YS + ["na150_72"]:
            dd = d[d.na_max_pre24 < 150] if y == "na150_72" else d
            o = oo[y] if oo is not None else mm.or_mi(dd, IMPS0, y)["vp"]
            k1, n1 = int(dd[dd.grp == "VP"][y].sum()), int((dd.grp == "VP").sum())
            k0, n0 = int(dd[dd.grp == "NE"][y].sum()), int((dd.grp == "NE").sum())
            AR.append(R(f"primary_events_{y}", gl, "T0_2grp", n=n1, events=k1, estimate=100 * k1 / n1, note=lab + "; % of n"))
            AR.append(R(f"primary_events_{y}", "NE without VP", "T0_2grp", n=n0, events=k0, estimate=100 * k0 / n0, note=lab + "; % of n"))
            AR.append(R(f"primary_aOR_{y}", gl + " vs NE without VP", "T0_2grp", n=len(dd), events=k1 + k0, estimate=o["OR"], lo=o["lo"], hi=o["hi"],
                        note=f"{lab}; VP {k1}/{n1}, NE {k0}/{n0}; p={o['p']:.4f}; COV_MI, MICE m=10 (impute on all stays as in script 02), subject-clustered SE"
                        + ("; Na denominator = pre-24 h Na max <150 (existing)" if y == "na150_72" else "")
                        + ("; >8–24 h = main risk set (earlier onset = non-event)" if y == "poly_8_24" else "")))
    say("two-group refit done")

    # ---- five groups at T0b, switch stays removed from NE-first ------------------
    MBx = MAINB[~((MAINB.ugrp == "NE-first") & (MAINB.sw_ne_vp == 1))]
    gm_x = {y: {r["group"]: r for r in group_models(MBx, IMPSB, y, "ugrp", G4)} for y in mm.YS}
    gm_x["na150_72"] = {r["group"]: r for r in group_models(MBx[MBx.na_max_pre24 < 150], IMPSB, "na150_72", "ugrp", G4)}
    for lab, gm, d, gl in (("all (reproduction)", gm_all, MAINB, "NE-first"), ("NE→VP switch stays removed from NE-first", gm_x, MBx, "NE-first excl. switch")):
        for y in mm.YS + ["na150_72"]:
            r = gm[y]["NE-first"]
            dd = d[d.na_max_pre24 < 150] if y == "na150_72" else d
            s1 = dd[dd.ugrp == "NE-first"]
            AR.append(R(f"fivegroup_aOR_{y}", gl + " vs NE without VP", "T0b", n=r["n_model"], events=int(s1[y].sum()),
                        estimate=r.get("adjusted_est"), lo=r.get("adjusted_lo"), hi=r.get("adjusted_hi"),
                        note=f"{lab}; NE-first {r['events']}; NE without VP {r['ref_NE_only']}; crude {r['crude_OR']}; "
                             "group_models (5 groups, other 3 groups kept in model), COV_MI, MICE m=10 (impute on all stays as in script 05), clustered SE"
                             + ("; Na denominator = pre-24 h Na max <150" if y == "na150_72" else "")
                             + ("; >8–24 h = main risk set" if y == "poly_8_24" else "")))
    swB = MAINB[(MAINB.ugrp == "NE-first") & (MAINB.sw_ne_vp == 1)]
    for y in mm.YS + ["na150_72"]:
        dd = swB[swB.na_max_pre24 < 150] if y == "na150_72" else swB
        AR.append(R(f"fivegroup_descriptive_{y}", "NE-first switch (NE→VP)", "T0b", n=len(dd), events=int(dd[y].sum()),
                    estimate=100 * dd[y].mean() if len(dd) else np.nan, note="switch subgroup only, crude %; no model"))
    say("five-group refit done")

    # ---- VP-stop anchor: NE-first without switch / switch only ------------------
    nf_main = D0[(D0.vpstop_main == 1) & (D0.ugrp == "NE-first")]
    nf_base = pop0[pop0.ugrp == "NE-first"]
    subsets = {"NE-first": (nf_main, nf_base),
               "NE-first excl. switch": (nf_main[nf_main.sw_ne_vp == 0], nf_base[nf_base.sw_ne_vp == 0]),
               "NE-first switch (NE→VP)": (nf_main[nf_main.sw_ne_vp == 1], nf_base[nf_base.sw_ne_vp == 1])}
    CIFX = {}
    for gl, (xm, xb) in subsets.items():
        rows_, s_ = cif_table(xm, gl, "VP_stop")
        CIFX[gl] = (xm, rows_)
        AR.append(R("vpstop_crude_poly8", gl, "VP_stop", n=len(xm), events=int(xm.poly8.sum()), estimate=100 * xm.poly8.mean(),
                    note="VP-stop new-onset population; crude %"))
        for t in (8, 24):
            rr = rows_[t]
            AR.append(R(f"vpstop_AJ_CIF_{t}h", gl, "VP_stop", n=len(xm), events=s_["aj_events_0_8"] if t == 8 else s_["aj_events_0_8"] + s_["aj_events_8_24"],
                        estimate=100 * rr["cif"], lo=100 * rr["lo"], hi=100 * rr["hi"],
                        note="Aalen–Johansen, competing death/ICU discharge/vasopressor restart (event_data); events = AJ polyuria events ≤t; subject cluster bootstrap B=%d percentile" % NBOOT))
        for lab, (lo_, hi_) in (("pre_-12to-1h", (-12, -1)), ("post_0to7h", (0, 7))):
            w_ = window_stats(xb, hr0, lo_, hi_)
            AR.append(R(f"vpstop_stayhour_inrun_pct_{lab}", gl, "VP_stop", n=w_["bins"], events=w_["in_run"],
                        estimate=100 * w_["in_run"] / w_["bins"] if w_["bins"] else np.nan,
                        note="basic population (n stays=%d); n = stay-hours with urine recorded, events = stay-hours in a polyuria run" % len(xb)))
        for lab, (lo_, hi_) in (("pre12h", (-12, -1)), ("post8h", (0, 7))):
            w_ = window_stats(xb, hr0, lo_, hi_)
            AR.append(R(f"vpstop_mean_uo_mlkgh_{lab}", gl, "VP_stop", n=w_["bins"], estimate=w_["uo_mean"],
                        note="basic population (n stays=%d); mean of hourly mL/kg/h over all recorded stay-hours (= vpstop_hourly definition)" % len(xb)))
    say("VP-stop anchor done")

    # ---- ATO, switch stays removed from both groups ------------------------------
    for e in L["est_rows"]:
        AR.append(R("ATO_OR_poly8_NEfirst_vs_VPfirst", "NE-first vs VP-first (all; reproduction)",
                    "VP_stop" if e["outcome"] == "poly8_VPstop_anchor" else "T0b", n=e["n_NEfirst"] + e["n_VPfirst"],
                    events=e["events_NEfirst"] + e["events_VPfirst"], estimate=e["ato_or"], lo=e["ato_lo"], hi=e["ato_hi"],
                    note="output of part 1; NE-first %d/%d vs VP-first %d/%d" % (e["events_NEfirst"], e["n_NEfirst"], e["events_VPfirst"], e["n_VPfirst"])))
    swset = set(SW.index[SW.sw_any == 1])
    AR += ato_rows(pops, X, IMP8, lambda pp: ~pp.stay_id.isin(swset), "NE-first excl. switch vs VP-first excl. switch")
    say("ATO done")
    AD = pd.DataFrame(AR)[COLS]
    AD.to_csv(os.path.join(outdir, "switch_sensitivity.csv"), index=False)

    # ---- closed cohort --------------------------------------------------------------
    FIG = {"NE-first": (pop0[pop0.ugrp == "NE-first"], "VP_stop"), "VP-first": (pop0[pop0.ugrp == "VP-first"], "VP_stop"),
           "NE without VP": (pop0[pop0.ugrp == "NE-only"], "NE_stop"),
           "NE-first excl. switch": (pop0[(pop0.ugrp == "NE-first") & (pop0.sw_ne_vp == 0)], "VP_stop")}
    for coh, (mpre, mpost) in (("closed", (6, 4)), ("closed_strict", (10, 7))):
        for gl, (pb, anc) in FIG.items():
            p = pb.set_index("stay_id")
            t_lo, t_hi = p.t0 - pd.Timedelta(hours=12), p.t0 + pd.Timedelta(hours=8)
            ok = (p.intime <= t_lo) & (p.outtime >= t_hi) & (p.deathtime.isna() | (p.deathtime >= t_hi))
            pre = per_stay(pb, hr0, -12, -1)
            post = per_stay(pb, hr0, 0, 7)
            ok &= (pre.nb.fillna(0).values >= mpre) & (post.nb.fillna(0).values >= mpost)
            cp = pb[ok.values]
            nt = f"{coh}: alive and in ICU from anchor−12 h to +8 h, urine recorded ≥{mpre}/12 bins −12..−1 h and ≥{mpost}/8 bins 0..7 h; from basic population n={len(pb)}"
            BR.append(R("closed_n_stays", gl, anc, n=len(pb), events=len(cp), estimate=100 * len(cp) / len(pb), note=nt + "; estimate = % of basic population"))
            ws = {}
            for lab, (lo_, hi_) in (("pre", (-12, -1)), ("post", (0, 7))):
                w_ = window_stats(cp, hr0, lo_, hi_)
                ws[lab] = w_
                BR.append(R(f"closed_stayhour_inrun_pct_{lab}", gl, anc, n=w_["bins"], events=w_["in_run"],
                            estimate=100 * w_["in_run"] / w_["bins"], note=nt + "; n = recorded stay-hours"))
            a_pre, a_post = per_stay(cp, hr0, -12, -1), per_stay(cp, hr0, 0, 7)
            x1, x2 = a_pre.anyrun.fillna(0).astype(int).values, a_post.anyrun.fillna(0).astype(int).values
            b_, c_, pv = mcnemar_exact(x1, x2)
            BR.append(R("closed_stay_anyrun_pct_pre", gl, anc, n=len(cp), events=int(x1.sum()), estimate=100 * x1.mean(), note=nt + "; ≥1 in-run hour in −12..−1 h"))
            BR.append(R("closed_stay_anyrun_pct_post", gl, anc, n=len(cp), events=int(x2.sum()), estimate=100 * x2.mean(), note=nt + "; ≥1 in-run hour in 0..7 h"))
            BR.append(R("closed_stay_anyrun_mcnemar_exact_p", gl, anc, n=len(cp), events=b_ + c_, estimate=pv,
                        note=nt + f"; discordant pairs pre-only={b_}, post-only={c_}; exact binomial McNemar, two-sided (stay-level; patients with >1 stay not clustered)"))
            u1, u2 = a_pre.uo.values, a_post.uo.values
            BR.append(R("closed_stay_mean_uo_pre", gl, anc, n=len(cp), estimate=np.nanmean(u1), note=nt + "; per-stay mean of recorded hourly mL/kg/h, then mean across stays"))
            BR.append(R("closed_stay_mean_uo_post", gl, anc, n=len(cp), estimate=np.nanmean(u2), note=nt))
            m_, l_, h_, nn = tci(u2 - u1)
            BR.append(R("closed_stay_mean_uo_paired_diff_post_minus_pre", gl, anc, n=nn, estimate=m_, lo=l_, hi=h_, note=nt + "; paired t 95% CI (stay-level)"))
            for r in hourly_table(cp, hr0, gl, anc):
                r["cohort"] = coh
                BH.append(r)
    BD = pd.DataFrame(BR)[COLS]
    BD.to_csv(os.path.join(outdir, "closed_cohort.csv"), index=False)
    BHD = pd.DataFrame(BH)
    BHD.to_csv(os.path.join(outdir, "closed_cohort_hourly.csv"), index=False)
    say("closed cohort done")

    # ---- risk table (VP-stop new-onset population; same AJ as vpstop_cif) -----------
    RG = [("NE-first", "VP_stop", D0[(D0.vpstop_main == 1) & (D0.ugrp == "NE-first")]),
          ("VP-first", "VP_stop", D0[(D0.vpstop_main == 1) & (D0.ugrp == "VP-first")]),
          ("NE without VP", "NE_stop", D0[(D0.vpstop_main == 1) & (D0.ugrp == "NE-only")]),
          ("NE-first excl. switch", "VP_stop", D0[(D0.vpstop_main == 1) & (D0.ugrp == "NE-first") & (D0.sw_ne_vp == 0)]),
          ("VP-only", "VP_stop", D0[(D0.vpstop_main == 1) & (D0.ugrp == "VP-only")]),
          ("Simultaneous", "VP_stop", D0[(D0.vpstop_main == 1) & (D0.ugrp == "Simultaneous")])]
    cif_old = pd.DataFrame(L["CIF"])
    for gl, anc, x in RG:
        if gl == "NE-first excl. switch":
            rows_ = CIFX[gl][1]
        else:
            g_old = "NE-only" if gl == "NE without VP" else gl
            rows_ = cif_old[(cif_old.group == g_old) & (cif_old.anchor == anc)].to_dict("records")
        T, Cc = mm.event_data(x, horizon=72.0)
        for t in TL:
            rr = [r for r in rows_ if int(r["time_h"]) == t][0]
            nr = int((T >= t).sum())
            assert nr == int(rr["n_at_risk"])
            CR.append(dict(db=DBNAME, analysis="risk_table", group=gl, anchor=anc, time_h=t, n=len(x), n_at_risk=nr,
                           cum_polyuria=int(((Cc == 1) & (T <= t)).sum()), cum_death=int(((Cc == 2) & (T <= t)).sum()),
                           cum_icu_discharge=int(((Cc == 3) & (T <= t)).sum()), cum_vaso_restart=int(((Cc == 4) & (T <= t)).sum()),
                           cum_censored=int(((Cc == 0) & (T <= t)).sum()),
                           events=int(((Cc == 1) & (T <= t)).sum()), estimate=100 * rr["cif"], lo=100 * rr["lo"], hi=100 * rr["hi"],
                           note="n_at_risk = T ≥ t (as aj / vpstop_cif); cum_* = first event of that type with T ≤ t (ties at t appear in both); "
                                "estimate = AJ CIF %% (B=%d cluster bootstrap); only administrative censoring at 72 h" % NBOOT))
    CD = pd.DataFrame(CR)
    CD.to_csv(os.path.join(outdir, "risk_table.csv"), index=False)
    say("risk table done")

    # ---- n of the basic population ---------------------------------------------------
    FIGD = dict(FIG)
    FIGD["NE-first switch (NE→VP)"] = (pop0[(pop0.ugrp == "NE-first") & (pop0.sw_ne_vp == 1)], "VP_stop")
    Hold = pd.DataFrame(L["H"])
    for gl, (pb, anc) in FIGD.items():
        ht = pd.DataFrame(hourly_table(pb, hr0, gl, anc))
        DR.append(R("basic_population_n", gl, anc, n=len(pb), events=int(ht.loc[ht.hour == 0, "n_alive_in_icu"].iloc[0]),
                    note="n = basic population (alive, in ICU at anchor; adult; no RRT; no pituitary/DI/brain death; NE without VP also no other vasopressor at T0); events = n_alive_in_icu at bin 0 start"))
        for hh in HL:
            r = ht[ht.hour == hh].iloc[0]
            DR.append(R("n_recorded", gl, anc, n=int(r["n_alive_in_icu"]), events=int(r["n_recorded"]),
                        estimate=hh, note="hour bin [anchor+h, anchor+h+1); n = alive in ICU at bin start; events = stays with urine recorded; estimate = hour"))
        if gl in ("NE-first", "VP-first", "NE without VP"):
            go = "NE-only" if gl == "NE without VP" else gl
            ho = Hold[(Hold.group == go) & (Hold.anchor == anc)].set_index("hour")
            mis = int((ho.loc[list(ht.hour), "n_recorded"].values != ht.n_recorded.values).sum())
            say("basic population n_recorded vs vpstop_hourly mismatches", gl, mis)
            REP.append(dict(check=f"basic population hourly n_recorded vs vpstop_hourly ({gl})", value=str(mis), expected="0", match=mis == 0))
    DD = pd.DataFrame(DR)[COLS]
    DD.to_csv(os.path.join(outdir, "basic_population_n.csv"), index=False)
    pd.DataFrame(REP).to_csv(os.path.join(outdir, "reproduction_checks.csv"), index=False)
    say("part 2 total %.0f s" % (time.time() - t1))
    say.write(os.path.join(outdir, "run_log.txt"))
    return dict(reproduced=True)


if __name__ == "__main__":
    main()
