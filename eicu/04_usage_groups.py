#!/usr/bin/env python3
"""
04_usage_groups.py - eICU-CRD v2.0, exploratory stop-order groups.

Regroups vasopressor stays by the pattern of vasopressin (VP) / norepinephrine (NE) use and by
stop order, anchored at liberation from all vasopressors (T0b). Post hoc and exploratory, except
where it reproduces the two-group primary analysis of 01_primary_analysis.py.

The cohort is built by eicu_cohort.build_usage_cohort (extraction, episode rule, NE-only
definition, risk set, urine, run rule and covariates are shared with the two-group cohort):
  Vasopressors = VP, NE, phenylephrine, epinephrine, dopamine, angiotensin II (dobutamine and
  milrinone are inotropes).
  NE-only: T0b = end of the first NE episode (= T0 of the two-group analysis).
  VP-containing stays: T0a(VP) = end of the first VP episode; T0b = liberation from all
     vasopressors; groups VP-only, NE-first, Simultaneous, VP-first, and, reported separately,
     VP-first+other and VP restarted (definitions in eicu_cohort.py).
  "No vasopressor running at T0b" ignores the records of the classes that are part of the
     liberation sequence. (The two-group analysis ignores only VP for VP stays, so an NE record
     within 60 min before the VP stop excludes the stay there; here such stays are kept and
     classified as Simultaneous.)
  All exclusions, covariates and outcomes are computed relative to T0b.

Optional environment variable: MICE_M (imputations, default 10).
If WORK_DIR/eicu/cohort/cohort_primary.parquet exists, stay sets are compared with the two-group
primary population.

Outputs
  OUT_DIR/eicu_04/   eICU_usage_groups_results.xlsx, run_log.txt
  WORK_DIR/eicu/cohort/cohort_usage_groups.parquet

Run:  python eicu/04_usage_groups.py   (after 01_primary_analysis.py)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # eicu_* modules; needed when Python runs in safe-path mode
from eicu_cohort import build_usage_cohort  # noqa: E402
from eicu_common import COHORT_PRIMARY, COHORT_USAGE, RUN_LOG, RunLog, cohort_path, out_dir  # noqa: E402
from eicu_extract import load_source  # noqa: E402
from eicu_stats import COV, fit_groups, fit_or, rate, smd2  # noqa: E402

MAIN = ["NE-only", "VP-only", "NE-first", "Simultaneous", "VP-first"]
ALLG = MAIN + ["VP-first+other", "VP restarted"]
G5 = ["VP-only", "NE-first", "Simultaneous", "VP-first"]
G3 = ["VP-only", "NE+VP"]
OUTC = (("polyuria_24h", "new-onset polyuria 0-24 h"), ("poly8", "onset 0-8 h"), ("poly_8_24", "onset >8-24 h (earlier onset = non-event)"))
CONT = [("age_n", "Age, y", "{:.0f}"), ("apache4", "APACHE IV score (ICU day 1)", "{:.0f}"), ("lactate_pre24", "Max lactate 24 h before T0b, mmol/L", "{:.1f}"),
        ("press_h_pre", "Time from first vasopressor start to T0b, h", "{:.1f}"), ("idx_dur_h", "First VP episode duration, h (VP groups) / first NE episode, h (NE-only)", "{:.1f}"),
        ("ne_peak_pre", "Peak NE before T0b, mcg/kg/min (convertible records)", "{:.2f}"), ("vp_peak_pre", "Peak VP before T0b, units/min (units/min records)", "{:.3f}"),
        ("fluid_pre24_l", "Recorded intake 24 h before T0b, L", "{:.1f}"), ("uo_pre6_rate", "Urine output 6 h before T0b, mL/kg/h (recorded bins)", "{:.2f}"),
        ("na_pre", "Last serum Na 24 h before T0b, mmol/L", "{:.0f}"), ("gap_t0a_t0b_h", "T0a to T0b, h", "{:.1f}")]
BIN = [("male", "Male"), ("sepsis", "Sepsis"), ("card_surg", "Cardiac surgery (APACHE dx)"), ("cvicu", "Cardiac ICU"),
       ("kdigo_0c", "KDIGO 0"), ("kdigo_1", "KDIGO 1"), ("kdigo_2", "KDIGO 2"), ("kdigo_3", "KDIGO 3"), ("kdigo_unk", "KDIGO unknown (no creatinine)"),
       ("other_press_pre", "Other vasopressor (PE/epi/dopamine/AT-II) before T0b"), ("mv_t0", "Mechanical ventilation 24 h before T0b"),
       ("steroid_pre24", "Corticosteroid 24 h before T0b"), ("inotrope_t0", "Inotrope at T0b"), ("diur_pre24", "Loop diuretic 24 h before T0b"),
       ("ne_dose_unknown", "NE dose not convertible"), ("vp_dose_unknown", "VP dose not convertible")]


def nn(x, y):
    """'events/n (%)' of column y in x."""
    return f"{int(x[y].sum())}/{len(x)} ({100 * x[y].mean():.1f}%)" if len(x) else "0/0"


def wil(k, n):
    """'k/n (%; Wilson 95% CI)'."""
    if n == 0:
        return "—"
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{k}/{n} ({100*k/n:.1f}%; 95% CI {100*lo:.1f}–{100*hi:.1f})"


def flow_table(C, flow):
    """Flow of stays and patients per usage group; `flow` holds the rows of the group definition."""
    FL = []
    for g in ALLG:
        G = C.grp == g
        steps = [("0 classified (index episode(s) reconstructed)", G), ("1 age >= 18", G & (C.adult == 1)),
                 ("2 no RRT during the stay", G & (C.adult == 1) & (C.rrt == 0)),
                 ("3 risk set: in ICU and alive at T0b", G & (C.riskset == 1)),
                 ("4 urine >= 12 of 24 bins after T0b", G & (C.evaluable == 1)),
                 ("5 no pituitary/DI/brain-death code", G & (C.evaluable == 1) & (C.excl_pit == 0)),
                 ("6 new-onset (no polyuria -12..-1 h before T0b)", G & (C.base_eval == 1)),
                 ("7 ANALYSIS: no non-sequence vasopressor running/recorded within 60 min before T0b", G & (C.primary == 1))]
        for lab_, m in steps:
            x = C[m]
            FL.append(dict(group=g, step=lab_, n_stays=len(x), n_patients=int(x.uniquepid.nunique())))
        FL.append(dict(group=g, step="  (3) excluded: death before T0b", n_stays=int((G & (C.adult == 1) & (C.rrt == 0) & (C.excl_death_pre == 1)).sum())))
        FL.append(dict(group=g, step="  (3) excluded: ICU discharge at/before T0b", n_stays=int((G & (C.adult == 1) & (C.rrt == 0) & (C.excl_disch_pre == 1)).sum())))
    x = C[(C.grp3 == "NE+VP") & (C.primary == 1)]
    FL.append(dict(group="NE+VP (pooled NE-first+Simultaneous+VP-first)", step="7 ANALYSIS", n_stays=len(x), n_patients=int(x.uniquepid.nunique())))
    return pd.DataFrame(list(flow) + FL), pd.DataFrame(FL)


def reproduction(PR):
    """Agreement with the two-group primary analysis (counts, stay sets, adjusted odds ratios)."""
    U = PR[PR.grp.isin(["VP-only", "NE-first", "Simultaneous"])]
    cur = PR[(PR.primary_rule == 1) & PR.grp.isin(["NE-only", "VP-only", "NE-first", "Simultaneous"])]
    rep = [dict(item="NE-only analysis population (two-group primary analysis: 3,422)", value=int((PR.grp == "NE-only").sum())),
           dict(item="NE-only 24-h polyuria (two-group primary analysis: 571)", value=int(PR.loc[PR.grp == "NE-only", "polyuria_24h"].sum())),
           dict(item="Union VP-only+NE-first+Simultaneous (usage-pattern classification)", value=len(U)),
           dict(item="  of which T0b = T0a(VP) and no NE record within 60 min before VP stop (= two-group primary rule; two-group primary analysis: 370)", value=int(U.primary_rule.sum())),
           dict(item="  24-h polyuria in that subset (two-group primary analysis: 81)", value=int(U.loc[U.primary_rule == 1, "polyuria_24h"].sum())),
           dict(item="  extra: T0b = T0a(VP) but NE recorded within 60 min before VP stop (the two-group primary analysis counts NE as 'running')", value=int(((U.t0 == U.t0a_vp) & (U.ne_rec_at_vpstop == 1)).sum())),
           dict(item="  extra: T0b > T0a(VP) (another vasopressor still running at VP stop; re-anchored)", value=int((U.t0 > U.t0a_vp).sum())),
           dict(item="VP stays meeting the two-group primary rule that fall into VP-first / VP-first+other / VP restarted", value=int(PR[PR.grp.isin(['VP-first', 'VP-first+other', 'VP restarted']) & (PR.primary_rule == 1)].shape[0])),
           dict(item="Union 24-h polyuria", value=rate(U.polyuria_24h))]
    primary_path = cohort_path(COHORT_PRIMARY)
    if os.path.exists(primary_path):
        o = pd.read_parquet(primary_path)
        op = o[o.primary == 1]
        a = set(op.loc[op.grp == "VP", "patientunitstayid"])
        b = set(cur.loc[cur.grp != "NE-only", "patientunitstayid"])
        rep.append(dict(item="stay-ID agreement with the VP group of the two-group primary population (|two-group|, |subset here|, |intersection|)", value=f"{len(a)}, {len(b)}, {len(a & b)}"))
        a = set(op.loc[op.grp == "NE", "patientunitstayid"])
        b = set(PR.loc[PR.grp == "NE-only", "patientunitstayid"])
        rep.append(dict(item="stay-ID agreement with the NE-only group of the two-group primary population", value=f"{len(a)}, {len(b)}, {len(a & b)}"))
    for y, lab_ in (("polyuria_24h", "24 h"), ("poly8", "0-8 h"), ("poly_8_24", ">8-24 h")):
        r_ = fit_or(cur, y, "vp", COV, label="two-group primary rule subset (2-group)")
        r_["outcome_label"] = lab_
        rep.append(r_)
        r_ = fit_or(PR[PR.grp.isin(["NE-only", "VP-only", "NE-first", "Simultaneous"])], y, "vp", COV, label="Union VP-only+NE-first+Simultaneous vs NE-only (2-group)")
        r_["outcome_label"] = lab_
        rep.append(r_)
    return pd.DataFrame(rep)


def table1(df, gcol, groups, ref="NE-only"):
    """Background characteristics per group with the standardised mean difference versus `ref`."""
    df = df.copy()
    df["kdigo_0c"] = (df.kdigo_cat == "0").astype(int)
    R = df[df[gcol] == ref]
    rows = []
    for c, lab_, f in CONT:
        r = {"variable": lab_}
        for g in groups:
            x = df.loc[df[gcol] == g, c]
            s = x.dropna()
            r[f"{g} (n={len(x)})"] = ((f + " [" + f + "–" + f + "]").format(s.median(), s.quantile(.25), s.quantile(.75)) if len(s) else "—") + f"; missing {int(x.isna().sum())}/{len(x)}"
            if g != ref:
                r[f"SMD {g}"] = smd2(x, R[c])
        rows.append(r)
    for c, lab_ in BIN:
        r = {"variable": lab_}
        for g in groups:
            x = df.loc[df[gcol] == g, c]
            r[f"{g} (n={len(x)})"] = f"{int(x.sum())}/{len(x)} ({100 * x.mean():.1f}%)"
            if g != ref:
                r[f"SMD {g}"] = smd2(x, R[c])
        rows.append(r)
    return pd.DataFrame(rows)


def gap_tables(C, PR):
    """Distribution of the intervals between the stops and T0b, and polyuria between the VP stop and T0b."""
    rows = []
    for popn, P_ in (("classified (all stays)", C), ("analysis population", PR)):
        for g in ["VP-only", "NE-first", "Simultaneous", "VP-first", "VP-first+other", "VP restarted"]:
            x = P_.loc[P_.grp == g]
            for var, vl in (("gap_t0a_t0b_h", "T0a (earlier of NE/VP stop) to T0b, h"), ("gap_vpstop_t0b_h", "VP stop to T0b, h"),
                            ("ne_minus_vp_stop_min", "last NE stop minus VP stop, min")):
                s = x[var].dropna()
                if not len(s):
                    continue
                q = s.quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]).round(2).tolist()
                rows.append(dict(population=popn, group=g, variable=vl, n=len(s), median=round(s.median(), 2), q1=round(s.quantile(.25), 2), q3=round(s.quantile(.75), 2),
                                 deciles_10_90=" / ".join(map(str, q)), n_zero=int((s == 0).sum()), min=round(s.min(), 2), max=round(s.max(), 2)))
    gaps = pd.DataFrame(rows)
    rows = []
    for g in ["VP-first", "VP-first+other", "Simultaneous", "VP-only", "NE-first"]:
        G = (C.grp == g)
        st5 = C[G & (C.evaluable == 1) & (C.excl_pit == 0)]       # before the new-onset step
        rows.append(dict(group=g, population="evaluable at T0b, no pituitary code (step 5)", n=len(st5),
                         poly_between_T0a_T0b=nn(st5, "poly_between_t0a_t0b"),
                         excluded_by_new_onset_rule_at_T0b=nn(st5, "poly_pre12"),
                         excl_new_onset_and_poly_between=int(((st5.poly_pre12 == 1) & (st5.poly_between_t0a_t0b == 1)).sum()),
                         median_gap_h=round(st5.gap_vpstop_t0b_h.median(), 2)))
        pr = PR[PR.grp == g]
        rows.append(dict(group=g, population="analysis population", n=len(pr), poly_between_T0a_T0b=nn(pr, "poly_between_t0a_t0b"),
                         excluded_by_new_onset_rule_at_T0b="n/a (already excluded)", median_gap_h=round(pr.gap_vpstop_t0b_h.median(), 2)))
    return gaps, pd.DataFrame(rows)


def urine_completeness(C):
    """Urine charting completeness and ICU time remaining after T0b in the risk set without the urine requirement."""
    rows = []
    RSP = C[(C.riskset == 1) & (C.excl_pit == 0) & (C.poly_pre12 == 0) & (C.press_t0 == 0)]
    for g in ALLG:
        x = RSP[RSP.grp == g]
        s = x.obs_h_24
        rows.append(dict(group=g, risk_set_n=len(x), evaluable=nn(x, "urine_evaluable"),
                         bins_0_24_median_IQR=f"{s.median():.0f} [{s.quantile(.25):.0f}–{s.quantile(.75):.0f}]" if len(x) else "—",
                         no_urine_0_24=int((s == 0).sum()), hourly_pairs_post_median=x.post_pairs.median(),
                         evaluable_bins_median_IQR=(lambda t: f"{t.median():.0f} [{t.quantile(.25):.0f}–{t.quantile(.75):.0f}]")(x.loc[x.urine_evaluable == 1, "obs_h_24"]) if x.urine_evaluable.sum() else "—"))
    x = RSP[RSP.grp3 == "NE+VP"]
    s = x.obs_h_24
    rows.append(dict(group="NE+VP", risk_set_n=len(x), evaluable=nn(x, "urine_evaluable"), bins_0_24_median_IQR=f"{s.median():.0f} [{s.quantile(.25):.0f}–{s.quantile(.75):.0f}]",
                     no_urine_0_24=int((s == 0).sum()), hourly_pairs_post_median=x.post_pairs.median()))
    for lab_, m in (("Simultaneous, NE and VP stop at identical minute", (RSP.grp == "Simultaneous") & (RSP.same_stop_minute == 1)),
                    ("Simultaneous, non-identical minute", (RSP.grp == "Simultaneous") & (RSP.same_stop_minute == 0))):
        x = RSP[m]
        s = x.obs_h_24
        rows.append(dict(group=lab_, risk_set_n=len(x), evaluable=nn(x, "urine_evaluable"), bins_0_24_median_IQR=f"{s.median():.0f} [{s.quantile(.25):.0f}–{s.quantile(.75):.0f}]" if len(x) else "—",
                         no_urine_0_24=int((s == 0).sum()), hourly_pairs_post_median=x.post_pairs.median()))
    completeness = pd.DataFrame(rows)
    rows = []
    for g in ALLG:
        for sub, m in (("all", RSP.grp == g), ("identical NE/VP stop minute", (RSP.grp == g) & (RSP.same_stop_minute == 1))):
            x = RSP[m]
            if not len(x):
                continue
            rows.append(dict(group=g, subset=sub, n=len(x), icu_h_after_T0b_median_IQR=f"{x.obs_after_t0_h.median():.1f} [{x.obs_after_t0_h.quantile(.25):.1f}–{x.obs_after_t0_h.quantile(.75):.1f}]",
                             icu_discharge_le_24h=nn(x.assign(_d=(x.obs_after_t0_h <= 24).astype(int)), "_d"), stop_charted_rate0=nn(x.assign(_s=x.idx_stop_charted.astype(int)), "_s")))
    completeness["note"] = "risk set = in ICU & alive at T0b, adult, no RRT, no pituitary code, new-onset, no non-sequence vasopressor at T0b (no urine requirement)"
    return completeness, pd.DataFrame(rows)


def outcome_models(PR, PR8):
    """Group odds ratios (crude and adjusted) for the polyuria outcomes, 5-group and 3-group."""
    res = []
    for gcol, gl, frame in (("grp", G5, "5-group"), ("grp3", G3, "3-group")):
        for y, yl in OUTC + (("poly_8_24|risk8", "onset >8-24 h | free of polyuria and in ICU at 8 h"),):
            yy = y.split("|")[0]
            D = PR8 if "|" in y else PR
            for adj, covs, mice in (("crude", [], False), ("adjusted", COV, True)):
                for r_ in fit_groups(D, yy, gcol, gl, covs=covs, mice=mice, label=f"{frame} {adj}"):
                    r_.update(outcome_label=yl, risk_set=("PR8: analysis population free of polyuria at 8 h and ICU stay > 8 h after T0b" if "|" in y else "analysis population (all stays; earlier onset = non-event)"))
                    res.append(r_)
    # sensitivity: VP-first + VP-first+other merged
    PRm = PR.assign(grp_m=PR.grp.replace({"VP-first+other": "VP-first"}))
    for y, yl in OUTC:
        for r_ in fit_groups(PRm, y, "grp_m", G5, label="5-group adjusted, VP-first incl. VP-first+other (sensitivity)"):
            r_.update(outcome_label=yl)
            res.append(r_)
    return pd.DataFrame(res)


def outcome_rates(PR, PR8):
    """Crude outcome rates per group, including the groups reported separately."""
    rows = []
    for g in ALLG + ["NE+VP"]:
        x = PR[(PR.grp == g) | (PR.grp3 == g)] if g == "NE+VP" else PR[PR.grp == g]
        x8 = PR8[(PR8.grp3 == g)] if g == "NE+VP" else PR8[PR8.grp == g]
        rows.append(dict(group=g, n_stays=len(x), n_patients=int(x.uniquepid.nunique()), poly_24h=nn(x, "polyuria_24h"), poly_0_8=nn(x, "poly8"),
                         poly_8_24_all=nn(x, "poly_8_24"), poly_8_24_conditional=nn(x8, "poly_8_24"),
                         onset_median_h=x.loc[x.polyuria_24h == 1, "uo_onset_h"].median()))
    for lab_, m in (("Simultaneous, identical stop minute", (PR.grp == "Simultaneous") & (PR.same_stop_minute == 1)),
                    ("Simultaneous, non-identical minute", (PR.grp == "Simultaneous") & (PR.same_stop_minute == 0)),
                    ("two-group primary VP group (T0b = VP stop, no NE record <= 60 min before VP stop)", (PR.primary_rule == 1) & (PR.grp != "NE-only")),
                    ("union VP-only+NE-first+Simultaneous NOT meeting the two-group primary rule", (PR.primary_rule == 0) & PR.grp.isin(["VP-only", "NE-first", "Simultaneous"]))):
        x = PR[m]
        x8 = x[(x.poly8 == 0) & (x.obs_after_t0_h > 8)]
        rows.append(dict(group=lab_, n_stays=len(x), n_patients=int(x.uniquepid.nunique()), poly_24h=nn(x, "polyuria_24h"), poly_0_8=nn(x, "poly8"),
                         poly_8_24_all=nn(x, "poly_8_24"), poly_8_24_conditional=nn(x8, "poly_8_24"), onset_median_h=x.loc[x.polyuria_24h == 1, "uo_onset_h"].median()))
    return pd.DataFrame(rows)


def sodium(PR):
    """Hypernatraemia within 72 h: counts per group and group odds ratios."""
    rows, nres = [], []
    for g in ALLG + ["NE+VP"]:
        x = PR[PR.grp3 == g] if g == "NE+VP" else PR[PR.grp == g]
        S_ = x[(x.na_pre_max < 150) & (x.na_post72_n >= 1)]
        rows.append(dict(group=g, analysis_n=len(x), no_Na_24h_before_T0b=int(x.na_pre_max.isna().sum()), excluded_preT0b_Na_ge150=int((x.na_pre_max >= 150).sum()),
                         no_Na_0_72h_after=int(((x.na_pre_max < 150) & (x.na_post72_n == 0)).sum()), Na_denominator=len(S_),
                         Na150_72h=f"{int((S_.na_post72_max >= 150).sum())}/{len(S_)} ({100*(S_.na_post72_max >= 150).mean():.1f}%)" if len(S_) else "—",
                         Na_tests_0_72h_median_IQR=f"{S_.na_post72_n.median():.0f} [{S_.na_post72_n.quantile(.25):.0f}–{S_.na_post72_n.quantile(.75):.0f}]" if len(S_) else "—"))
    SN = PR[(PR.na_pre_max < 150) & (PR.na_post72_n >= 1)].copy()
    SN["na150_72h"] = (SN.na_post72_max >= 150).astype(int)
    SN["log_na_n"] = np.log(SN.na_post72_n)
    for gcol, gl, frame in (("grp", G5, "5-group"), ("grp3", G3, "3-group")):
        for adj, covs, mice in (("crude", [], False), ("adjusted MICE", COV, True), ("adjusted MICE + log(Na tests 0-72 h)", COV + ["log_na_n"], True)):
            for r_ in fit_groups(SN, "na150_72h", gcol, gl, covs=covs, mice=mice, label=f"{frame} {adj}"):
                r_.update(outcome_label="Na >= 150 within 72 h (all Na < 150 in 24 h before T0b; >= 1 Na 0-72 h)")
                nres.append(r_)
    return pd.DataFrame(rows), pd.DataFrame(nres)


def restart(PR):
    """Vasopressor restart and outcomes counted only before a restart."""
    rows = []
    for g in ALLG + ["NE+VP"]:
        x = PR[PR.grp3 == g] if g == "NE+VP" else PR[PR.grp == g]
        rows.append(dict(group=g, n=len(x), restart_24h=nn(x, "restart24"), restart_2h=f"{int((x.restart_h <= 2).sum())}/{len(x)}",
                         poly24_before_restart=nn(x, "poly24_before_restart"), poly8_before_restart=nn(x, "poly8_before_restart")))
    rr = []
    for gcol, gl, frame in (("grp", G5, "5-group"), ("grp3", G3, "3-group")):
        for y, yl in (("restart24", "vasopressor restart within 24 h (crude)"),):
            for r_ in fit_groups(PR, y, gcol, gl, covs=[], mice=False, label=f"{frame} crude"):
                r_.update(outcome_label=yl)
                rr.append(r_)
        for y, yl in (("poly24_before_restart", "polyuria 0-24 h counted only before restart"), ("poly8_before_restart", "0-8 h, before restart"), ("poly824_before_restart", ">8-24 h, before restart")):
            for r_ in fit_groups(PR, y, gcol, gl, label=f"{frame} adjusted"):
                r_.update(outcome_label=yl)
                rr.append(r_)
    return pd.DataFrame(rows), pd.DataFrame(rr)


def anchor_vp_stop(C):
    """Exploratory counts with the VP stop as the anchor."""
    rows = []
    for g in ["VP-first", "VP-first+other", "NE-first", "Simultaneous", "VP-only"]:
        x0 = C[C.grp == g]
        x1 = x0[(x0.a_riskset == 1)]
        x2 = x1[x1.a_obs_h_24 >= 12]
        x3 = x2[x2.excl_pit == 0]
        x4 = x3[x3.a_poly_pre12 == 0]
        k24 = int((x4.a_onset_h <= 24).sum())
        k8 = int((x4.a_onset_h <= 8).sum())
        k824 = int(((x4.a_onset_h > 8) & (x4.a_onset_h <= 24)).sum())
        x8 = x4[~(x4.a_onset_h <= 8) & (x4.a_obs_after_h > 8)]
        k824c = int(((x8.a_onset_h > 8) & (x8.a_onset_h <= 24)).sum())
        rows.append(dict(group=g, classified=len(x0), at_risk_at_VP_stop=len(x1), urine_ge12_bins=len(x2), no_pit=len(x3),
                         new_onset_at_VP_stop=len(x4), poly_0_24=wil(k24, len(x4)), poly_0_8=wil(k8, len(x4)), poly_8_24_all=wil(k824, len(x4)),
                         poly_8_24_conditional=wil(k824c, len(x8)),
                         poly_between_VPstop_T0b=wil(int(x4.poly_between_t0a_t0b.sum()), len(x4)),
                         median_VPstop_to_T0b_h=round(x4.gap_vpstop_t0b_h.median(), 2)))
    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore")
    t_start = time.time()
    log = RunLog()
    say = log.say
    out = out_dir("04")

    src = load_source(say)
    C, aux = build_usage_cohort(src, say)
    C.to_parquet(cohort_path(COHORT_USAGE), index=False)
    say("cohort built in %.0f s" % (time.time() - t_start))

    C["grp3"] = C.grp.map({"NE-only": "NE-only", "VP-only": "VP-only", "NE-first": "NE+VP", "Simultaneous": "NE+VP", "VP-first": "NE+VP"})
    TAB = {}
    TAB["01_flow"], FL = flow_table(C, aux.flow)
    say(FL.pivot_table(index="step", columns="group", values="n_stays", aggfunc="first").to_string())
    PR = C[C.primary == 1].copy()
    TAB["02_reproduction"] = reproduction(PR)
    say(TAB["02_reproduction"].to_string(index=False))
    TAB["03_background_5grp"] = table1(PR, "grp", MAIN + ["VP-first+other", "VP restarted"])
    TAB["03b_background_3grp"] = table1(PR[PR.grp3.notna()], "grp3", ["NE-only", "VP-only", "NE+VP"])
    TAB["04_gap_T0a_T0b"], TAB["04b_VPfirst_between_T0a_T0b"] = gap_tables(C, PR)
    TAB["05_urine_completeness"], TAB["05b_icu_time_after_T0b"] = urine_completeness(C)
    PR8 = PR[(PR.poly8 == 0) & (PR.obs_after_t0_h > 8)]
    TAB["06_outcomes_OR"] = outcome_models(PR, PR8)
    say(TAB["06_outcomes_OR"][["analysis", "outcome", "group", "events_group", "events_ref", "OR_95CI", "method"]].to_string(index=False))
    TAB["06a_outcome_rates"] = outcome_rates(PR, PR8)
    TAB["06c_primary_rule_by_group"] = pd.crosstab(PR.grp, PR.primary_rule.map({1: "meets two-group primary rule", 0: "does not"})).reset_index()
    say(TAB["06a_outcome_rates"].to_string(index=False))
    TAB["07a_sodium_counts"], TAB["07b_sodium_OR"] = sodium(PR)
    say(TAB["07b_sodium_OR"][["analysis", "group", "events_group", "OR_95CI"]].to_string(index=False))
    TAB["08a_restart"], TAB["08b_restart_sensitivity_OR"] = restart(PR)
    TAB["09_anchor_VP_stop_exploratory"] = anchor_vp_stop(C)
    say(TAB["09_anchor_VP_stop_exploratory"].to_string(index=False))

    NOTE = pd.DataFrame({"note": [
        "04_usage_groups.py, eICU-CRD v2.0. All analyses post hoc / exploratory except the reproduction of the 2-group primary.",
        "Analysis population = adult, no RRT, in ICU & alive at T0b, >=12 urine bins 0-24 h, no pituitary/DI/brain-death code, no polyuria -12..-1 h before T0b, no non-sequence vasopressor running/recorded within 60 min before T0b.",
        "Groups: see script header. VP-first+other and VP restarted are reported separately and are not part of the 5-group/3-group models.",
        "Adjusted ORs: covariate set of the two-group primary analysis (COV), MICE m=10 for Na/MAP/APACHE IV, logistic with hospital-clustered SE; Firth (model-based SE) when any group in the model has <10 events.",
        "Risk sets: '>8-24 h (all)' = analysis population, onset <=8 h counted as non-event; 'conditional' = free of polyuria at 8 h and ICU stay >8 h after T0b.",
        "Firth penalisation applies to all coefficients; no clustering in Firth models."]})
    with pd.ExcelWriter(os.path.join(out, "eICU_usage_groups_results.xlsx")) as xw:
        NOTE.to_excel(xw, sheet_name="00_notes", index=False)
        for k in sorted(TAB):
            TAB[k].to_excel(xw, sheet_name=k[:31], index=False)
    log.write(os.path.join(out, RUN_LOG))
    say("done in %.0f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
