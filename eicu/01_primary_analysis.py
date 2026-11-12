#!/usr/bin/env python3
"""
01_primary_analysis.py - eICU-CRD v2.0, primary two-group analysis (replication cohort).

Post hoc analysis. Unit = ICU stay (patients = uniquepid reported alongside); standard errors
clustered on hospital.

  VP group       stays with a vasopressin episode in infusionDrug; T0 = end of the first VP episode
  NE-only group  stays with a norepinephrine episode and no vasopressin anywhere (infusionDrug
                 name, medication name or HICL 2839, treatment string); T0 = end of the first
                 NE episode
  The episode, urine, polyuria, risk-set and exclusion rules are described in eicu_cohort.py.
  Primary population = evaluable, new onset, no pituitary / diabetes insipidus / brain-death
  code and no vasopressor (vasopressin, norepinephrine, phenylephrine, epinephrine, dopamine,
  angiotensin II) running at T0; dobutamine and milrinone are a covariate only.

Analyses: flow and descriptive tables, adjusted logistic regression (multiple imputation,
hospital-clustered SE), inverse probability weighting for the average treatment effect in the
treated with a hospital-cluster bootstrap, Aalen-Johansen cumulative incidence, cause-specific
Cox models, inverse probability of evaluability weighting, comparison within the VP group,
sodium and clinical consequences, sensitivity, subgroup and robustness analyses.

Optional environment variables: NBOOT (bootstrap replicates, default 500), MICE_M (imputations, 10).
The large raw tables are filtered once into WORK_DIR/eicu/ex (see eicu_extract.py).

Outputs
  OUT_DIR/eicu_01/   eICU_primary_results.xlsx, fig_data_forest_or.csv, fig_data_cif_hourly.csv,
                     fig_data_onset_windows.csv, run_log.txt
  WORK_DIR/eicu/cohort/cohort_primary.parquet

Run:  python eicu/01_primary_analysis.py
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # eicu_* modules; needed when Python runs in safe-path mode
from eicu_cohort import build_primary_cohort  # noqa: E402
from eicu_common import COHORT_PRIMARY, RUN_LOG, RunLog, cohort_path, env_int, out_dir  # noqa: E402
from eicu_extract import load_source  # noqa: E402
from eicu_stats import (COV, LAB, MICE_M, MISS, aj_cif, cr_times, evalue, fit_or, fmt, hosp_boot, logit_cluster,  # noqa: E402
                        ps_weights, rate, smd)

NBOOT = env_int("NBOOT", 500)
OUTCOMES = ("polyuria_24h", "poly8", "poly_8_24")
ADJ = "adjusted (MICE, hospital-clustered)"
SHEET_ORDER = ["00_README", "01_flow", "02_VP_counts", "03_table1_primary", "03b_table1_riskset_primary", "04_crude_rates",
               "04b_onset_windows", "05_primary_OR", "06_IPW_ATT", "07_SMD_primary", "09_csCox_time_split", "10_CIF_8h_24h",
               "11_IPEW_riskset", "12_internal_VP", "13_sodium_clinical", "14_sensitivity", "15_subgroups", "16_cointerventions",
               "16b_urine_charting", "17_NE_dose_units", "18_coef_primary", "19_robustness"]


def vp_counts(C):
    """Counts of the VP group from the index episode to the evaluable stays."""
    V = C[C.grp == "VP"]
    rep = [dict(item="VP stays with episode", value=len(V)),
           dict(item="adult", value=int(V.adult.sum())),
           dict(item="adult, no RRT", value=int(((V.adult == 1) & (V.rrt == 0)).sum())),
           dict(item="  excluded: death recorded before T0", value=int(((V.adult == 1) & (V.rrt == 0) & (V.excl_death_pre == 1)).sum())),
           dict(item="  excluded: ICU discharge at/before T0", value=int(((V.adult == 1) & (V.rrt == 0) & (V.excl_disch_pre == 1)).sum())),
           dict(item="risk set", value=int(V.riskset.sum())),
           dict(item="evaluable", value=int(V.evaluable.sum())),
           dict(item="24-h polyuria in evaluable, %", value=round(100 * V.loc[V.evaluable == 1, "polyuria_24h"].mean(), 1))]
    return pd.DataFrame(rep)


def flow_table(C, flow):
    """Flow of stays and patients per group; `flow` holds the rows of the group definition."""
    flow = list(flow)

    def npt(mask):
        x = C[mask]
        return len(x), x.uniquepid.nunique()

    for g, gl in (("VP", "VP"), ("NE", "NE-only")):
        G = C.grp == g
        steps = [("Index episode reconstructed (analysis start)", G),
                 ("Age >= 18", G & (C.adult == 1)),
                 ("No RRT during the stay", G & (C.adult == 1) & (C.rrt == 0)),
                 ("Risk set: in ICU and alive at T0", G & (C.riskset == 1)),
                 ("Evaluable: >= 12 urine bins in 0-24 h", G & (C.evaluable == 1)),
                 ("No pituitary/DI/brain-death code", G & (C.evaluable == 1) & (C.excl_pit == 0)),
                 ("New-onset (no polyuria -12..-1 h)", G & (C.base_eval == 1)),
                 ("PRIMARY: no vasopressor running at T0", G & (C.primary == 1)),
                 ("Risk-set primary (no evaluability requirement)", G & (C.rs_primary == 1))]
        for lab_, msk in steps:
            n, p_ = npt(msk)
            flow.append(dict(step=lab_, group=gl, n_stays=n, n_patients=p_))
        flow.append(dict(step="  excluded from risk set: death before T0", group=gl, n_stays=int((G & (C.adult == 1) & (C.rrt == 0) & (C.excl_death_pre == 1)).sum())))
        flow.append(dict(step="  excluded from risk set: ICU discharge at/before T0", group=gl, n_stays=int((G & (C.adult == 1) & (C.rrt == 0) & (C.excl_disch_pre == 1)).sum())))
    return pd.DataFrame(flow)


def table1(df, by="grp_label"):
    """Baseline characteristics by group with standardised mean differences."""
    rows = []

    def cont(c, lab_, f="{:.1f}"):
        r = {"variable": lab_}
        for g, x in df.groupby(by):
            s = x[c].dropna()
            r[f"{g} (n={len(x)})"] = (f + " [" + f + "–" + f + "]").format(s.median(), s.quantile(.25), s.quantile(.75)) + (f"; missing {x[c].isna().sum()}" if x[c].isna().any() else "")
        r["SMD"] = round(smd(df[c].fillna(df[c].median()), df.vp), 2)
        rows.append(r)

    def binr(c, lab_):
        r = {"variable": lab_}
        for g, x in df.groupby(by):
            r[f"{g} (n={len(x)})"] = f"{int(x[c].sum())} ({100 * x[c].mean():.1f})"
        r["SMD"] = round(smd(df[c], df.vp), 2)
        rows.append(r)

    cont("age_n", "Age, y", "{:.0f}")
    binr("male", "Male")
    cont("apache4", "APACHE IV score", "{:.0f}")
    binr("cvicu", "Cardiac ICU")
    binr("card_surg", "Cardiac surgery (APACHE dx)")
    binr("sepsis", "Sepsis")
    binr("elective", "Elective surgical admission")
    binr("mv_t0", "Mechanical ventilation at T0")
    binr("steroid_pre24", "Corticosteroid 24 h before T0")
    binr("inotrope_t0", "Inotrope at T0")
    binr("other_press_pre", "Vasopressor other than VP/NE before T0")
    cont("press_h_pre", "Vasopressor time before T0, h")
    cont("idx_dur_h", "Index (first VP / first NE) episode, h")
    cont("ne_peak_pre", "Peak NE before T0, mcg/kg/min (convertible)", "{:.2f}")
    binr("ne_dose_unknown", "NE dose not convertible (ml/hr etc.)")
    for k, l_ in (("kdigo_1", "KDIGO 1"), ("kdigo_2", "KDIGO 2"), ("kdigo_3", "KDIGO 3"), ("kdigo_unk", "KDIGO unknown")):
        binr(k, l_)
    cont("na_pre", "Na before T0, mmol/L", "{:.0f}")
    cont("map_base", "MAP 6 h before T0, mm Hg", "{:.0f}")
    cont("lactate_pre24", "Max lactate 24 h before T0", "{:.1f}")
    cont("fluid_pre24_l", "Recorded intake 24 h before T0, L")
    binr("diur_pre24", "Loop diuretic 24 h before T0")
    cont("urine_gap_med_min", "Median urine charting interval, min", "{:.0f}")
    cont("obs_h_24", "Urine bins recorded 0-24 h", "{:.0f}")
    binr("hosp_death", "Hospital death (descriptive)")
    return pd.DataFrame(rows)


def crude_rates(C, PR, BE):
    """Crude frequencies of the outcomes in four populations."""
    rows = []
    for popn, P_ in (("Evaluable, all (risk set, >= 12 urine bins in 0-24 h; no further exclusion)", C[C.evaluable == 1]),
                     ("Evaluable, new-onset, pit. excl. (secondary pop.)", BE),
                     ("PRIMARY (no vasopressor at T0)", PR), ("Secondary: vasopressor running at T0", BE[BE.press_t0 == 1])):
        for g, x in P_.groupby("grp_label"):
            rows.append(dict(population=popn, group=g, n_stays=len(x), n_patients=x.uniquepid.nunique(),
                             poly24=rate(x.polyuria_24h), poly_0_8=rate(x.poly8), poly_8_24=rate(x.poly_8_24),
                             marked24=rate(x.marked_24h), uo24_def=rate(x.uo24_def), poly72=rate(x.polyuria_72h),
                             pre_T0_polyuria=rate(x.poly_pre12), onset_median_h=x.loc[x.polyuria_24h == 1, "uo_onset_h"].median()))
    return pd.DataFrame(rows)


def onset_windows(PR, BE):
    """Distribution of the onset hour (figure data)."""
    od = []
    for popn, P_ in (("primary", PR), ("secondary_all", BE)):
        for g, x in P_.groupby("grp_label"):
            for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 24)):
                k = int(((x.uo_onset_h > lo) & (x.uo_onset_h <= hi)).sum())
                od.append(dict(population=popn, group=g, window=f"{lo}-{hi} h", events=k, n=len(x), pct=100 * k / len(x)))
    return pd.DataFrame(od)


def primary_models(PR, BE, forest, say):
    """Adjusted and crude odds ratios for 0-24 h, 0-8 h and >8-24 h in the primary and the
    secondary population, plus the >8-24 h model conditional on being at risk at 8 h.
    Returns the table of odds ratios and the coefficients of the primary 24-h model."""
    rows = []
    coef = None
    for y, ylab in (("polyuria_24h", "polyuria 0-24 h"), ("poly8", "onset 0-8 h"), ("poly_8_24", "onset >8-24 h")):
        for popn, P_ in (("PRIMARY (no vasopressor at T0)", PR), ("Secondary: all evaluable new-onset (vasopressor at T0 allowed)", BE)):
            row, res = fit_or(P_, y, "vp", COV, label=popn, full=True)
            crude = fit_or(P_, y, "vp", [], mice=False, label=popn)
            row["crude_OR_95CI"] = crude["aOR_95CI"]
            row["outcome_label"] = ylab
            row["rate_VP"] = rate(P_.loc[P_.vp == 1, y])
            row["rate_NE"] = rate(P_.loc[P_.vp == 0, y])
            rows.append(row)
            forest.append(dict(row, model=ADJ))
            forest.append(dict(crude, outcome_label=ylab, model="crude"))
            if popn.startswith("PRIMARY") and y == "polyuria_24h":
                coef = res.copy()
                coef.index = [LAB.get(i, i) for i in coef.index]
                coef["aOR_95CI"] = [fmt(a, b, c) for a, b, c in zip(coef.OR, coef.lo, coef.hi)]
                coef = coef.reset_index().rename(columns={"index": "term"})
    say(pd.DataFrame(rows)[["analysis", "outcome", "n", "events", "rate_VP", "rate_NE", "crude_OR_95CI", "aOR_95CI"]].to_string(index=False))
    # >8-24 h conditional on being free of polyuria and in the ICU at 8 h
    P8 = PR[(PR.poly8 == 0) & (PR.obs_after_t0_h > 8)]
    r8 = fit_or(P8, "poly_8_24", "vp", COV, label="PRIMARY, at risk at 8 h (conditional)")
    r8["outcome_label"] = "onset >8-24 h | free and in ICU at 8 h"
    rows.append(r8)
    forest.append(dict(r8, model=ADJ))
    return pd.DataFrame(rows), coef


def att_effect(df, y):
    """Outcome risk in the VP group and ATT-weighted risk in the NE-only group."""
    w, _ = ps_weights(df, COV)
    t = df.vp.values == 1
    p1 = np.average(df[y].values[t], weights=w.values[t])
    p0 = np.average(df[y].values[~t], weights=w.values[~t])
    return p1, p0


def ipw_att(PR, w_pr, Xps):
    """ATT-weighted risks, risk difference, risk ratio and odds ratio with hospital-cluster
    bootstrap intervals (propensity score refitted in every replicate), and the covariate
    balance before and after weighting."""
    ipw_rows = []
    for y in OUTCOMES:
        p1, p0 = att_effect(PR, y)
        bs = hosp_boot(PR, lambda d: att_effect(d, y), B=NBOOT)
        rd_ = bs[:, 0] - bs[:, 1]
        rr_ = bs[:, 0] / bs[:, 1]
        or_ = (bs[:, 0] / (1 - bs[:, 0])) / (bs[:, 1] / (1 - bs[:, 1]))
        RR = p1 / p0
        OR = (p1 / (1 - p1)) / (p0 / (1 - p0))
        rrlo, rrhi = np.nanpercentile(rr_, [2.5, 97.5])
        ipw_rows.append(dict(population="PRIMARY", outcome=y, n_VP=int(PR.vp.sum()), n_NE=int((PR.vp == 0).sum()),
                             risk_VP=round(100 * p1, 1), risk_NE_weighted=round(100 * p0, 1),
                             RD_pct_95CI=f"{100 * (p1 - p0):.1f} ({100 * np.nanpercentile(rd_, 2.5):.1f} to {100 * np.nanpercentile(rd_, 97.5):.1f})",
                             RR_95CI=fmt(RR, rrlo, rrhi), OR_95CI=fmt(OR, *np.nanpercentile(or_, [2.5, 97.5])),
                             E_value_point=round(evalue(RR), 2), E_value_CI=round(evalue(rrlo) if rrlo > 1 else (evalue(rrhi) if rrhi < 1 else 1.0), 2),
                             n_boot_ok=len(bs), bootstrap="hospital-cluster, PS refitted each replicate",
                             NE_weight_trim="99th percentile", ESS_NE=round(w_pr[PR.vp == 0].sum() ** 2 / (w_pr[PR.vp == 0] ** 2).sum(), 0)))
    sm_rows = []
    for c in Xps.columns:
        sm_rows.append(dict(covariate=LAB.get(c, c), SMD_unweighted=round(smd(Xps[c], PR.vp), 3), SMD_ATT_weighted=round(smd(Xps[c], PR.vp, w_pr), 3)))
    return pd.DataFrame(ipw_rows), pd.DataFrame(sm_rows)


def cif_difference(d, ipw):
    """Cumulative incidence at 8 and 24 h in the VP group and in the NE-only group."""
    tt, ss = cr_times(d)
    ww = ps_weights(d, COV)[0].values if ipw else np.ones(len(d))
    a = aj_cif(tt[d.vp.values == 1], ss[d.vp.values == 1], ww[d.vp.values == 1], np.array([8, 24]))
    b = aj_cif(tt[d.vp.values == 0], ss[d.vp.values == 0], ww[d.vp.values == 0], np.array([8, 24]))
    return np.concatenate([a, b])


def cumulative_incidence(PR, RS, w_pr, w_rs):
    """Hourly Aalen-Johansen cumulative incidence (competing: ICU death, ICU discharge,
    vasopressor restart), crude and ATT-weighted, and the differences at 8 and 24 h with a
    hospital-cluster bootstrap. Returns the hourly curves and the summary table."""
    grid = np.arange(0, 73)
    cif_rows, cif_tab = [], []
    for popn, P_, W_ in (("PRIMARY evaluable", PR, w_pr), ("Risk-set primary", RS, w_rs)):
        t, s = cr_times(P_)
        for wt, WW in (("crude", None), ("ATT-IPW", W_.values)):
            for g in (1, 0):
                m_ = P_.vp.values == g
                cif = aj_cif(t[m_], s[m_], None if WW is None else WW[m_], grid)
                for h, v in zip(grid, cif):
                    cif_rows.append(dict(population=popn, weighting=wt, group="VP" if g else "NE-only", hour=int(h), cif=v))
        for ipw in (False, True):
            est = cif_difference(P_, ipw)
            bs = hosp_boot(P_, lambda d: cif_difference(d, ipw), B=NBOOT)
            for j, hh in enumerate((8, 24)):
                dd = bs[:, j] - bs[:, j + 2]
                cif_tab.append(dict(population=popn, weighting="ATT-IPW" if ipw else "crude", hour=hh,
                                    n_VP=int(P_.vp.sum()), n_NE=int((P_.vp == 0).sum()),
                                    CIF_VP_pct=round(100 * est[j], 1), CIF_NE_pct=round(100 * est[j + 2], 1),
                                    diff_pct_95CI=f"{100 * (est[j] - est[j + 2]):.1f} ({100 * np.percentile(dd, 2.5):.1f} to {100 * np.percentile(dd, 97.5):.1f})",
                                    events_VP_by_h=int(((s == 1) & (t <= hh) & (P_.vp.values == 1)).sum()),
                                    events_NE_by_h=int(((s == 1) & (t <= hh) & (P_.vp.values == 0)).sum())))
        # competing-event counts by 24 h
        for g in (1, 0):
            m_ = P_.vp.values == g
            cif_tab.append(dict(population=popn, weighting="event counts by 24 h", hour=24, n_VP=int(m_.sum()) if g else None,
                                n_NE=None if g else int(m_.sum()),
                                CIF_VP_pct=None, CIF_NE_pct=None,
                                diff_pct_95CI="%s: polyuria %d, death %d, discharge %d, restart %d" % (
                                    "VP" if g else "NE-only", ((s == 1) & (t <= 24) & m_).sum(), ((s == 2) & (t <= 24) & m_).sum(),
                                    ((s == 3) & (t <= 24) & m_).sum(), ((s == 4) & (t <= 24) & m_).sum())))
    return pd.DataFrame(cif_rows), pd.DataFrame(cif_tab)


def vp_riskset_cif(C):
    """24-h cumulative incidence in all risk-set stays of the VP group (restart not a competing event)."""
    VRS = C[(C.grp == "VP") & (C.riskset == 1)]
    rows = []
    for lab_, use_dis in (("death competing, discharge censored", False), ("death and discharge competing", True)):
        tt, ss = cr_times(VRS, use_restart=False)
        if not use_dis:
            ss = np.where(ss == 3, 0, ss)
        rows.append(pd.DataFrame([dict(item=f"VP risk set 24-h CIF, {lab_}", value=round(100 * aj_cif(tt, ss, None, np.array([24]))[0], 1))]))
    return rows


def cause_specific_cox(PR, RS, w_pr, w_rs):
    """Cause-specific Cox models for the windows 0-8 h and >8-24 h, crude and ATT-weighted."""
    cox_rows = []
    for popn, P_, W_ in (("PRIMARY evaluable", PR, w_pr), ("Risk-set primary", RS, w_rs)):
        t, s = cr_times(P_)
        D = pd.DataFrame({"t": t, "e": (s == 1).astype(int), "vp": P_.vp.values, "w": W_.values, "hosp": P_.hospitalid.values})
        for lo, hi in ((0, 8), (8, 24)):
            d = D[D.t > lo].copy()
            d["entry"] = lo
            d["e"] = ((d.e == 1) & (d.t <= hi)).astype(int)
            d["t"] = np.minimum(d.t, hi)
            d.loc[d.t <= lo, "t"] = lo + 1e-6
            for wt in ("crude", "ATT-IPW"):
                cph = CoxPHFitter()
                cols = ["t", "e", "vp", "entry", "hosp"] + (["w"] if wt != "crude" else [])
                cph.fit(d[cols], duration_col="t", event_col="e", entry_col="entry", cluster_col="hosp",
                        weights_col="w" if wt != "crude" else None, robust=True)
                s_ = cph.summary.loc["vp"]
                cox_rows.append(dict(population=popn, window=f"{lo}-{hi} h", weighting=wt, n_at_risk=len(d),
                                     events_VP=int(d.loc[d.vp == 1, "e"].sum()), events_NE=int(d.loc[d.vp == 0, "e"].sum()),
                                     csHR_95CI=fmt(s_["exp(coef)"], s_["exp(coef) lower 95%"], s_["exp(coef) upper 95%"]),
                                     p=s_["p"], note="cause-specific; censored at death, discharge, vasopressor restart; hospital-clustered robust SE"))
    return pd.DataFrame(cox_rows)


def ipew_fit(RS, y):
    """Odds ratio in the risk-set primary population by inverse probability of evaluability weighting."""
    d = RS.copy()
    X = d[COV + ["vp"]].astype(float)
    for c in MISS:
        X[c + "_miss"] = X[c].isna().astype(int)
        X[c] = X[c].fillna(X[c].median())
    X = X[[c for c in X.columns if X[c].nunique() > 1]]
    pe = sm.Logit(d.evaluable.astype(float), sm.add_constant(X)).fit(disp=0, maxiter=200).predict(sm.add_constant(X))
    d["w_eval"] = 1 / pe
    E_ = d[d.evaluable == 1].copy()
    E_["w_eval"] = np.minimum(E_.w_eval, np.quantile(E_.w_eval, 0.99))
    cc = [c for c in COV if c not in MISS]
    Xe = E_[["vp"] + cc + MISS].astype(float)
    for c in MISS:
        Xe[c + "_miss"] = Xe[c].isna().astype(int)
        Xe[c] = Xe[c].fillna(Xe[c].median())
    Xe = Xe[[c for c in Xe.columns if Xe[c].nunique() > 1]]
    r = sm.GLM(E_[y].astype(float), sm.add_constant(Xe), family=sm.families.Binomial(), var_weights=E_.w_eval).fit(
        cov_type="cluster", cov_kwds={"groups": E_.hospitalid})
    ci = r.conf_int().loc["vp"]
    return dict(analysis="IPEW (inverse prob. of evaluability; model in risk-set primary)", outcome=y, n_riskset=len(d),
                n_evaluable=len(E_), aOR_95CI=fmt(np.exp(r.params["vp"]), np.exp(ci[0]), np.exp(ci[1])),
                note="covariates with missing indicators; weights truncated at 99th pct; hospital-clustered SE")


def internal_vp(BE, forest):
    """Comparison within the VP group: NE stopped vs running, and no vasopressor vs vasopressor at T0."""
    VB = BE[BE.vp == 1].copy()
    VB["ne_off_t0"] = 1 - VB.ne_t0
    VB["press_free_t0"] = 1 - VB.press_t0
    cov_int = [c for c in COV if c not in ("ne_dose_unknown",)]
    rows = []
    for expo, lab_ in (("ne_off_t0", "NE stopped (not running) vs running at T0"), ("press_free_t0", "No vasopressor vs vasopressor continuing at T0")):
        for y in OUTCOMES:
            r_ = fit_or(VB, y, expo, cov_int, label=lab_)
            r_["rate_exposed"] = rate(VB.loc[VB[expo] == 1, y])
            r_["rate_reference"] = rate(VB.loc[VB[expo] == 0, y])
            r_["crude_OR_95CI"] = fit_or(VB, y, expo, [], mice=False)["aOR_95CI"]
            rows.append(r_)
            forest.append(dict(r_, outcome_label=y, model="internal VP, adjusted"))
    return pd.DataFrame(rows)


def sodium_clinical(PR, RS, forest):
    """Hypernatraemia within 72 h (adjusted) and descriptive clinical consequences."""
    rows = []
    for popn, P_ in (("PRIMARY evaluable", PR), ("Risk-set primary", RS)):
        S_ = P_[(P_.na_pre_max < 150) & (P_.na_post72_n >= 1)].copy()
        S_["na150_72h"] = (S_.na_post72_max >= 150).astype(int)
        S_["na_rise8_72h"] = ((S_.na_post72_max - S_.na_pre) >= 8).astype(int)
        for y, lab_ in (("na150_72h", "Na >= 150 within 72 h"), ("na_rise8_72h", "Na rise >= 8 within 72 h")):
            r_ = fit_or(S_, y, "vp", COV, label=popn)
            r_.update(outcome_label=lab_, rate_VP=rate(S_.loc[S_.vp == 1, y]), rate_NE=rate(S_.loc[S_.vp == 0, y]),
                      denominator="all Na 24 h before T0 < 150 and >= 1 Na in 0-72 h",
                      crude_OR_95CI=fit_or(S_, y, "vp", [], mice=False)["aOR_95CI"])
            rows.append(r_)
            forest.append(dict(r_, model=ADJ))
        P_ = P_.copy()
        P_["restart24"] = (P_.restart_h <= 24).astype(int)
        for y, lab_ in (("restart24", "Vasopressor restart within 24 h"), ("ddavp_antidiuretic_48h", "Antidiuretic desmopressin within 48 h (treatment table)"),
                        ("hosp_death", "Hospital death (descriptive)")):
            rows.append(dict(analysis=popn, outcome=y, outcome_label=lab_, n=len(P_), rate_VP=rate(P_.loc[P_.vp == 1, y]),
                             rate_NE=rate(P_.loc[P_.vp == 0, y]), aOR_95CI="descriptive"))
    return pd.DataFrame(rows)


def sensitivity(C, PR):
    """Sensitivity analyses of the 24-h odds ratio (VP vs NE-only). Returns the result rows."""
    EV = C[(C.riskset == 1) & (C.excl_pit == 0)]
    sens = []

    def add(lab_, df, y="polyuria_24h", covs=COV, mice=True):
        r_ = fit_or(df, y, "vp", covs, mice=mice, label=lab_)
        r_["rate_VP"] = rate(df.loc[df.vp == 1, y])
        r_["rate_NE"] = rate(df.loc[df.vp == 0, y])
        sens.append(r_)

    add("Reference: PRIMARY", PR)
    add("Crude", PR, covs=[], mice=False)
    add("Complete case (no MICE)", PR, mice=False)
    add("(a) include stays polyuric before T0", EV[(EV.urine_evaluable == 1) & (EV.press_t0 == 0)])
    add("(b) evaluability >= 6 bins", EV[(EV.obs_h_24 >= 6) & (EV.poly_pre12 == 0) & (EV.press_t0 == 0)])
    add("(b) evaluability >= 18 bins", EV[(EV.obs_h_24 >= 18) & (EV.poly_pre12 == 0) & (EV.press_t0 == 0)])
    add("(d) outcome: 24-h volume > 3 L or > 50 mL/kg", PR, y="uo24_def")
    add("Outcome: marked polyuria > 5 mL/kg/h x 2 h", PR, y="marked_24h")
    add("Outcome: polyuria within 72 h", PR, y="polyuria_72h")
    add("(g) no vasopressor restart within 24 h", PR[~(PR.restart_h <= 24)])
    add("(i) exclude any neurological diagnosis", PR[PR.neuro_any == 0])
    add("(k) one stay per patient (earliest T0 per uniquepid)", PR.sort_values(["uniquepid", "unitvisitnumber", "t0"]).drop_duplicates("uniquepid"))
    add("Vasopressor-at-T0 by record rule only ([T0-60, T0] min)", C[(C.base_eval == 1) & (C.press_t0_rec == 0)])
    add("+ lactate and pre-T0 intake (missing indicators)", PR.assign(lac_m=PR.lactate_pre24.isna().astype(int), lac=PR.lactate_pre24.fillna(PR.lactate_pre24.median()),
                                                                     fl_m=PR.fluid_pre24_l.isna().astype(int), fl=PR.fluid_pre24_l.fillna(0)),
        covs=COV + ["lac_m", "lac", "fl_m", "fl"])
    add("No inotrope at T0", PR[PR.inotrope_t0 == 0])
    add("Index (first) episode >= 6 h", PR[PR.idx_dur_h >= 6])
    return sens


def subgroups(PR):
    """24-h odds ratio within subgroups and the p value of the interaction."""
    sub = []
    for var, lab_ in (("sepsis", "Sepsis"), ("card_surg", "Cardiac surgery"), ("cvicu", "Cardiac ICU"), ("press6", "Vasopressor >= 6 h before T0")):
        P_ = PR.copy()
        P_["press6"] = (P_.press_h_pre >= 6).astype(int)
        for lv in (1, 0):
            x = P_[P_[var] == lv]
            try:
                r_ = fit_or(x, "polyuria_24h", "vp", [c for c in COV if c != var], label=f"{lab_} = {lv}")
            except Exception as e:
                r_ = dict(analysis=f"{lab_} = {lv}", aOR_95CI=f"not estimable ({type(e).__name__})", n=len(x))
            r_["rate_VP"] = rate(x.loc[x.vp == 1, "polyuria_24h"])
            r_["rate_NE"] = rate(x.loc[x.vp == 0, "polyuria_24h"])
            sub.append(r_)
        # interaction p (single data set: median imputation with missing indicators)
        Q = P_.copy()
        Q["ix"] = Q.vp * Q[var]
        xs = ["vp", var, "ix"] + [c for c in COV if c != var]
        for c in MISS:
            Q[c + "_m"] = Q[c].isna().astype(int)
            Q[c] = Q[c].fillna(Q[c].median())
            xs.append(c + "_m")
        p_, v_ = logit_cluster(Q, "polyuria_24h", xs)
        sub.append(dict(analysis=f"{lab_}: interaction p", p=2 * norm.sf(abs(p_["ix"] / np.sqrt(v_.loc["ix", "ix"]))), aOR_95CI="—"))
    return pd.DataFrame(sub)


def robustness(PR, ev, osm):
    """Additional robustness analyses (complete case unless stated). ev and osm are the
    loop-diuretic and osmotic-agent events with h = hours from T0."""
    robust = []

    def add_robust(lab_, df, y, covs, mice=False):
        try:
            r_ = fit_or(df, y, "vp", covs, mice=mice, label=lab_)
        except Exception as e:
            r_ = dict(analysis=lab_, outcome=y, aOR_95CI=f"not estimable ({type(e).__name__})")
        robust.append(r_)

    cov_noap = [c for c in COV if c != "apache10"]
    hd = pd.get_dummies(PR.hospitalid.astype(int), prefix="hosp", drop_first=True).astype(int)
    PRH = pd.concat([PR, hd], axis=1)
    # (h) polyuria not preceded within 6 h by loop diuretic / mannitol / hypertonic saline (outcome-side definition)
    co = pd.concat([ev[["patientunitstayid", "h"]], osm[["patientunitstayid", "h"]]])
    co = co.merge(PR[["patientunitstayid", "uo_onset_h"]], on="patientunitstayid")
    bad = set(co.loc[(co.h <= co.uo_onset_h) & (co.h >= co.uo_onset_h - 6), "patientunitstayid"])
    PRh = PR.copy()
    for y in OUTCOMES:
        PRh[y] = np.where(PRh.patientunitstayid.isin(bad), 0, PRh[y])
    for y in OUTCOMES:
        add_robust("CC: full covariates (reference)", PR, y, COV)
        add_robust("CC: without APACHE IV", PR, y, cov_noap)
        add_robust("CC: APS (acute physiology score) instead of APACHE IV", PR, y, cov_noap + ["aps10"])
        add_robust("CC: + hospital fixed effects", PRH, y, COV + [c for c in hd.columns if PRH[c].sum() > 0])
        add_robust("CC: + pre-T0 hourly charting pairs (-24..-1 h)", PR, y, COV + ["pre_pairs"])
        add_robust("CC: hourly-charted stays only (>= 12 consecutive pairs 0-24 h)", PR[PR.post_pairs >= 12], y, COV)
        add_robust("CC: T0 = charted stop (rate-0 record) only", PR[PR.idx_stop_charted], y, COV)
        add_robust("CC: no vasopressor restart within 12 h", PR[~(PR.restart_h <= 12)], y, COV)
        add_robust("CC: (h) no diuretic/mannitol/hypertonic saline within 6 h before onset", PRh, y, COV)
    S_ = PR[(PR.na_pre_max < 150) & (PR.na_post72_n >= 1)].copy()
    S_["na150_72h"] = (S_.na_post72_max >= 150).astype(int)
    S_["log_na_n"] = np.log(S_.na_post72_n)
    add_robust("Na >= 150 within 72 h, CC", S_, "na150_72h", COV)
    add_robust("Na >= 150 within 72 h, CC + log(number of Na tests 0-72 h)", S_, "na150_72h", COV + ["log_na_n"])
    hg = PR.groupby("hospitalid").vp.agg(["sum", "count"])
    robust.append(dict(analysis="Hospitals in PRIMARY", n=len(hg), aOR_95CI=f"{int((hg['sum'] == 0).sum())} hospitals with no VP stay ({int(hg.loc[hg['sum'] == 0, 'count'].sum())} NE stays); {int((hg['sum'] == hg['count']).sum())} with no NE stay"))
    return pd.DataFrame(robust)


def cointerventions(PR):
    """Post-T0 co-interventions by group and outcome (descriptive)."""
    rows = []
    PR2 = PR.assign(glu250=(PR.glucose_max24 >= 250).astype(int))
    for g, x in PR2.groupby(["grp_label", "polyuria_24h"]):
        rows.append(dict(group=g[0], polyuria_24h=g[1], n=len(x), loop_diuretic_0_24h=rate(x.diur_post24),
                         mannitol_or_hypertonic_0_24h=rate(x.osmo_post24), glucose_ge250_0_24h=rate(x.glu250),
                         glucose_measured=int(x.glucose_max24.notna().sum()),
                         note="RRT stays excluded by design (any RRT during the stay)"))
    return pd.DataFrame(rows)


def urine_charting(C):
    """Urine charting by group in the risk set."""
    rows = []
    for g, x in C[C.riskset == 1].groupby("grp_label"):
        rows.append(dict(group=g, n_riskset=len(x), any_urine_record=rate(x.has_urine_any), evaluable=rate(x.evaluable),
                         median_charting_interval_min=x.urine_gap_med_min.median(),
                         IQR=f"{x.urine_gap_med_min.quantile(.25):.0f}–{x.urine_gap_med_min.quantile(.75):.0f}"))
    return pd.DataFrame(rows)


def write_workbook(path, TAB):
    """Write the result sheets; OR, lo and hi are rounded to two decimals and p values formatted."""
    with pd.ExcelWriter(path) as wr:
        for k in SHEET_ORDER:
            if k in TAB:
                t_ = TAB[k].copy()
                for c in ("OR", "lo", "hi"):
                    if c in t_.columns:
                        t_[c] = pd.to_numeric(t_[c], errors="coerce").round(2)
                if "p" in t_.columns:
                    t_["p"] = pd.to_numeric(t_["p"], errors="coerce").map(lambda v: "" if pd.isna(v) else ("<0.001" if v < 0.001 else f"{v:.3f}"))
                t_.to_excel(wr, sheet_name=k[:31], index=False)


def main():
    warnings.filterwarnings("ignore")
    t_start = time.time()
    log = RunLog()
    say = log.say
    out = out_dir("01")

    # ---- cohort ---------------------------------------------------------------------------
    src = load_source(say)
    C, aux = build_primary_cohort(src, say)
    C.to_parquet(cohort_path(COHORT_PRIMARY), index=False)
    say("cohort built in %.0f s" % (time.time() - t_start))

    # ---- counts, flow and descriptive tables -----------------------------------------------
    TAB = {}
    TAB["02_VP_counts"] = vp_counts(C)
    say(TAB["02_VP_counts"].to_string(index=False))
    TAB["01_flow"] = flow_table(C, aux.flow)
    say(TAB["01_flow"].to_string(index=False))
    PR = C[C.primary == 1].copy()
    RS = C[C.rs_primary == 1].copy()
    BE = C[C.base_eval == 1].copy()
    TAB["03_table1_primary"] = table1(PR)
    TAB["03b_table1_riskset_primary"] = table1(RS)
    TAB["17_NE_dose_units"] = aux.ne_units
    TAB["04_crude_rates"] = crude_rates(C, PR, BE)
    say(TAB["04_crude_rates"][["population", "group", "n_stays", "poly24", "poly_0_8", "poly_8_24"]].to_string(index=False))
    TAB["04b_onset_windows"] = onset_windows(PR, BE)
    TAB["04b_onset_windows"].to_csv(os.path.join(out, "fig_data_onset_windows.csv"), index=False)

    # ---- adjusted logistic regression -------------------------------------------------------
    FOREST = []
    TAB["05_primary_OR"], TAB["18_coef_primary"] = primary_models(PR, BE, FOREST, say)

    # ---- ATT weighting ----------------------------------------------------------------------
    w_pr, Xps = ps_weights(PR, COV)
    TAB["06_IPW_ATT"], TAB["07_SMD_primary"] = ipw_att(PR, w_pr, Xps)
    say(TAB["06_IPW_ATT"].to_string(index=False))
    say("max |SMD| weighted: %.3f" % TAB["07_SMD_primary"].SMD_ATT_weighted.abs().max())

    # ---- cumulative incidence, cause-specific Cox, risk-set estimand -------------------------
    w_rs, _ = ps_weights(RS, COV)
    CIF, TAB["10_CIF_8h_24h"] = cumulative_incidence(PR, RS, w_pr, w_rs)
    CIF.to_csv(os.path.join(out, "fig_data_cif_hourly.csv"), index=False)
    say(TAB["10_CIF_8h_24h"].to_string(index=False))
    TAB["02_VP_counts"] = pd.concat([TAB["02_VP_counts"]] + vp_riskset_cif(C))
    say(TAB["02_VP_counts"].tail(2).to_string(index=False))
    TAB["09_csCox_time_split"] = cause_specific_cox(PR, RS, w_pr, w_rs)
    say(TAB["09_csCox_time_split"].to_string(index=False))
    TAB["11_IPEW_riskset"] = pd.DataFrame([ipew_fit(RS, y) for y in OUTCOMES])
    say(TAB["11_IPEW_riskset"].to_string(index=False))

    # ---- comparison within the VP group, sodium, sensitivity, subgroups, robustness ----------
    TAB["12_internal_VP"] = internal_vp(BE, FOREST)
    say(TAB["12_internal_VP"][["analysis", "outcome", "n", "rate_exposed", "rate_reference", "crude_OR_95CI", "aOR_95CI"]].to_string(index=False))
    TAB["13_sodium_clinical"] = sodium_clinical(PR, RS, FOREST)
    say(TAB["13_sodium_clinical"][["analysis", "outcome", "n", "rate_VP", "rate_NE", "aOR_95CI"]].to_string(index=False))
    sens = sensitivity(C, PR)
    TAB["14_sensitivity"] = pd.DataFrame(sens)
    say(TAB["14_sensitivity"][["analysis", "outcome", "n", "events", "rate_VP", "rate_NE", "aOR_95CI"]].to_string(index=False))
    for r_ in sens:
        FOREST.append(dict(r_, outcome_label=r_["outcome"], model="sensitivity"))
    TAB["15_subgroups"] = subgroups(PR)
    TAB["19_robustness"] = robustness(PR, aux.ev, aux.osm)
    say(TAB["19_robustness"][["analysis", "outcome", "n", "events", "aOR_95CI"]].to_string(index=False))
    TAB["16_cointerventions"] = cointerventions(PR)
    TAB["16b_urine_charting"] = urine_charting(C)

    # ---- forest data and workbook -------------------------------------------------------------
    FO = pd.DataFrame(FOREST)
    keep = [c for c in ["model", "analysis", "outcome", "outcome_label", "exposure", "n", "events", "n_exposed", "n_hospitals",
                        "method", "OR", "lo", "hi", "p", "aOR_95CI", "rate_VP", "rate_NE"] if c in FO.columns]
    FO[keep].to_csv(os.path.join(out, "fig_data_forest_or.csv"), index=False)
    TAB["00_README"] = pd.DataFrame({"item": [
        "Post hoc comparator analysis, eICU-CRD v2.0. Unit = ICU stay; patients = uniquepid.",
        "PRIMARY population = evaluable, new-onset (no polyuria -12..-1 h), pituitary/DI/brain-death excluded, no vasopressor running at T0.",
        "Adjusted models: logistic, hospital-clustered SE, MICE m=%d (Na, MAP, APACHE IV) with Rubin's rules; KDIGO missing = 'unknown' category." % MICE_M,
        "Covariates: " + ", ".join(LAB[c] for c in COV),
        "APACHE IV is a first-ICU-day score (may include data after T0 when T0 falls on ICU day 1).",
        "IPW: ATT (VP treated), PS with missing indicators, NE weights truncated at the 99th percentile, hospital-cluster bootstrap B=%d." % NBOOT,
        "CIF: Aalen-Johansen, competing = ICU death, ICU discharge alive, any vasopressor restart; ties go to the competing event.",
        "poly_8_24 = onset in (8, 24] h counted over the whole population; a conditional version is in 05_primary_OR.",
        "Urine: label list includes 'Urinary Catheter Output: Indwelling/Continuous Ure'; void-count labels excluded; mirrored same-time entries counted once; weight 30-300 kg with infusion-weight fallback. Sheet 19 = robustness analyses."]})
    write_workbook(os.path.join(out, "eICU_primary_results.xlsx"), TAB)
    log.write(os.path.join(out, RUN_LOG))
    say("done in %.1f min ->" % ((time.time() - t_start) / 60), out)


if __name__ == "__main__":
    main()
