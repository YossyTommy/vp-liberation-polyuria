#!/usr/bin/env python3
"""
01_pooled_estimates.py - fixed-effect pooled estimates across the two databases.

The databases are not merged at the stay level. Each pooled value is the inverse-variance
weighted mean of the two database-specific adjusted log odds ratios, with
SE = (ln upper - ln lower) / 3.92 taken from the 95% CI of each estimate. All pooled values
are computed from the database-specific estimates at full precision (not from the rounded
values shown in the tables). I2 (heterogeneity, %) is derived from Cochran's Q.

  pooled_estimates.csv   primary analysis: polyuria within 24 h, 0-8 h and >8-24 h, Na >= 150 mmol/L
                         within 72 h, and the ratio of odds ratios (0-8 h vs >8-24 h)
  pooled_secondary.csv   fluid-balance sensitivity analyses and the usage-pattern / stopping-order groups
Columns: mimic, eicu and pooled as "estimate (95% CI)" to two decimals; est, lo, hi = pooled estimate at full precision.

Inputs
  OUT_DIR/mimic_02/fig_mimic_forest_estimates.csv   primary adjusted OR (MICE): 24 h, 0-8 h, >8-24 h
  OUT_DIR/eicu_01/fig_data_forest_or.csv            same outcomes, eICU-CRD
  OUT_DIR/mimic_08/sodium_harmonized.csv            Na >= 150 mmol/L within 72 h (harmonized denominator)
  OUT_DIR/eicu_01/eICU_primary_results.xlsx        sheet 13_sodium_clinical
  OUT_DIR/mimic_04/window_test_mimic.csv, OUT_DIR/eicu_03/window_test_eicu.csv   ratio of odds ratios
  OUT_DIR/mimic_06/results_mimic_fluid_balance.xlsx, OUT_DIR/eicu_05/eICU_fluid_balance_results.xlsx
  OUT_DIR/mimic_05/results_mimic_usage_groups.xlsx, OUT_DIR/eicu_04/eICU_usage_groups_results.xlsx
Outputs  OUT_DIR/pooled/pooled_estimates.csv, OUT_DIR/pooled/pooled_secondary.csv

Run:  python pooled/01_pooled_estimates.py   (after mimic/02, 04, 05, 06, 08 and eicu/01, 03, 04, 05)
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

M_ADJ = ("adjusted_OR", "adjusted_est", "adjusted_lo", "adjusted_hi")   # text column, then estimate and 95% CI
M_EST = ("aOR", "est", "lo", "hi")
E_OR = ("OR_95CI", "OR", "lo", "hi")


def pool(ests):
    """Fixed-effect pooled (OR, lower, upper, p, I2 %) of (OR, lower, upper) tuples."""
    L = np.log([e[0] for e in ests])
    S = np.array([(np.log(e[2]) - np.log(e[1])) / 3.92 for e in ests])
    w = 1 / S ** 2
    lp = (w * L).sum() / w.sum()
    sp = w.sum() ** -.5
    q = float((w * (L - lp) ** 2).sum())
    i2 = 100 * max(0.0, (q - (len(ests) - 1)) / q) if q > 0 else 0.0
    return np.exp(lp), np.exp(lp - 1.96 * sp), np.exp(lp + 1.96 * sp), 2 * norm.sf(abs(lp / sp)), i2


def fmt(e):
    return "%.2f (%.2f–%.2f)" % tuple(e[:3])


def pick(d, cols, starts=None, **equal):
    """Estimate and 95% CI at full precision from the single row of d matching the filters.
    cols = (text column, estimate, lower, upper); the numbers must round to the text."""
    for k, v in equal.items():
        d = d[d[k] == v]
    if starts:
        d = d[d[starts[0]].astype(str).str.startswith(starts[1])]
    if len(d) != 1:
        raise ValueError(f"{len(d)} rows match {equal} {starts}")
    r = d.iloc[0]
    e = tuple(float(r[c]) for c in cols[1:])
    shown = tuple(float(x) for x in re.findall(r"\d+\.\d+", str(r[cols[0]]))[:3])
    if tuple(round(x, 2) for x in e) != shown:
        raise ValueError(f"{cols[1:]} = {e} do not match {r[cols[0]]!r}")
    return e


def secondary(o):
    """Pooled estimates of the fluid-balance sensitivity analyses and the stopping-order groups."""
    M = pd.read_excel(o("mimic_06", "results_mimic_fluid_balance.xlsx"), sheet_name=None)
    E = pd.read_excel(o("eicu_05", "eICU_fluid_balance_results.xlsx"), sheet_name=None)
    na = pd.read_csv(o("mimic_08", "sodium_harmonized.csv"))
    na = na[na.denominator == "harmonized"]
    two = "2-group primary VP vs NE"
    OUT = [("poly24", "polyuria_24h", "Polyuria within 24 h"), ("poly8", "poly8", "Polyuria 0-8 h"),
           ("poly_8_24", "poly_8_24", "Polyuria >8-24 h")]
    rows = []

    def add(table, analysis, outcome, m, e):
        p = pool([m, e])
        rows.append(dict(table=table, analysis=analysis, outcome=outcome, mimic=fmt(m), eicu=fmt(e),
                         pooled=fmt(p), i2_percent=round(p[4]), est=p[0], lo=p[1], hi=p[2]))

    # fluid balance: models with cumulative fluid overload (FO%) added
    for mo, eo, lab in OUT:
        add("fluid balance", "+ cumulative FO%", lab,
            pick(M["04_b_FO_adjustment"], M_ADJ, analysis="(b) " + two, outcome=mo, model="B: COV_MI + FO% (linear)"),
            pick(E["b_FO_before_after"], E_OR, comparison=two, outcome=eo, model="COV+FO%", population="all analysis stays"))
    add("fluid balance", "+ cumulative FO%", "Na >= 150 mmol/L within 72 h",
        pick(na, M_EST, analysis="2grp: + cumulative FO% (B: COV_MI + FO%)"),
        pick(E["b_FO_before_after"], E_OR, comparison=two, outcome="na150_72h", model="COV+FO%", population="all analysis stays"))
    for mo, eo, lab in OUT[:2]:
        add("fluid balance", "overlap weighting, propensity score including FO%", lab,
            pick(M["08_f_ATO"], ("OR_ATO", "est", "lo", "hi"), outcome=mo),
            pick(E["f_ATO"], E_OR, population="2-group primary", ps_model="PS: COV+FO%", outcome=eo))
    add("fluid balance", "overlap weighting, propensity score including FO%", "Na >= 150 mmol/L within 72 h",
        pick(na, M_EST, analysis="2grp: ATO overlap weighting, PS incl FO%"),
        pick(E["f_ATO"], E_OR, population="2-group primary, Na evaluable", ps_model="PS: COV+FO%", outcome="na150_72h"))
    for mo, eo, lab in OUT[:2]:
        add("fluid balance", "vasopressors >= 24 h and FO% >= comparator median (+ FO%)", lab,
            pick(M["07_e_severe_fluid"], M_ADJ, starts=("analysis", "(e) " + two), outcome=mo, model="B: COV_MI + FO%"),
            pick(E["e_long_course_high_FO"], E_OR, comparison=two, outcome=eo, model="COV+FO%"))
        add("fluid balance", "no loop diuretic 0-24 h after T0 (+ FO%)", lab,
            pick(M["09_g_no_diuretic_h2h"], M_ADJ, starts=("analysis", "(g1) no loop diuretic 0–24 h after T0: 2-group"),
                 outcome=mo, model="B: COV_MI + FO%"),
            pick(E["g1_no_diuretic"], E_OR, comparison=two, outcome=eo, model="COV+FO%"))
    add("fluid balance", "within comparator stays: per 5-point FO% increase", "Polyuria 0-8 h",
        pick(M["06_d_NEonly_FO"], ("OR", "est", "lo", "hi"), outcome="poly8", model="adjusted COV_MI"),
        pick(E["d_NEonly_FO"], ("aOR_95CI", "OR", "lo", "hi"), population="recorded", outcome="poly8", method="MICE m=10"))

    # usage pattern / stopping order (time zero T0b), each group versus the comparator
    m5 = pd.read_excel(o("mimic_05", "results_mimic_usage_groups.xlsx"), sheet_name="06_outcomes")
    m5 = m5[m5.analysis.astype(str).str.startswith("5 groups, main risk set")]
    e5 = pd.read_excel(o("eicu_04", "eICU_usage_groups_results.xlsx"), sheet_name="06_outcomes_OR")
    e5 = e5[(e5.analysis == "5-group adjusted") & e5.risk_set.astype(str).str.startswith("analysis population (all stays")]
    e5na = pd.read_excel(o("eicu_04", "eICU_usage_groups_results.xlsx"), sheet_name="07b_sodium_OR")
    e5na = e5na[(e5na.analysis == "5-group adjusted MICE") & (e5na.outcome == "na150_72h")]
    for g in ("VP-only", "NE-first", "Simultaneous", "VP-first"):
        for mo, eo, lab in (OUT[1], OUT[2], OUT[0]):
            add("stopping order", g, lab, pick(m5, M_ADJ, outcome=mo, group=g), pick(e5, E_OR, outcome=eo, group=g))
        add("stopping order", g, "Na >= 150 mmol/L within 72 h",
            pick(na, M_EST, analysis=f"5grp: {g} vs NE-only"), pick(e5na, E_OR, group=g))
    return pd.DataFrame(rows)


def main():
    o = lambda *p: os.path.join(config.OUT_DIR, *p)
    fm = pd.read_csv(o("mimic_02", "fig_mimic_forest_estimates.csv"))
    fm = fm[(fm.section == "primary") & (fm.analysis == "adjusted logistic (MICE)") & (fm.population == "primary")].set_index("outcome")
    fe = pd.read_csv(o("eicu_01", "fig_data_forest_or.csv"))
    fe = fe[(fe.model == "adjusted (MICE, hospital-clustered)") & (fe.analysis == "PRIMARY (no vasopressor at T0)")].set_index("outcome")
    nm = pd.read_csv(o("mimic_08", "sodium_harmonized.csv"))
    nm = nm[(nm.analysis == "2grp: primary") & (nm.denominator == "harmonized")].iloc[0]
    ne = pd.read_excel(o("eicu_01", "eICU_primary_results.xlsx"), sheet_name="13_sodium_clinical")
    ne = ne[(ne.analysis == "PRIMARY evaluable") & (ne.outcome == "na150_72h")].iloc[0]
    wm = pd.read_csv(o("mimic_04", "window_test_mimic.csv")).iloc[0]
    we = pd.read_csv(o("eicu_03", "window_test_eicu.csv")).iloc[0]
    items = [("Polyuria within 24 h", tuple(fm.loc["poly24", ["est", "lo", "hi"]]), tuple(fe.loc["polyuria_24h", ["OR", "lo", "hi"]])),
             ("Polyuria 0-8 h", tuple(fm.loc["poly8", ["est", "lo", "hi"]]), tuple(fe.loc["poly8", ["OR", "lo", "hi"]])),
             ("Polyuria >8-24 h", tuple(fm.loc["poly_8_24", ["est", "lo", "hi"]]), tuple(fe.loc["poly_8_24", ["OR", "lo", "hi"]])),
             ("Na >= 150 mmol/L within 72 h", (nm.est, nm.lo, nm.hi), (ne.OR, ne.lo, ne.hi)),
             ("Ratio of odds ratios, 0-8 h vs >8-24 h", (wm.ratio_est, wm.ratio_lo, wm.ratio_hi), (we.ratio_est, we.ratio_lo, we.ratio_hi))]
    rows = []
    for lab, m, e in items:
        p = pool([m, e])
        rows.append(dict(outcome=lab, mimic=fmt(m), eicu=fmt(e), pooled=fmt(p), p=round(p[3], 4), i2_percent=round(p[4]),
                         est=p[0], lo=p[1], hi=p[2]))
    R = pd.DataFrame(rows)
    print(R.to_string(index=False))
    R.to_csv(os.path.join(config.out_dir("pooled"), "pooled_estimates.csv"), index=False)
    S = secondary(o)
    print(S.to_string(index=False))
    S.to_csv(os.path.join(config.out_dir("pooled"), "pooled_secondary.csv"), index=False)


if __name__ == "__main__":
    main()
