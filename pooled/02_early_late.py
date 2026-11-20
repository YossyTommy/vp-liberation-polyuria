#!/usr/bin/env python3
"""
02_early_late.py - pooled estimates of the post hoc exploratory analyses added after v15 and the
multiplicity table of the v15 secondary end points (both reported in the manuscript; the
exploratory analyses as post hoc, the multiplicity table as Supplementary Table S18).

  B  multinomial logistic regression (no polyuria / polyuria without sodium rise / polyuria with
     sodium rise): adjusted OR of the VP group for categories 1 and 2, pooled across databases
  A  (supplementary) group x timing interaction of the linear model of the sodium rise, pooled
  C  Holm adjustment of the pooled p values of the four v15 secondary end points
     (polyuria 0-8 h, polyuria >8-24 h, Na >= 150 mmol/L within 72 h, ratio of odds ratios)

Pooling follows pooled/01_pooled_estimates.py: fixed-effect inverse-variance weighting of the
database-specific estimates at full precision, SE = (upper - lower) / 3.92 on the log scale
(odds ratios) or on the original scale (linear coefficients); I2 from Cochran's Q. For C the
pooled p values are recomputed at full precision from est / lo / hi of pooled_estimates.csv
(SE = (ln hi - ln lo) / 3.92, two-sided normal test), which is the calculation of script 01.

Inputs   OUT_DIR/mimic_10/B_multinomial.csv, A_regression_na_rise.csv
         OUT_DIR/eicu_08/B_multinomial.csv, A_regression_na_rise.csv
         OUT_DIR/pooled/pooled_estimates.csv
Outputs  OUT_DIR/pooled_02/B_multinomial_pooled.csv, A_interaction_pooled.csv, C_holm_secondary.csv
Run:     python pooled/02_early_late.py   (after mimic/10, eicu/08 and pooled/01)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def pool(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float)
    w = 1 / se ** 2
    p_ = (w * est).sum() / w.sum()
    sp = w.sum() ** -.5
    q = float((w * (est - p_) ** 2).sum())
    i2 = 100 * max(0.0, (q - (len(est) - 1)) / q) if q > 0 else 0.0
    return p_, sp, 2 * norm.sf(abs(p_ / sp)), i2


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    o = lambda *p: os.path.join(config.OUT_DIR, *p)
    out = config.out_dir("pooled_02")

    # ---- B ------------------------------------------------------------------------------
    M = pd.read_csv(o("mimic_10", "B_multinomial.csv"))
    E = pd.read_csv(o("eicu_08", "B_multinomial.csv"))
    rows = []
    for (a, c), m in M.groupby(["analysis", "category"]):
        e = E[(E.analysis == a) & (E.category == c)].iloc[0]
        m = m.iloc[0]
        L = [np.log(m.OR), np.log(e.OR)]
        S = [(np.log(m.hi) - np.log(m.lo)) / 3.92, (np.log(e.hi) - np.log(e.lo)) / 3.92]
        lp, sp, p, i2 = pool(L, S)
        rows.append(dict(definition=a, category=c,
                         label={1: "polyuria without Na rise", 2: "polyuria with Na rise"}[c],
                         mimic="%.2f (%.2f–%.2f)" % (m.OR, m.lo, m.hi), mimic_p=m.p,
                         eicu="%.2f (%.2f–%.2f)" % (e.OR, e.lo, e.hi), eicu_p=e.p,
                         pooled="%.2f (%.2f–%.2f)" % (np.exp(lp), np.exp(lp - 1.96 * sp), np.exp(lp + 1.96 * sp)),
                         pooled_p=p, i2_percent=round(i2), est=np.exp(lp), lo=np.exp(lp - 1.96 * sp), hi=np.exp(lp + 1.96 * sp),
                         log_or=lp, se=sp))
    B = pd.DataFrame(rows)
    # ratio of the category-2 and category-1 odds ratios (difference of pooled log ORs; the two
    # pooled estimates share the reference category, so their covariance is ignored here and the
    # interval is approximate)
    for a, g in B.groupby("definition"):
        b1, b2 = g[g.category == 1].iloc[0], g[g.category == 2].iloc[0]
        d = b2.log_or - b1.log_or
        rows.append(dict(definition=a, category="2 vs 1", label="ratio of ORs (with / without Na rise), approximate",
                         pooled="%.2f" % np.exp(d), est=np.exp(d)))
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(out, "B_multinomial_pooled.csv"), index=False)
    print(B[["definition", "category", "mimic", "eicu", "pooled", "pooled_p", "i2_percent"]].to_string(index=False))

    # ---- A: interaction -------------------------------------------------------------------
    MA = pd.read_csv(o("mimic_10", "A_regression_na_rise.csv"))
    EA = pd.read_csv(o("eicu_08", "A_regression_na_rise.csv"))
    rows = []
    for (a, t), m in MA.groupby(["analysis", "term"], sort=False):
        e = EA[(EA.analysis == a) & (EA.term == t)].iloc[0]
        m = m.iloc[0]
        est, sp, p, i2 = pool([m.est, e.est], [(m.hi - m.lo) / 3.92, (e.hi - e.lo) / 3.92])
        rows.append(dict(analysis=a, term=t, mimic="%.2f (%.2f to %.2f)" % (m.est, m.lo, m.hi), mimic_p=m.p,
                         eicu="%.2f (%.2f to %.2f)" % (e.est, e.lo, e.hi), eicu_p=e.p,
                         pooled="%.2f (%.2f to %.2f)" % (est, est - 1.96 * sp, est + 1.96 * sp), pooled_p=p,
                         i2_percent=round(i2), est=est, lo=est - 1.96 * sp, hi=est + 1.96 * sp))
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(out, "A_interaction_pooled.csv"), index=False)
    print(A[["analysis", "term", "mimic", "eicu", "pooled", "pooled_p"]].to_string(index=False))

    # ---- C: Holm ---------------------------------------------------------------------------
    PE = pd.read_csv(o("pooled", "pooled_estimates.csv")).set_index("outcome")
    sec = ["Polyuria 0-8 h", "Polyuria >8-24 h", "Na >= 150 mmol/L within 72 h", "Ratio of odds ratios, 0-8 h vs >8-24 h"]
    r = PE.loc[sec].copy()
    se = (np.log(r.hi) - np.log(r.lo)) / 3.92
    r["p_full"] = 2 * norm.sf(np.abs(np.log(r.est) / se))
    assert np.allclose(r.p_full.round(4), r.p), "recomputed p values differ from pooled_estimates.csv"
    r["p_holm"] = holm(r.p_full.values)
    r["rank"] = r.p_full.rank().astype(int)
    r["significant_unadjusted"] = r.p_full < 0.05
    r["significant_holm"] = r.p_holm < 0.05
    C = r.reset_index()[["outcome", "pooled", "p_full", "rank", "p_holm", "significant_unadjusted", "significant_holm", "mimic", "eicu"]]
    C.to_csv(os.path.join(out, "C_holm_secondary.csv"), index=False)
    print(C.to_string(index=False))


if __name__ == "__main__":
    main()
