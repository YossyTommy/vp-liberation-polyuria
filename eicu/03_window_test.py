#!/usr/bin/env python3
"""
03_window_test.py - eICU-CRD v2.0, comparison of the 0-8 h and >8-24 h windows.

Multinomial logistic regression in the primary population (complete cases) with a three-level
outcome: no polyuria within 24 h (reference), onset within 0-8 h, onset within >8-24 h.
Covariates are those of the primary model; standard errors are clustered on hospital. The
ratio of the two odds ratios (early / late) and its Wald test use the full covariance matrix
of the two vasopressin coefficients.

Input   WORK_DIR/eicu/cohort/cohort_primary.parquet
Output  OUT_DIR/eicu_03/window_test_eicu.csv

Run:  python eicu/03_window_test.py   (after 01_primary_analysis.py)
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # eicu_* modules; needed when Python runs in safe-path mode
from eicu_common import COHORT_PRIMARY, cohort_path, out_dir  # noqa: E402
from eicu_stats import COV  # noqa: E402

XCOLS = ["vp"] + COV


def eqtest(r):
    """Odds ratios of the two windows and their ratio from a fitted MNLogit (outcome levels 1 and 2)."""
    V = r.cov_params()
    b1, b2 = r.params.loc["vp", 0], r.params.loc["vp", 1]
    v1, v2, c = V.loc[("1", "vp"), ("1", "vp")], V.loc[("2", "vp"), ("2", "vp")], V.loc[("1", "vp"), ("2", "vp")]
    d = b1 - b2
    se = np.sqrt(v1 + v2 - 2 * c)
    return dict(OR_early=f"{np.exp(b1):.2f} ({np.exp(b1 - 1.96 * np.sqrt(v1)):.2f}–{np.exp(b1 + 1.96 * np.sqrt(v1)):.2f})",
                OR_late=f"{np.exp(b2):.2f} ({np.exp(b2 - 1.96 * np.sqrt(v2)):.2f}–{np.exp(b2 + 1.96 * np.sqrt(v2)):.2f})",
                ratio=f"{np.exp(d):.2f} ({np.exp(d - 1.96 * se):.2f}–{np.exp(d + 1.96 * se):.2f})",
                ratio_est=float(np.exp(d)), ratio_lo=float(np.exp(d - 1.96 * se)), ratio_hi=float(np.exp(d + 1.96 * se)),
                p=round(2 * norm.sf(abs(d / se)), 4), n=int(r.nobs))


def main():
    C = pd.read_parquet(cohort_path(COHORT_PRIMARY))
    E = C[C.primary == 1].copy()
    E["cat"] = np.select([E.uo_onset_h <= 8, (E.uo_onset_h > 8) & (E.uo_onset_h <= 24)], [1, 2], 0)
    assert (E.cat == 1).sum() == E.poly8.sum() and (E.cat == 2).sum() == E.poly_8_24.sum()
    Q = E.dropna(subset=XCOLS)
    Xe = sm.add_constant(Q[XCOLS].astype(float))
    r = sm.MNLogit(Q.cat, Xe).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": Q.hospitalid})
    res = dict(db="eICU-CRD", **eqtest(r))
    print(res)
    pd.DataFrame([res]).to_csv(os.path.join(out_dir("03"), "window_test_eicu.csv"), index=False)


if __name__ == "__main__":
    main()
