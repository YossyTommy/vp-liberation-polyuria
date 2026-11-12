#!/usr/bin/env python3
"""
04_window_test.py - MIMIC-IV v3.1, comparison of the 0-8 h and >8-24 h windows.

Multinomial logistic regression in the primary population (complete cases) with a three-level
outcome: no polyuria within 24 h (reference), onset within 0-8 h, onset within >8-24 h.
Covariates are those of the primary model (KDIGO stage as categories including "unknown");
standard errors are clustered on subject_id. The ratio of the two odds ratios (early / late)
and its Wald test use the full covariance matrix of the two vasopressin coefficients.

Input   WORK_DIR/mimic/cohort/analysis_ready_mimic_primary.parquet
Output  OUT_DIR/mimic_04/window_test_mimic.csv

Run:  python mimic/04_window_test.py   (after 02_primary_analysis.py)
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # folder of the shared modules
from mimic_common import config, PRIMARY_TABLE, cohort_file  # noqa: E402

COV = ["vp", "age10", "male", "cvicu", "na_pre", "map_base", "log_lact", "diur_pre24", "fluid_pre24_l", "log_press_h",
       "ne_peak_c", "other_vaso_pre", "vent_t0", "steroid_pre24", "inotrope_t0", "sepsis", "card_surg", "emergency"]


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
    D = pd.read_parquet(cohort_file(PRIMARY_TABLE))
    P = D[D.pop_primary == 1].copy()
    P["cat"] = np.select([P.onset <= 8, (P.onset > 8) & (P.onset <= 24)], [1, 2], 0)
    Q = P.dropna(subset=COV).copy()
    Kd = pd.get_dummies(Q.kdigo_cat, prefix="k", drop_first=True).astype(float)
    Xm = sm.add_constant(pd.concat([Q[COV].astype(float), Kd], axis=1))
    r = sm.MNLogit(Q.cat, Xm).fit(disp=0, cov_type="cluster", cov_kwds={"groups": Q.subject_id})
    res = dict(db="MIMIC-IV", **eqtest(r))
    print(res)
    pd.DataFrame([res]).to_csv(os.path.join(config.out_dir("mimic_04"), "window_test_mimic.csv"), index=False)


if __name__ == "__main__":
    main()
