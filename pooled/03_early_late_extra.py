#!/usr/bin/env python3
"""
03_early_late_extra.py - pooled estimates of the second set of exploratory analyses on early and
late polyuria (mimic/11_early_late_extra.py, eicu/09_early_late_extra.py; not part of the
manuscript analyses).

Pooling follows pooled/01_pooled_estimates.py: fixed-effect inverse-variance weighting of the
database-specific estimates at full precision, SE = (upper - lower) / 3.92 on the log scale
(odds ratios) or on the original scale (linear coefficients); I2 from Cochran's Q. Rows that
were not estimable in one database are not pooled.

Inputs   OUT_DIR/mimic_11/ and OUT_DIR/eicu_09/: E1_multinomial5.csv, E2_regression_onset_anchored.csv,
         E3_death_models.csv, E4_vp_dose_models.csv
Outputs  OUT_DIR/pooled_03/E1_multinomial5_pooled.csv, E2_regression_pooled.csv,
         E3_death_models_pooled.csv, E4_vp_dose_models_pooled.csv
Run:     python pooled/03_early_late_extra.py   (after mimic/11 and eicu/09)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from early_late_helpers import pool_fixed  # noqa: E402


def pooled_table(M, E, keys, ratio, est="OR"):
    """Merge the two database tables on `keys` and pool `est` (ratio scale when `ratio`)."""
    rows = []
    for _, m in M.iterrows():
        e = E
        for k in keys:
            e = e[e[k] == m[k]]
        r = {k: m[k] for k in keys}
        ok = len(e) == 1 and pd.notna(m.get(est)) and pd.notna(e.iloc[0].get(est))
        if not ok:
            r["note"] = "not pooled (not estimable in one database)"
            rows.append(r)
            continue
        e = e.iloc[0]
        f = np.log if ratio else (lambda x: x)
        g = np.exp if ratio else (lambda x: x)
        q, sp, p, i2 = pool_fixed([f(m[est]), f(e[est])], [(f(m.hi) - f(m.lo)) / 3.92, (f(e.hi) - f(e.lo)) / 3.92])
        fm = "%.2f (%.2f–%.2f)" if ratio else "%.2f (%.2f to %.2f)"
        r.update(mimic=fm % (m[est], m.lo, m.hi), mimic_p=m.p, eicu=fm % (e[est], e.lo, e.hi), eicu_p=e.p,
                 pooled=fm % (g(q), g(q - 1.96 * sp), g(q + 1.96 * sp)), pooled_p=p, i2_percent=round(i2),
                 est=g(q), lo=g(q - 1.96 * sp), hi=g(q + 1.96 * sp))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    o = lambda *p: os.path.join(config.OUT_DIR, *p)
    out = config.out_dir("pooled_03")
    rd = lambda f: (pd.read_csv(o("mimic_11", f)), pd.read_csv(o("eicu_09", f)))
    for f, keys, ratio, est, name in (("E1_multinomial5.csv", ["analysis", "term"], True, "OR", "E1_multinomial5_pooled.csv"),
                                      ("E2_regression_onset_anchored.csv", ["variable", "term"], False, "est", "E2_regression_pooled.csv"),
                                      ("E3_death_models.csv", ["analysis", "population", "term"], True, "OR", "E3_death_models_pooled.csv"),
                                      ("E4_vp_dose_models.csv", ["exposure", "outcome", "model"], True, "OR", "E4_vp_dose_models_pooled.csv")):
        M, E = rd(f)
        T = pooled_table(M, E, keys, ratio, est)
        T.to_csv(os.path.join(out, name), index=False)
        cols = [c for c in keys + ["mimic", "eicu", "pooled", "pooled_p", "i2_percent", "note"] if c in T.columns]
        print(name)
        print(T[cols].to_string(index=False))


if __name__ == "__main__":
    main()
