#!/usr/bin/env python3
"""
03_cif_bootstrap.py - MIMIC-IV v3.1, bootstrap confidence intervals of the weighted
cumulative incidence curves of the primary analysis.

Uses impute, with_imp, ps_weights, event_data and aj of mimic_models.py on the analysis-ready
cohort table. Point estimates: ATT weights averaged over the imputations, Aalen-Johansen
cumulative incidence of polyuria 0-24 h (competing: death, ICU discharge, vasopressor
restart). CI: percentiles of a patient-level cluster bootstrap in which the propensity score
and the weights are re-estimated in every replicate (replicate b uses imputed data set
b mod m, seed 12345 + b).

Input   WORK_DIR/mimic/cohort/analysis_ready_mimic_primary.parquet
Output  OUT_DIR/mimic_03/fig2_cif_mimic.csv
Optional environment variables: NBOOT (default 500), NJOBS (10), M_IMP (10).

Run:  python mimic/03_cif_bootstrap.py   (after 02_primary_analysis.py)
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # folder of the shared modules
from mimic_common import config, PRIMARY_TABLE, cohort_file  # noqa: E402
from mimic_models import impute, with_imp, ps_weights, event_data, aj  # noqa: E402

NBOOT = int(os.environ.get("NBOOT", 500))
NJOBS = int(os.environ.get("NJOBS", 10))


def main():
    D = pd.read_parquet(cohort_file(PRIMARY_TABLE))
    IMPS = impute(D)
    P = D[D.pop_primary == 1]
    Wl = [ps_weights(with_imp(P, imp))[0] for imp in IMPS]
    Wbar = np.mean(Wl, axis=0)
    T, Cc = event_data(P)
    grid = np.arange(25)
    res = {}
    for g in ("VP", "NE"):
        s = (P.grp == g).values
        res[g, "w"] = aj(T[s], Cc[s], Wbar[s], grid=grid, causes=(1,))
        res[g, "c"] = aj(T[s], Cc[s], None, grid=grid, causes=(1,))

    subj = P.subject_id.values
    grp_idx = pd.Series(np.arange(len(P))).groupby(subj).apply(np.array).to_dict()
    us = np.unique(subj)

    def boot_cif(b, seed=12345):
        rng = np.random.default_rng(seed + b)
        pick = rng.choice(us, len(us), replace=True)
        idx = np.concatenate([grp_idx[s] for s in pick])
        x = with_imp(P, IMPS[b % len(IMPS)]).iloc[idx]
        try:
            w, _ = ps_weights(x)
        except Exception:
            return None
        T_, C_ = event_data(x)
        out = {}
        for g in ("VP", "NE"):
            s = (x.grp == g).values
            out[g] = aj(T_[s], C_[s], w[s], grid=grid, causes=(1,))[1]
        return out

    BR = Parallel(n_jobs=NJOBS, backend="threading")(delayed(boot_cif)(b) for b in range(NBOOT))
    ok = [r for r in BR if r is not None]

    rows = []
    lab = {"VP": "VP", "NE": "NE without VP"}
    for g in ("VP", "NE"):
        s = (P.grp == g).values
        Tg, Cg = T[s], Cc[s]
        B = np.array([r[g] for r in ok])
        lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
        for h in grid:
            rows.append(dict(db="MIMIC-IV v3.1", group=lab[g], hour=int(h), cif_weighted=res[g, "w"][1][h], lo=lo[h], hi=hi[h],
                             cif_crude=res[g, "c"][1][h], n_at_risk=int(res[g, "c"]["nrisk"][h]),
                             cum_events_polyuria=int(((Cg == 1) & (Tg <= h)).sum()),
                             cum_competing=int((np.isin(Cg, [2, 3, 4]) & (Tg <= h)).sum())))
    pd.DataFrame(rows).to_csv(os.path.join(config.out_dir("mimic_03"), "fig2_cif_mimic.csv"), index=False)


if __name__ == "__main__":
    main()
