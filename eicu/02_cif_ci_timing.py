#!/usr/bin/env python3
"""
02_cif_ci_timing.py - eICU-CRD v2.0, bootstrap confidence intervals of the weighted
cumulative incidence curves and sensitivity to the timing rules.

The two-group cohort of 01_primary_analysis.py is rebuilt with eicu_cohort.build_primary_cohort
once per timing variant. Only two quantities are varied:
  gap  join threshold of infusion records (minutes; primary 180)
  dur  duration credited to a point record (minutes; primary 60): the "running at T0" window
       [T0 - dur, T0] and the episode duration end - start + dur

Outputs (OUT_DIR/eicu_02/)
  fig2_cif_eicu.csv              hourly ATT-weighted and crude cumulative incidence 0-24 h with
                                 percentile CI from a hospital-cluster bootstrap in which the
                                 propensity score is re-estimated in every replicate
  eicu_timing_sensitivity.csv    adjusted odds ratios under each timing variant
Cohort tables of the variants are written to WORK_DIR/eicu/cohort_timing_rules/<variant>/.
Optional environment variables: NBOOT (default 500), MICE_M (10).

Run:  python eicu/02_cif_ci_timing.py   (re-uses the extracts of 01_primary_analysis.py)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # eicu_* modules; needed when Python runs in safe-path mode
from eicu_cohort import build_primary_cohort  # noqa: E402
from eicu_common import RunLog, env_int, out_dir, timing_cohort_path  # noqa: E402
from eicu_extract import load_source  # noqa: E402
from eicu_stats import COV, aj_cif, cr_times, fit_or, hosp_boot, ps_weights  # noqa: E402

NBOOT = env_int("NBOOT", 500)
GRID = np.arange(0, 25)
VARIANTS = [("base: join 180 min, point record 60 min", 180, 60, "base"), ("(a) join threshold 60 min", 60, 60, "a_gap60"),
            ("(b) join threshold 360 min", 360, 60, "b_gap360"), ("(c) point record lasts 30 min", 180, 30, "c_dur30"),
            ("(d) point record lasts 2 h", 180, 120, "d_dur120")]


def cif_both(d):
    """ATT-weighted cumulative incidence on GRID in the VP group followed by the NE-only group."""
    tt, ss = cr_times(d)
    ww = ps_weights(d, COV)[0].values
    v = d.vp.values == 1
    return np.concatenate([aj_cif(tt[v], ss[v], ww[v], GRID), aj_cif(tt[~v], ss[~v], ww[~v], GRID)])


def weighted_cif_table(PR):
    """Weighted cumulative incidence with bootstrap CI, crude incidence and numbers at risk."""
    BS = hosp_boot(PR, cif_both, B=NBOOT, seed=2026)
    est = cif_both(PR)
    t, s = cr_times(PR)
    rows = []
    for k, (g, lab) in enumerate(((1, "VP"), (0, "NE without VP"))):
        mk = PR.vp.values == g
        tg, sg = t[mk], s[mk]
        crude = aj_cif(tg, sg, None, GRID)
        lo, hi = np.percentile(BS[:, k * 25:(k + 1) * 25], [2.5, 97.5], axis=0)
        for j, h in enumerate(GRID):
            e1 = int(((sg == 1) & (tg <= h)).sum())
            ec = int(((sg > 1) & (tg <= h)).sum())
            ce = int(((sg == 0) & (tg <= h)).sum())
            rows.append(dict(db="eICU-CRD v2.0", group=lab, hour=int(h), cif_weighted=est[k * 25 + j], lo=lo[j], hi=hi[j],
                             cif_crude=crude[j], n_at_risk=int(mk.sum()) - e1 - ec - ce, cum_events_polyuria=e1, cum_competing=ec))
    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore")
    say = RunLog().say
    out = out_dir("02")
    src = load_source(say)

    def build(gap, dur, tag):
        C, _ = build_primary_cohort(src, say, gap=float(gap), dur=float(dur))
        C.to_parquet(timing_cohort_path(tag), index=False)
        return C

    Cb = build(180, 60, "base")
    PR = Cb[Cb.primary == 1].reset_index(drop=True)
    weighted_cif_table(PR).to_csv(os.path.join(out, "fig2_cif_eicu.csv"), index=False)

    # ---- timing-rule variants ---------------------------------------------------------------
    bt0 = Cb.set_index("patientunitstayid").t0
    bpr = set(Cb.loc[Cb.primary == 1, "patientunitstayid"])
    sens = []
    for lab, gap, dur, tag in VARIANTS:
        C_ = Cb if tag == "base" else build(gap, dur, tag)
        P_ = C_[C_.primary == 1]
        t0c = C_.set_index("patientunitstayid").t0.reindex(bt0.index)
        row = dict(variant=lab, join_threshold_min=gap, point_record_min=dur, n_VP=int(P_.vp.sum()), n_NE=int((P_.vp == 0).sum()),
                   pct_index_T0_changed_vs_base=round(100 * np.mean(~np.isclose(t0c.values, bt0.values)), 1),
                   n_primary_shared_with_base=len(set(P_.patientunitstayid) & bpr))
        for y, yl in (("polyuria_24h", "24h"), ("poly8", "0_8h"), ("poly_8_24", "8_24h")):
            r = fit_or(P_, y, "vp", COV)
            row.update({f"{yl}_events_VP": r["events_exposed"], f"{yl}_events_NE": r["events"] - r["events_exposed"],
                        f"{yl}_aOR": round(r["OR"], 3), f"{yl}_lo": round(r["lo"], 3), f"{yl}_hi": round(r["hi"], 3),
                        f"{yl}_aOR_95CI": r["aOR_95CI"]})
        row["method"] = r["method"] + ", hospital-clustered SE"
        sens.append(row)
    pd.DataFrame(sens).to_csv(os.path.join(out, "eicu_timing_sensitivity.csv"), index=False)


if __name__ == "__main__":
    main()
