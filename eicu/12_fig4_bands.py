#!/usr/bin/env python3
"""
12_fig4_bands.py - eICU-CRD v2.0, manuscript v17: 95% confidence bands for the closed-cohort
time course around the vasopressin stop (Figure 4 of v17; Figure 5 of v16).

No new analysis: the anchor population is rebuilt with eicu_cohort.build_usage_cohort and
eicu_vpstop.anchor_population, and the closed cohort follows 07_switch_closed.py (in the ICU from
12 hr before to 8 hr after the anchor, urine recorded in >= 6 of the 12 hourly bins before and
>= 4 of the 8 bins after). The script checks that the number of stays and the plotted point
estimates equal closed_cohort_hourly.csv of script 07 and stops if they do not. It then resamples
stays with replacement within each group (NBOOT replicates, default 1,000; seed 2026) and writes
the 2.5th and 97.5th percentiles for each 2-hr interval.

Outputs  OUT_DIR/eicu_12/reproduction_checks.csv, fig4_bands.csv (aggregates only)
Run:     python eicu/12_fig4_bands.py   (after eicu/07_switch_closed.py)
"""
import os
import sys
import time
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import config  # noqa: E402
from eicu_cohort import build_usage_cohort, columns_differing_from_stored  # noqa: E402
from eicu_common import COHORT_USAGE, DB, RunLog, cohort_path, env_int, out_dir  # noqa: E402
from eicu_extract import load_source  # noqa: E402
from eicu_vpstop import anchor_population  # noqa: E402
from early_late_bands import SEED, band_table, check_against_hourly, public_frame, stay_arrays  # noqa: E402

NBOOT = env_int("NBOOT_BANDS", 1000)
COHORT16 = "closed (>=6/12 pre, >=4/8 post)"
GROUPS = [("NE-first", "NE-first", 297), ("VP-first", "VP-first", 135), ("NE without VP", "NE-only", 2099)]


def main():
    warnings.filterwarnings("ignore")
    t1 = time.time()
    say = RunLog().say
    out = out_dir("12")
    src = load_source(say)
    C, tabs = build_usage_cohort(src, say)
    bad = columns_differing_from_stored(C, pd.read_parquet(cohort_path(COHORT_USAGE)))
    assert not bad, f"rebuilt cohort differs from {COHORT_USAGE}: {bad}"
    anc = anchor_population(C, tabs.ur)
    A, HRA, RUN = anc.A, anc.HRA, anc.RUN
    # closed cohort: the lines of script 07
    npre = HRA[(HRA.h >= -12) & (HRA.h <= -1)].groupby("patientunitstayid").h.nunique()
    npost = HRA[(HRA.h >= 0) & (HRA.h <= 7)].groupby("patientunitstayid").h.nunique()
    A["n_rec_pre12"] = A.patientunitstayid.map(npre).fillna(0).astype(int)
    A["n_rec_post8"] = A.patientunitstayid.map(npost).fillna(0).astype(int)
    A["in_icu_m12_p8"] = ((A.anc - 720 >= 0) & (A.unitdischargeoffset >= A.anc + 480)).astype(int)
    H = HRA[(HRA.h >= -24) & (HRA.h <= 23)].copy()
    H["in_run"] = [h in RUN.get(s, ()) for s, h in zip(H.patientunitstayid, H.h)]
    H16 = pd.read_csv(os.path.join(config.OUT_DIR, "eicu_07", "closed_cohort_hourly.csv"))
    H16 = H16[H16.cohort == COHORT16]

    rep, rows = [], []
    for k, (gl, g, n_exp) in enumerate(GROUPS):
        X = A[(A.grp == g) & (A.in_icu_m12_p8 == 1) & (A.n_rec_pre12 >= 6) & (A.n_rec_post8 >= 4)]
        an = X.anc_name.iloc[0]
        rep.append(dict(check=f"closed cohort, {gl}: stays", value=len(X), expected=n_exp, match=len(X) == n_exp))
        h = H[H.patientunitstayid.isin(set(X.patientunitstayid))]
        r = band_table(stay_arrays(h, X.patientunitstayid.values, "patientunitstayid"), DB, gl, an, NBOOT, SEED + k)
        same, dmax = check_against_hourly(r, H16[H16.group == gl])
        rep.append(dict(check=f"closed cohort, {gl}: point estimates of the 24 intervals equal closed_cohort_hourly.csv of script 07",
                        value="largest difference %.1e" % dmax, expected="< 1e-9 and equal counts", match=same))
        rows += r
        say(gl, len(X), "max diff %.2e" % dmax)
    REP = pd.DataFrame(rep)
    REP.to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
    print(REP.to_string(index=False))
    if not REP.match.all():
        print("REPRODUCTION FAILED - stopped before writing the bands")
        sys.exit(1)
    public_frame(rows).to_csv(os.path.join(out, "fig4_bands.csv"), index=False)
    say("finished in %.0f s" % (time.time() - t1))


if __name__ == "__main__":
    main()
