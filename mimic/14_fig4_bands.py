#!/usr/bin/env python3
"""
14_fig4_bands.py - MIMIC-IV v3.1, manuscript v17: 95% confidence bands for the closed-cohort
time course around the vasopressin stop (Figure 4 of v17; Figure 5 of v16).

No new analysis: the closed cohort of 07_vpstop_timing.py is rebuilt with the functions of that
script (alive and in the ICU from 12 hr before to 8 hr after the anchor, urine recorded in >= 6 of
the 12 hourly bins before and >= 4 of the 8 bins after). The script checks that the number of
stays and the plotted point estimates equal closed_cohort_hourly.csv of script 07 and stops if
they do not. It then resamples stays with replacement within each group (NBOOT replicates,
default 1,000; seed 2026) and writes the 2.5th and 97.5th percentiles for each 2-hr interval.

Outputs  OUT_DIR/mimic_14/reproduction_checks.csv, fig4_bands.csv (aggregates only)
Run:     python mimic/14_fig4_bands.py   (after mimic/07_vpstop_timing.py)
"""
import importlib.util
import os
import sys
import time
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import mimic_common as mc  # noqa: E402
from mimic_common import config  # noqa: E402
from early_late_bands import SEED, band_table, check_against_hourly, public_frame, stay_arrays  # noqa: E402

warnings.filterwarnings("ignore")
NBOOT = int(os.environ.get("NBOOT_BANDS", 1000))
GROUPS = [("NE-first", "NE-first", "VP_stop", 605), ("VP-first", "VP-first", "VP_stop", 504), ("NE without VP", "NE-only", "NE_stop", 3304)]


def load_script07():
    """07_vpstop_timing.py as a module (its file name is not a valid module name); the script body runs only under __main__."""
    spec = importlib.util.spec_from_file_location("vpstop_timing_07", os.path.join(HERE, "07_vpstop_timing.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    t1 = time.time()
    s07 = load_script07()
    say = s07.say
    out = config.out_dir("mimic_14")
    mc.check_extracts()
    mc.build_caches(say)
    u = mc.load_all(say)["u"]
    D0 = pd.read_parquet(mc.cohort_file(mc.PRIMARY_TABLE))
    DB = pd.read_parquet(mc.cohort_file(mc.USAGE_TABLE))
    # populations: the lines of timing_analyses() in script 07
    D0["ugrp"] = D0.stay_id.map(DB.set_index("stay_id").ugrp)
    D0.loc[D0.grp == "NE", "ugrp"] = "NE-only"
    base0 = (D0.ex_age == 0) & (D0.ex_notinicu == 0) & (D0.ex_deadpre == 0) & (D0.rrt_any == 0) & (D0.ex_neuroendo == 0)
    base0 &= (D0.grp == "VP") | (D0.vaso_at_t0 == 0)
    pop0 = D0[base0]
    hr0 = s07.in_run_flags(s07.bins_window(u, pop0[["stay_id", "t0", "weight"]], lo=-26, hi=72))
    H16 = pd.read_csv(os.path.join(config.OUT_DIR, "mimic_07", "closed_cohort_hourly.csv"))
    H16 = H16[H16.cohort == "closed"]

    rep, rows = [], []
    for k, (gl, ug, anc, n_exp) in enumerate(GROUPS):
        pb = pop0[pop0.ugrp == ug]
        p = pb.set_index("stay_id")
        t_lo, t_hi = p.t0 - pd.Timedelta(hours=12), p.t0 + pd.Timedelta(hours=8)
        ok = (p.intime <= t_lo) & (p.outtime >= t_hi) & (p.deathtime.isna() | (p.deathtime >= t_hi))
        pre, post = s07.per_stay(pb, hr0, -12, -1), s07.per_stay(pb, hr0, 0, 7)
        ok &= (pre.nb.fillna(0).values >= 6) & (post.nb.fillna(0).values >= 4)
        cp = pb[ok.values]
        rep.append(dict(check=f"closed cohort, {gl}: stays", value=len(cp), expected=n_exp, match=len(cp) == n_exp))
        h = hr0[hr0.stay_id.isin(set(cp.stay_id))]
        r = band_table(stay_arrays(h, cp.stay_id.values, "stay_id"), "MIMIC-IV", gl, anc, NBOOT, SEED + k)
        same, dmax = check_against_hourly(r, H16[H16.group == gl])
        rep.append(dict(check=f"closed cohort, {gl}: point estimates of the 24 intervals equal closed_cohort_hourly.csv of script 07",
                        value="largest difference %.1e" % dmax, expected="< 1e-9 and equal counts", match=same))
        rows += r
        say(gl, len(cp), "max diff %.2e" % dmax)
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
