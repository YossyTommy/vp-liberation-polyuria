#!/usr/bin/env python3
"""
09_full_charting_table.py - MIMIC-IV v3.1, characteristics of stays with and without complete
urine charting.

Primary population (pop_primary == 1) of the analysis-ready cohort table, by group (VP / NE) and
by whether urine was recorded in all 24 hourly bins after T0 (obs24 == 24) or in fewer
(obs24 < 24; the primary population requires >= 12). Rows: number of stays; cardiac surgery,
cardiovascular ICU, sepsis and invasive ventilation at T0 as n/N (%); vasopressor time before
T0 as median [IQR]; ICU discharge and death within 24 h of T0; new-onset polyuria within 24 h
and within 0-8 h; number of recorded hourly urine bins as median [IQR].

Input   WORK_DIR/mimic/cohort/analysis_ready_mimic_primary.parquet
Output  OUT_DIR/mimic_09/etable11_full_charting.csv
        one row per characteristic; for each of the four columns (VP 24/24, VP <24, NE 24/24,
        NE <24) the formatted cell and the numbers behind it (n and N, or median, q1 and q3)

Run:  python mimic/09_full_charting_table.py   (after 02_primary_analysis.py)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # folder of the shared modules
from mimic_common import config, PRIMARY_TABLE, cohort_file  # noqa: E402

COLUMNS = [("VP 24/24", "VP", True), ("VP <24", "VP", False), ("NE 24/24", "NE", True), ("NE <24", "NE", False)]
BINARY = [("Cardiac surgery", "card_surg"), ("Cardiovascular ICU", "cvicu"), ("Sepsis", "sepsis")]


def n_pct(n, N):
    """n/N (%) with thousands separators."""
    return f"{n:,}/{N:,} ({100 * n / N:.1f}%)"


def count_row(label, variable, flags):
    """Row of counts; flags maps a column name to a boolean Series over the stays of that column."""
    r = dict(characteristic=label, variable=variable, statistic="n/N (%)")
    for col, f in flags.items():
        n, N = int(f.sum()), len(f)
        r[col] = n_pct(n, N)
        r[f"{col} n"], r[f"{col} N"] = n, N
    return r


def median_row(label, variable, values, digits):
    """Row of medians [IQR]; values maps a column name to a numeric Series."""
    r = dict(characteristic=label, variable=variable, statistic="median [IQR]")
    for col, x in values.items():
        med, q1, q3 = x.median(), x.quantile(.25), x.quantile(.75)
        r[col] = f"{med:.{digits}f} [{q1:.{digits}f}–{q3:.{digits}f}]"
        r[f"{col} median"], r[f"{col} q1"], r[f"{col} q3"] = med, q1, q3
    return r


def main():
    D = pd.read_parquet(cohort_file(PRIMARY_TABLE))
    P = D[D.pop_primary == 1]
    sub = {col: P[(P.grp == g) & ((P.obs24 == 24) if full else (P.obs24 < 24))] for col, g, full in COLUMNS}
    hours = pd.Timedelta(hours=24)
    rows = [dict(characteristic="Stays", variable="stay_id", statistic="n",
                 **{col: f"{len(s):,}" for col, s in sub.items()}, **{f"{col} n": len(s) for col, s in sub.items()})]
    for label, var in BINARY:
        rows.append(count_row(label, var, {col: s[var] == 1 for col, s in sub.items()}))
    rows.append(median_row("Vasopressor time before T0, hr", "press_h", {col: s.press_h for col, s in sub.items()}, 1))
    rows.append(count_row("Invasive ventilation", "vent_t0", {col: s.vent_t0 == 1 for col, s in sub.items()}))
    rows.append(count_row("ICU discharge within 24 hr", "outtime", {col: s.outtime <= s.t0 + hours for col, s in sub.items()}))
    rows.append(count_row("Death within 24 hr", "deathtime", {col: s.deathtime <= s.t0 + hours for col, s in sub.items()}))
    rows.append(count_row("New-onset polyuria within 24 hr", "poly24", {col: s.poly24 == 1 for col, s in sub.items()}))
    rows.append(count_row("Onset 0-8 hr", "poly8", {col: s.poly8 == 1 for col, s in sub.items()}))
    rows.append(median_row("Recorded urine bins", "obs24", {col: s.obs24 for col, s in sub.items()}, 0))
    T = pd.DataFrame(rows)
    names = [c for c, _, _ in COLUMNS]
    order = ["characteristic", "variable", "statistic"] + names + \
        [f"{c} {k}" for c in names for k in ("n", "N", "median", "q1", "q3")]
    T = T[order]
    for c in names:
        for k in ("n", "N"):
            T[f"{c} {k}"] = T[f"{c} {k}"].astype("Int64")      # counts as integers, empty where not applicable
    print(T[["characteristic"] + names].to_string(index=False))
    T.to_csv(os.path.join(config.out_dir("mimic_09"), "etable11_full_charting.csv"), index=False)


if __name__ == "__main__":
    main()
