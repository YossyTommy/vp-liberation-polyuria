"""
early_late_bands.py - shared helper of mimic/14_fig4_bands.py and eicu/12_fig4_bands.py (manuscript v17).

Stay-level bootstrap intervals for the closed-cohort time course around the vasopressin stop
(Figure 4 of manuscript v17; Figure 5 of v16, Figure 4 of v15). The point estimates are those of
the existing figure: hours are grouped in 2-hr intervals; the percentage of recorded stay-hours
that belong to a polyuria run is sum(in-run stay-hours) / sum(recorded stay-hours), and the mean
hourly urine output is the mean over all recorded stay-hours of the interval.
"""
import numpy as np
import pandas as pd

HOURS = np.arange(-24, 24)
NB2 = 24                       # number of 2-hr intervals
SEED = 2026


def stay_arrays(h, ids, idcol):
    """h: stay-hour rows (idcol, h, rate, in_run) of the stays `ids`, hours -24..23.
    Returns an array (n stays, 24 intervals, 3): recorded stay-hours, in-run stay-hours, sum of rates."""
    ids = np.asarray(ids)
    pos = {s: i for i, s in enumerate(ids)}
    A = np.zeros((len(ids), NB2, 3))
    h = h[(h.h >= -24) & (h.h <= 23)]
    i = h[idcol].map(pos).values
    b = ((h.h.values + 24) // 2).astype(int)
    np.add.at(A, (i, b, 0), 1.0)
    np.add.at(A, (i, b, 1), h.in_run.values.astype(float))
    np.add.at(A, (i, b, 2), h.rate.values.astype(float))
    return A


def band_table(A, db, group, anchor, nboot, seed):
    """Point estimates and percentile intervals (2.5th, 97.5th) from `nboot` resamples of stays."""
    n = A.shape[0]
    tot = A.sum(axis=0)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=nboot).astype(float)      # (nboot, n) resampling counts
    T = (W @ A.reshape(n, -1)).reshape(nboot, NB2, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = 100 * T[:, :, 1] / T[:, :, 0]
        uo = T[:, :, 2] / T[:, :, 0]
    rows = []
    for b in range(NB2):
        rows.append(dict(db=db, group=group, anchor=anchor, n_stays=n, interval_start_hr=-24 + 2 * b, interval_mid_hr=-24 + 2 * b + 1,
                         in_run_pct=100 * tot[b, 1] / tot[b, 0], in_run_pct_lo=np.nanpercentile(pct[:, b], 2.5), in_run_pct_hi=np.nanpercentile(pct[:, b], 97.5),
                         uo_mean=tot[b, 2] / tot[b, 0], uo_mean_lo=np.nanpercentile(uo[:, b], 2.5), uo_mean_hi=np.nanpercentile(uo[:, b], 97.5),
                         n_boot=nboot, seed=seed, _rec=tot[b, 0], _run=tot[b, 1], _sum=tot[b, 2]))
    return rows


def check_against_hourly(rows, hourly):
    """Compare the point estimates with the hourly table already written by script 07 (same grouping
    as the existing figure). Returns (all equal, largest absolute difference of the two plotted values)."""
    x = hourly.copy()
    x["b"] = (x.hour // 2) * 2
    x["w"] = x.uo_mean * x.n_recorded
    s = x.groupby("b")[["n_in_run", "n_recorded", "w"]].sum()
    ok, dmax = True, 0.0
    for r in rows:
        q = s.loc[r["interval_start_hr"]]
        ok &= int(q.n_recorded) == int(r["_rec"]) and int(q.n_in_run) == int(r["_run"])
        dmax = max(dmax, abs(100 * q.n_in_run / q.n_recorded - r["in_run_pct"]), abs(q.w / q.n_recorded - r["uo_mean"]))
    return bool(ok and dmax < 1e-9), dmax


def public_frame(rows):
    """Aggregate table for OUT_DIR: percentages, means and intervals only (no counts per interval)."""
    return pd.DataFrame(rows).drop(columns=["_rec", "_run", "_sum"])
