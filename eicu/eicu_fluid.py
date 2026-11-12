"""Fluid-balance variables from intakeOutput and overlap weighting (eICU-CRD v2.0).

Offsets are minutes from ICU admission (= 0).
  * Records: cellpath category 'Intake (ml)' -> intake (0 < v < 5000 mL); 'Output (ml)' ->
    urine (urine labels, 0 <= v < 3000 mL, a positive volume charted at the same time under
    two urine labels counted once) + other output (drains, NG, stool, chest tubes, blood loss,
    fluid removal; 0 <= v < 5000 mL; count/occurrence labels removed). 'Dialysis (ml)' and
    'Weight' cells are not used. Exact duplicate rows (stay, cellpath, offset, value) are removed.
  * Time apportioning: a record charted at offset t covers (t - L, t], L = minutes since the
    previous record of the same label in the same stay if 0 < gap <= LMAX, otherwise LDEF.
    Volume is allocated to a window in proportion to the overlap.
  * Windows relative to the anchor T: cumulative (0, T]; pre-24 h (max(0, T - 1440), T];
    post 0-8 h (T, T + 480]; post 0-24 h (T, T + 1440].
  * Coverage: a quantity is missing (never 0) when no record of that type overlaps the window.
    Balance and FO% require both intake and output records.
    FO% = (input - output, L) / weight (kg) x 100. T <= ICU admission -> cumulative window
    undefined. Dense charting = intake records in >= 50% of clock hours from admission to T.
"""
import os
import subprocess

import numpy as np
import pandas as pd
import statsmodels.api as sm

from eicu_cohort import URINE_EXCL_RE, URINE_RE
from eicu_common import ex_fluid_dir, raw

OUT_EXCL_RE = r"count|occurrence|number of|^incontinent$"
LMAX, LDEF = 240.0, 60.0


def load_intake_output(stay_ids, say):
    """intakeOutput rows of the given stays (all columns) with the cellpath category `cat` and
    the numeric value `v`. The filtered table is cached in WORK_DIR/eicu/ex_fluid."""
    exdir = ex_fluid_dir()
    idf = os.path.join(exdir, "ids.txt")
    xio = os.path.join(exdir, "x_io.csv")
    pd.Series(sorted(stay_ids)).to_csv(idf, index=False, header=False)
    if not os.path.exists(xio):
        say("extracting intakeOutput for", len(stay_ids), "stays")
        cmd = (f"gunzip -c '{raw('intakeOutput.csv.gz')}' | LC_ALL=C awk -v idf='{idf}' -F',' "
               "'BEGIN{while((getline l<idf)>0)k[l]=1} NR==1{print;next} ($2 in k){print}'" + f" > '{xio}.tmp' && mv '{xio}.tmp' '{xio}'")
        subprocess.run(cmd, shell=True, check=True)
    io = pd.read_csv(xio, low_memory=False)
    add_category_and_value(io)
    return io


def add_category_and_value(io):
    """Add the cellpath category (fourth path element) and the numeric cell value, in place."""
    io["cat"] = io.cellpath.astype(str).str.split("|").str[3]
    io["v"] = pd.to_numeric(io.cellvaluenumeric, errors="coerce")


def build_records(io, point=False):
    """Intake, urine and other-output records with the interval (s, e] each record covers.
    point=True attributes the whole volume to the charting time. Returns the records and
    a dict of processing counts."""
    v = io[io.cat.isin(["Intake (ml)", "Output (ml)"])].copy()
    n0 = len(v)
    v = v[~v.duplicated(["patientunitstayid", "cellpath", "intakeoutputoffset", "v"])]
    ndup = n0 - len(v)
    lab = v.celllabel.astype(str)
    isur = (v.cat == "Output (ml)") & lab.str.contains(URINE_RE, case=False) & ~lab.str.contains(URINE_EXCL_RE, case=False)
    v["kind"] = np.select([v.cat == "Intake (ml)", isur, v.cat == "Output (ml)"], ["in", "urine", "oth"], "x")
    keep = (((v.kind == "in") & (v.v > 0) & (v.v < 5000)) | ((v.kind == "urine") & (v.v >= 0) & (v.v < 3000)) |
            ((v.kind == "oth") & (v.v >= 0) & (v.v < 5000) & ~lab.str.contains(OUT_EXCL_RE, case=False)))
    info = dict(intake_output_rows=n0, exact_duplicates_removed=ndup,
                intake_removed_nonpositive_or_ge5000=int(((v.kind == "in") & ~keep).sum()),
                urine_removed_negative_or_ge3000=int(((v.kind == "urine") & ~keep).sum()),
                other_output_removed_count_label_or_range=int(((v.kind == "oth") & ~keep).sum()))
    v = v[keep & v.v.notna()]
    u = v[v.kind == "urine"]
    z = u.v == 0
    u = pd.concat([u[z], u[~z].drop_duplicates(["patientunitstayid", "intakeoutputoffset", "v"])])
    info["urine_mirrored_entries_removed"] = int((v.kind == "urine").sum() - len(u))
    v = pd.concat([v[v.kind != "urine"], u]).sort_values(["patientunitstayid", "kind", "celllabel", "intakeoutputoffset"])
    gap = v.groupby(["patientunitstayid", "kind", "celllabel"]).intakeoutputoffset.diff()
    v["L"] = 1e-6 if point else np.where((gap > 0) & (gap <= LMAX), gap, LDEF)
    v["s"] = v.intakeoutputoffset - v.L
    v["e"] = v.intakeoutputoffset.astype(float)
    for k in ("in", "urine", "oth"):
        info[f"records_used_{k}"] = int((v.kind == k).sum())
    return v[["patientunitstayid", "kind", "celllabel", "intakeoutputoffset", "v", "s", "e", "L"]], info


def fluid_vars(REC, T):
    """Apportioned volumes, coverage flags, balances and FO% per stay and anchor.
    T has the columns patientunitstayid, t0 (anchor, minutes from ICU admission) and weight_kg."""
    T = T[["patientunitstayid", "t0", "weight_kg"]].drop_duplicates(["patientunitstayid", "t0"]).copy()
    R = REC.merge(T[["patientunitstayid", "t0"]], on="patientunitstayid")
    t0 = R.t0.values.astype(float)
    win = {"cum": (np.zeros(len(R)), t0), "pre24": (np.maximum(0, t0 - 1440), t0), "post8": (t0, t0 + 480), "post24": (t0, t0 + 1440)}
    out = T.set_index(["patientunitstayid", "t0"])
    for w, (a, b) in win.items():
        ov = np.clip(np.minimum(R.e.values, b) - np.maximum(R.s.values, a), 0, None)
        R["_vol"] = R.v.values * (ov / R.L.values)
        R["_hit"] = ov > 0
        for k in ("in", "urine", "oth"):
            g = R[(R.kind == k) & R._hit].groupby(["patientunitstayid", "t0"])
            out[f"{k}_{w}_ml"] = g._vol.sum()
            out[f"{k}_{w}_nrec"] = g.size()
        for k in ("in", "out"):
            m = (R.kind == "in") if k == "in" else R.kind.isin(["urine", "oth"])
            m = m & (R.intakeoutputoffset > a) & (R.intakeoutputoffset <= b)
            out[f"{k}_{w}_hours"] = R[m].assign(hb=np.floor(R.intakeoutputoffset[m] / 60)).groupby(["patientunitstayid", "t0"]).hb.nunique()
    out = out.reset_index()
    for w in win:
        for c in ("in", "urine", "oth"):
            out[f"{c}_{w}_nrec"] = out[f"{c}_{w}_nrec"].fillna(0).astype(int)
        for c in ("in", "out"):
            out[f"{c}_{w}_hours"] = out[f"{c}_{w}_hours"].fillna(0).astype(int)
        out[f"has_in_{w}"] = (out[f"in_{w}_nrec"] > 0).astype(int)
        out[f"has_out_{w}"] = ((out[f"urine_{w}_nrec"] + out[f"oth_{w}_nrec"]) > 0).astype(int)
    out["cum_win_h"] = np.where(out.t0 > 0, out.t0 / 60, np.nan)
    out["pre24_win_h"] = np.where(out.t0 > 0, np.minimum(out.t0, 1440) / 60, np.nan)
    valid = out.t0 > 0
    for w in ("cum", "pre24"):
        hi = out[f"has_in_{w}"] == 1
        ho = out[f"has_out_{w}"] == 1
        inl = out[f"in_{w}_ml"].fillna(0) / 1000
        outl = (out[f"urine_{w}_ml"].fillna(0) + out[f"oth_{w}_ml"].fillna(0)) / 1000
        out[f"{w}_in_l"] = np.where(valid & hi, inl, np.nan)
        out[f"{w}_out_l"] = np.where(valid & ho, outl, np.nan)
        out[f"{w}_urine_l"] = np.where(valid & ho, out[f"urine_{w}_ml"].fillna(0) / 1000, np.nan)
        out[f"{w}_bal_l"] = np.where(valid & hi & ho, inl - outl, np.nan)
        out[f"cov_{w}"] = np.select([~valid, hi & ho, hi & ~ho, ~hi & ho], ["T0b<=ICU admission", "intake+output", "intake only", "output only"], "neither")
        out[f"in_{w}_hourfrac"] = np.where(valid, out[f"in_{w}_hours"] / np.ceil(out[f"{w}_win_h"]).clip(lower=1), np.nan)
    out["fo_pct"] = out.cum_bal_l / out.weight_kg * 100
    out["fo_rec"] = out.fo_pct.notna().astype(int)
    out["fo_dense"] = ((out.fo_rec == 1) & (out.in_cum_hourfrac >= 0.5)).astype(int)
    out["fo_pct_dense"] = out.fo_pct.where(out.fo_dense == 1)
    out["pre24_rec"] = out.pre24_bal_l.notna().astype(int)
    for w in ("post8", "post24"):
        out[f"{w}_in_l"] = np.where(out[f"has_in_{w}"] == 1, out[f"in_{w}_ml"].fillna(0) / 1000, np.nan)
    return out.drop(columns=["weight_kg"])


def ato_fit(s, xcols, ys):
    """Overlap weighting: propensity score of s.vp on xcols (logistic), weights 1 - ps (vp = 1)
    and ps (vp = 0). Returns the log odds ratios of the weighted outcome proportions for each
    outcome in ys, the weights and the propensity scores."""
    X = s[xcols].astype(float)
    X = X[[c for c in X.columns if X[c].nunique() > 1]]
    ps = sm.Logit(s.vp.astype(float), sm.add_constant(X)).fit(disp=0, maxiter=200).predict(sm.add_constant(X))
    w = np.where(s.vp == 1, 1 - ps, ps)
    out = []
    for y in ys:
        t = s.vp.values == 1
        yy = s[y].values.astype(float)
        p1 = np.average(yy[t], weights=w[t])
        p0 = np.average(yy[~t], weights=w[~t])
        out.append(np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0)))
    return np.array(out), w, ps
