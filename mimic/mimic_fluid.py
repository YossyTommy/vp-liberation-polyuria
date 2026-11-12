"""
mimic_fluid.py - MIMIC-IV v3.1: cumulative fluid balance, FO%-augmented imputation and overlap
weighting shared by the numbered MIMIC scripts.

Contents
  prep_io, win_in, fluid_vars   input, output, balance and fluid overload % (FO%) per stay
  with_imp_fo, impute_fo        imputation with FO% added to the imputed variables
  ato_est, ato_analysis         overlap weighting (ATO) with a subject-level cluster bootstrap
  wsmd                          unweighted and weighted standardised mean difference

Fluid variables (icu/inputevents.csv.gz, icu/outputevents.csv.gz)
  input     all inputevents rows with amountuom == "mL" and amount > 0 (the row rule of
            fluid_pre24_l in mimic_common.build; includes drug carriers and enteral feeds),
            apportioned by time over starttime-endtime into the window [a, b); rows with
            starttime == endtime are points at starttime. Rows charted in "L" are not used.
  output    all outputevents items with charttime in [a, b), values as recorded (negatives kept)
  GU irrigation   the urine rule counts 227489 "GU Irrigant/Urine Volume Out" gross and does not
            subtract 227488. For consistency 227489 enters output gross and 227488 "GU Irrigant
            Volume In" is added to input, so the balance equals net urine. fo_pct_nogu removes
            both sides as a sensitivity variable.
  windows   cum = [ICU admission, T0); pre24 = [max(ICU admission, T0 - 24 h), T0);
            post8 = [T0, T0 + 8 h); post24 = [T0, T0 + 24 h)
  coverage  a window without any input row (or any output row) is missing, not 0; balance and
            FO% are missing if either side is missing.
            FO% = (input - output) L / body weight kg x 100
  limits    fluids and output before ICU admission and insensible losses are not included
"""
import numpy as np
import pandas as pd
import patsy

from mimic_models import IMPVARS, COV_MI, impute, with_imp

GU_IN, GU_OUT = 227488, 227489
COV_FO = COV_MI + "+fo_i"


# ===========================================================================
# Fluid balance
def prep_io(ie, oe):
    io_counts = dict(ie_rows_cohort=len(ie), ie_mL_rows=int((ie.amountuom == "mL").sum()),
                 ie_mL_amount_le0_rows_dropped=int(((ie.amountuom == "mL") & ~(ie.amount > 0)).sum()),
                 ie_L_unit_rows_not_used=int((ie.amountuom == "L").sum()),
                 ie_mL_rewritten_rows=int(((ie.amountuom == "mL") & (ie.statusdescription == "Rewritten")).sum()),
                 ie_mL_bolus_zero_duration_rows=int(((ie.amountuom == "mL") & (ie.amount > 0) & (ie.endtime <= ie.starttime)).sum()),
                 oe_rows_cohort=len(oe), oe_negative_value_rows_kept=int((oe.value < 0).sum()),
                 oe_227488_GU_in_rows_moved_to_input=int((oe.itemid == GU_IN).sum()),
                 oe_227489_GU_out_rows_kept_in_output=int((oe.itemid == GU_OUT).sum()),
                 oe_valueuom=",".join(sorted(oe.valueuom.dropna().unique())))
    I = ie[(ie.amountuom == "mL") & (ie.amount > 0)][["stay_id", "starttime", "endtime", "amount", "itemid"]].copy()
    g = oe[oe.itemid == GU_IN]
    I = pd.concat([I, pd.DataFrame({"stay_id": g.stay_id.values, "starttime": g.charttime.values, "endtime": g.charttime.values,
                                    "amount": g.value.values, "itemid": GU_IN})], ignore_index=True)
    O = oe[oe.itemid != GU_IN][["stay_id", "charttime", "itemid", "value"]].copy()
    return I, O, io_counts


def win_in(I, a, b):
    """Volume apportioned to [a, b); rows of zero duration are points at starttime."""
    s, e = I.starttime.values, I.endtime.values
    dur = (e - s) / np.timedelta64(1, "h")
    ov = (np.minimum(e, b) - np.maximum(s, a)) / np.timedelta64(1, "h")
    pos = dur > 0
    frac = np.where(pos, np.clip(ov, 0, None) / np.where(pos, dur, 1.0), ((s >= a) & (s < b)).astype(float))
    return I.amount.values * frac, frac > 0


def fluid_vars(I, O, anc):
    anc = anc.set_index("stay_id")
    H = pd.Timedelta(hours=1)
    W = {"cum": (anc.intime, anc.t0), "pre24": (np.maximum(anc.intime, anc.t0 - 24 * H), anc.t0),
         "post8": (anc.t0, anc.t0 + 8 * H), "post24": (anc.t0, anc.t0 + 24 * H)}
    Ii, Oo = I[I.stay_id.isin(anc.index)], O[O.stay_id.isin(anc.index)]
    out = pd.DataFrame(index=anc.index)
    for k, (a, b) in W.items():
        amt, hit = win_in(Ii, Ii.stay_id.map(a).values, Ii.stay_id.map(b).values)
        x = pd.DataFrame({"stay_id": Ii.stay_id.values, "amt": amt, "hit": hit, "gu": (Ii.itemid.values == GU_IN) * amt})
        gs = x.groupby("stay_id").agg(amt=("amt", "sum"), hit=("hit", "sum"), gu=("gu", "sum"))
        cov = gs.hit.reindex(out.index).fillna(0) > 0
        out[f"in_{k}_l"] = (gs.amt.reindex(out.index) / 1000).where(cov)
        out[f"in_cov_{k}"] = cov.astype(int)
        out[f"gu_in_{k}_l"] = (gs.gu.reindex(out.index).fillna(0) / 1000).where(cov)
        if k in ("cum", "pre24"):
            oa, ob = Oo.stay_id.map(a).values, Oo.stay_id.map(b).values
            oo = Oo[(Oo.charttime.values >= oa) & (Oo.charttime.values < ob)]
            oo = oo.assign(gu=np.where(oo.itemid == GU_OUT, oo.value, 0.0))
            gs = oo.groupby("stay_id").agg(v=("value", "sum"), n=("value", "size"), gu=("gu", "sum"))
            ocov = gs.n.reindex(out.index).fillna(0) > 0
            out[f"out_{k}_l"] = (gs.v.reindex(out.index) / 1000).where(ocov)
            out[f"out_cov_{k}"] = ocov.astype(int)
            out[f"gu_out_{k}_l"] = (gs.gu.reindex(out.index).fillna(0) / 1000).where(ocov)
            out[f"bal_{k}_l"] = out[f"in_{k}_l"] - out[f"out_{k}_l"]
    out["icu_pre_h"] = (anc.t0 - anc.intime).dt.total_seconds() / 3600
    out["pre24_truncated"] = (out.icu_pre_h < 24).astype(int)
    w = anc.weight.where(anc.weight > 0)
    out["fo_pct"] = out.bal_cum_l / w * 100                      # 1 L ≈ 1 kg
    out["fo_pct_nogu"] = (out.bal_cum_l - out.gu_in_cum_l.fillna(0) + out.gu_out_cum_l.fillna(0)) / w * 100
    out["bal_pre24_mlkg"] = out.bal_pre24_l * 1000 / w
    out["gu_irrig_cum"] = ((out.gu_in_cum_l.fillna(0) > 0) | (out.gu_out_cum_l.fillna(0) > 0)).astype(int)
    return out.reset_index()


# ===========================================================================
# Imputation with FO%
def with_imp_fo(d, imp):
    """with_imp of mimic_models plus the imputed FO% (fo_i, and fo5_i = FO% / 5) when the imputation
    contains fo_pct; the four imputed covariates are unchanged."""
    x = with_imp(d, imp)
    if "fo_pct" in imp.columns:
        x["fo_i"] = imp.loc[d.index, "fo_pct"]
        x["fo5_i"] = x.fo_i / 5
    return x


def impute_fo(d):
    """impute() of mimic_models with fo_pct added to the imputed variables."""
    return impute(d, impvars=IMPVARS + ["fo_pct"])


# --- overlap weighting -----------------------------------------------------
def _logit(X, y, w=None, it=50):
    b = np.zeros(X.shape[1])
    w = np.ones(len(y)) if w is None else w
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = w * p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        st = np.linalg.solve(H, X.T @ (w * (y - p)))
        b += st
        if np.abs(st).max() < 1e-8:
            break
    return b


def ato_est(X, z, Y, idx):
    Xb, zb = X[idx], z[idx]
    b = _logit(Xb, zb)
    ps = 1 / (1 + np.exp(-(Xb @ b)))
    w = np.where(zb == 1, 1 - ps, ps)
    res = {}
    for k, yv in Y.items():
        yb = yv[idx]
        p1 = np.sum(w * zb * yb) / np.sum(w * zb)
        p0 = np.sum(w * (1 - zb) * yb) / np.sum(w * (1 - zb))
        res[k] = (p1, p0, np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0)), p1 - p0)
    return res, w, ps


def ato_analysis(d, imps, ys, ps_cov, nboot, with_imp, seed=2026):
    """with_imp = function that merges one imputation into d (it must provide the variables of ps_cov).
    Per imputation: PS (logit: vp ~ ps_cov) -> ATO weights -> weighted proportions, marginal OR and RD.
    Log OR and RD are averaged over the imputations. CI = percentiles of a subject-level cluster bootstrap
    (imputations fixed; all imputations re-estimated in each replicate)."""
    d = d.reset_index(drop=False).rename(columns={"index": "_ix"}).set_index("_ix")
    Xs, z = [], d.vp.values.astype(float)
    Y = {y: d[y].values.astype(float) for y in ys}
    for imp in imps:
        x = with_imp(d, imp)
        Xs.append(patsy.dmatrix(ps_cov, x, return_type="dataframe").values)
    full = np.arange(len(d))
    pts = [ato_est(X, z, Y, full) for X in Xs]
    point = {y: dict(p1=np.mean([p[0][y][0] for p in pts]), p0=np.mean([p[0][y][1] for p in pts]),
                     lor=np.mean([p[0][y][2] for p in pts]), rd=np.mean([p[0][y][3] for p in pts])) for y in ys}
    subj = d.subject_id.values
    us = np.unique(subj)
    pos = pd.Series(np.arange(len(d))).groupby(subj).apply(np.array).to_dict()
    rng = np.random.RandomState(seed)
    B = {y: [] for y in ys}
    for b in range(nboot):
        pick = rng.choice(us, len(us), replace=True)
        idx = np.concatenate([pos[s] for s in pick])
        rs = [ato_est(X, z, Y, idx)[0] for X in Xs]
        for y in ys:
            B[y].append((np.mean([r[y][2] for r in rs]), np.mean([r[y][3] for r in rs])))
    out = {}
    for y in ys:
        a = np.array(B[y])
        out[y] = dict(point[y], lor_lo=np.percentile(a[:, 0], 2.5), lor_hi=np.percentile(a[:, 0], 97.5),
                      rd_lo=np.percentile(a[:, 1], 2.5), rd_hi=np.percentile(a[:, 1], 97.5), nboot=len(a))
    return out, pts, Xs


def wsmd(x, z, w, col):
    v = pd.to_numeric(x[col], errors="coerce").values.astype(float)
    ok = ~np.isnan(v)
    v, zz, ww = v[ok], z[ok], w[ok]
    a, b = v[zz == 1], v[zz == 0]
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if sd == 0:
        return 0.0, 0.0
    un = (a.mean() - b.mean()) / sd
    wt = (np.average(a, weights=ww[zz == 1]) - np.average(b, weights=ww[zz == 0])) / sd
    return un, wt
