"""
mimic_groups.py - MIMIC-IV v3.1: stop-order groups and group-dummy models shared by the
numbered MIMIC scripts.

Contents
  classify          pattern of vasopressin (VP) / norepinephrine (NE) use and the order in which
                    the two drugs were stopped; T0a(VP), T0a and T0b of VP-containing stays
  anchor_cols       T0-dependent columns of mimic_common.anchors() at an arbitrary anchor time
  shifted_anchors   anchor table with T0 of the VP-containing stays moved to another time
  extra_exposure    VP infusion time, peak VP rate and urine output in the 6 h before the anchor
  firth_fit, or_mi_terms, dummies, group_models
                    one logistic model with the group dummies as exposure (MICE, clustered SE);
                    Firth penalised regression for groups with < 10 events
  nN, smd           events / N (%) and the standardised mean difference

Definitions
  Vasopressors   NE 221906, VP 222315, phenylephrine 221749, epinephrine 221289, dopamine 221662,
                 angiotensin II 229709/229764 (dobutamine and milrinone are not vasopressors)
  Episode        rows of one drug joined when the gap is <= 1 h; "running at t" = start <= t < end
  NE-only        no VP record in the stay; anchor = end of the first NE episode
  VP-containing stays
     T0a(VP) = end of the first VP episode. T0b = first time at or after T0a(VP) when no
     vasopressor episode is running (advanced to the end of each running episode in turn).
     A VP episode starting between T0a(VP) and T0b = "VP restarted before liberation".
     Drugs started between the first vasopressor start and T0b classify the stay: no NE =
     VP-only; NE = NE+VP. NE+VP stays are split by diff = last NE end - T0a(VP) (h):
     diff < -1 = NE-first (VP stopped last), |diff| <= 1 = Simultaneous, diff > 1 = VP-first.
     T0a = min(last NE end, T0a(VP)).
"""
import numpy as np
import pandas as pd
import patsy
from scipy.special import expit

from mimic_common import VP, NE, VASOP, hrs, episodes
from mimic_models import COV_MI, with_imp, glm_fit, rubin, fmt

GAP = 1.0        # episode-joining gap, h
SIM_H = 1.0      # |last NE end - VP stop| <= SIM_H h = stopped simultaneously
G4 = ["NE-only", "VP-only", "NE-first", "Simultaneous", "VP-first"]
G3 = ["NE-only", "VP-only", "NE+VP"]


# ===========================================================================
# T0b and usage pattern
def classify(v):
    """Stop-order group and anchor times of every VP-containing stay; returns the classification
    table and the vasopressor episodes."""
    ep = episodes(v, GAP)
    ep = ep[ep.itemid.isin(VASOP)].copy()
    vp_st = set(ep.loc[ep.itemid == VP, "stay_id"])
    rows = []
    for sid, e in ep[ep.stay_id.isin(vp_st)].groupby("stay_id"):
        e = e.sort_values("start")
        evp = e[e.itemid == VP]
        t0a_vp = evp.end.iloc[0]
        st, en = e.start.values, e.end.values
        t = np.datetime64(t0a_vp)
        while True:
            run = (st <= t) & (en > t)
            if not run.any():
                break
            t = en[run].max()
        t0b = pd.Timestamp(t)
        pre = e[e.start < t0b]
        vp_restart = bool(((evp.start > t0a_vp) & (evp.start < t0b)).any())
        nee = pre[pre.itemid == NE]
        ne_last = nee.end.max() if len(nee) else pd.NaT
        oth = pre[~pre.itemid.isin([NE, VP])]
        other_after = bool(((oth.end > t0a_vp) & (oth.start < t0b)).any())
        if vp_restart:
            g = "VP restarted before liberation"
        elif not len(nee):
            g = "VP-only"
        else:
            dh = (ne_last - t0a_vp).total_seconds() / 3600
            g = "NE-first" if dh < -SIM_H else ("VP-first" if dh > SIM_H else "Simultaneous")
        rows.append(dict(stay_id=sid, ugrp=g, t0a_vp=t0a_vp, t0b=t0b, ne_last_end=ne_last,
                         first_press_start=e.start.min(), other_after_vp=int(other_after),
                         n_vp_ep_pre=int((evp.start < t0b).sum())))
    c = pd.DataFrame(rows)
    c["diff_ne_vp_h"] = hrs(c.ne_last_end, c.t0a_vp)
    c["t0a"] = c[["ne_last_end", "t0a_vp"]].min(axis=1)
    c["gap_t0a_t0b_h"] = hrs(c.t0b, c.t0a)
    c["gap_vp_t0b_h"] = hrs(c.t0b, c.t0a_vp)
    return c, ep


def anchor_cols(v, ep, d):
    """T0-dependent columns of mimic_common.anchors(), computed at an arbitrary t0 (for example T0b)
    for VP-containing stays (index drug = VP). d is indexed by stay_id and holds the column t0."""
    d = d.copy()
    r = v[v.rate.notna() & (v.rate > 0)].merge(d[["t0"]], left_on="stay_id", right_index=True)
    cur_all = r[(r.starttime <= r.t0) & (r.endtime > r.t0) & r.itemid.isin(VASOP)]
    d["vaso_any_at_t0b_check"] = d.index.isin(set(cur_all.stay_id)).astype(int)    # 0 by definition
    r = r[r.itemid != VP]
    cur_incl = r[(r.starttime <= r.t0) & (r.endtime >= r.t0)]
    cur = r[(r.starttime <= r.t0) & (r.endtime > r.t0)]
    d["vaso_at_t0"] = d.index.isin(set(cur.loc[cur.itemid.isin(VASOP), "stay_id"])).astype(int)
    d["vaso_at_t0_incl"] = d.index.isin(set(cur_incl.loc[cur_incl.itemid.isin(VASOP), "stay_id"])).astype(int)
    d["old_other_at_t0"] = d.index.isin(set(cur_incl.stay_id)).astype(int)
    d["dob_at_t0"] = d.index.isin(set(cur.loc[cur.itemid == 221653, "stay_id"])).astype(int)
    sw = r[r.itemid.isin(VASOP) & (r.starttime > r.t0) & (r.starttime <= r.t0 + pd.Timedelta(hours=1))]
    d["switch_1h"] = d.index.isin(set(sw.stay_id)).astype(int)
    ne = ep[ep.itemid == NE].merge(d[["t0"]], left_on="stay_id", right_index=True)
    d["ne_at_t0"] = d.index.isin(set(ne.loc[(ne.start <= ne.t0) & (ne.end > ne.t0), "stay_id"])).astype(int)
    e2 = ep[ep.itemid.isin(VASOP)].merge(d[["t0"]], left_on="stay_id", right_index=True)
    e2 = e2[e2.start > e2.t0]
    d["restart_t"] = e2.assign(h=hrs(e2.start, e2.t0)).groupby("stay_id").h.min()
    e3 = e2[e2.itemid == VP]
    d["idx_restart_t"] = e3.assign(h=hrs(e3.start, e3.t0)).groupby("stay_id").h.min()
    pre = v.merge(d[["t0"]], left_on="stay_id", right_index=True)
    pre = pre[pre.starttime < pre.t0]
    pv = pre[pre.itemid.isin(VASOP)]
    d["press_h"] = (d.t0 - pv.groupby("stay_id").starttime.min()).dt.total_seconds() / 3600
    d["ne_peak"] = pre[(pre.itemid == NE) & (pre.rateuom == "mcg/kg/min")].groupby("stay_id").rate.max() \
        .reindex(d.index).fillna(0)
    d["other_vaso_pre"] = d.index.isin(set(pre.loc[pre.itemid.isin(VASOP - {VP, NE}), "stay_id"])).astype(int)
    d["dopamine_pre"] = d.index.isin(set(pre.loc[pre.itemid == 221662, "stay_id"])).astype(int)
    d["weight"] = v.groupby("stay_id").weight.median().reindex(d.index)
    return d


def shifted_anchors(v, ep, d_orig, t0_new):
    """Anchor table for mimic_common.build(): the table of anchors() (d_orig, indexed by stay_id) with
    T0 of the VP-containing stays replaced by t0_new (Series indexed by stay_id) and their T0-dependent
    columns recomputed by anchor_cols(). NE stays are unchanged."""
    dvp = d_orig[d_orig.grp == "VP"].copy()
    dvp["t0"] = t0_new.reindex(dvp.index)
    newcols = anchor_cols(v, ep, dvp[["t0"]])
    for k in newcols.columns:
        dvp[k] = newcols[k]
    return pd.concat([d_orig[d_orig.grp == "NE"], dvp]).reset_index()


def extra_exposure(v, u, D):
    """VP infusion time, peak VP rate (U/min) and urine output in the 6 h before the anchor (mL/kg/h).
    v = vasopressor rows, u = urine rows, D = stay_id, t0 and weight of the stays."""
    t0 = D.set_index("stay_id").t0
    x = v[(v.itemid == VP) & v.stay_id.isin(t0.index)].copy()
    x["t0"] = x.stay_id.map(t0)
    x = x[x.starttime < x.t0]
    x["end_c"] = x[["endtime", "t0"]].min(axis=1)
    x["dur"] = hrs(x.end_c, x.starttime).clip(lower=0)
    # infusion time = union of intervals, so duplicated rows at the same time are not double-counted
    tot = {}
    for sid, g in x.sort_values(["stay_id", "starttime"]).groupby("stay_id"):
        s_, e_ = g.starttime.values, g.end_c.values
        acc, cs, ce = 0.0, s_[0], e_[0]
        for a, b in zip(s_[1:], e_[1:]):
            if a > ce:
                acc += (ce - cs) / np.timedelta64(1, "h")
                cs, ce = a, b
            else:
                ce = max(ce, b)
        acc += (ce - cs) / np.timedelta64(1, "h")
        tot[sid] = acc
    out = pd.DataFrame(index=t0.index)
    out["vp_dur_h"] = pd.Series(tot).reindex(out.index)
    ru = np.where(x.rateuom == "units/hour", x.rate / 60, np.where(x.rateuom == "units/min", x.rate, np.nan))
    out["vp_peak_umin"] = pd.Series(ru, index=x.index).groupby(x.stay_id).max().reindex(out.index)
    u = u[u.stay_id.isin(t0.index)]
    u = u.assign(h=hrs(u.charttime, u.stay_id.map(t0)))
    u6 = u[(u.h >= -6) & (u.h < 0)].groupby("stay_id").value.sum()
    w = D.set_index("stay_id").weight
    out["uo_pre6_mlkgh"] = (u6.reindex(out.index) / w.reindex(out.index) / 6).where(w.reindex(out.index) > 0)
    return out.reset_index().rename(columns={"index": "stay_id"})


# ===========================================================================
# Estimation tools
def firth_fit(y, X, maxit=200, tol=1e-8):
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        p = expit(X @ b)
        W = p * (1 - p)
        I = X.T @ (X * W[:, None])
        Ii = np.linalg.pinv(I)
        h = W * np.einsum("ij,jk,ik->i", X, Ii, X)
        U = X.T @ (y - p + h * (0.5 - p))
        step = Ii @ U
        mx = np.abs(step).max()
        if mx > 5:
            step *= 5 / mx
        b = b + step
        if np.abs(step).max() < tol:
            break
    p = expit(X @ b)
    I = X.T @ (X * (p * (1 - p))[:, None])
    return b, np.linalg.pinv(I)


def or_mi_terms(d, imps, y, terms, cov=COV_MI, firth=False, with_imp=with_imp):
    """One model with the group dummies as exposure. GLM: subject-clustered robust SE; Firth: model-based SE;
    pooled with Rubin's rules."""
    expo = "+".join(terms)
    f = f"{y}~{expo}+{cov}" if cov else f"{y}~{expo}"
    acc = {t: ([], []) for t in terms}
    for imp in imps:
        x = with_imp(d, imp)
        if firth:
            yy, X = patsy.dmatrices(f, x, return_type="dataframe")
            b, V = firth_fit(yy.values.ravel(), X.values)
            cols = list(X.columns)
            for t in terms:
                k = cols.index(t)
                acc[t][0].append(b[k])
                acc[t][1].append(V[k, k])
        else:
            r = glm_fit(f, x)
            for t in terms:
                acc[t][0].append(r.params[t])
                acc[t][1].append(r.cov_params().loc[t, t])
    out = {}
    for t in terms:
        q, se, lo, hi, p = rubin(*acc[t])
        out[t] = dict(OR=np.exp(q), lo=np.exp(lo), hi=np.exp(hi), p=p)
    return out


def nN(s, y):
    n, N = int(s[y].sum()), len(s)
    return "%d/%d (%.1f%%)" % (n, N, 100 * n / N if N else np.nan)


def dummies(d, col, levels):
    d = d.copy()
    names = []
    for g in levels[1:]:
        nm = "g_" + g.replace("-", "").replace("+", "p").replace(" ", "")
        d[nm] = (d[col] == g).astype(int)
        names.append(nm)
    return d, names


def group_models(d, imps, y, col, levels, cov=COV_MI, label="", with_imp=with_imp):
    """Crude and adjusted (MICE, clustered SE) models. If a group has < 10 events a Firth model (model-based SE)
    is also fitted and used for that group. The columns *_est, *_lo and *_hi hold the reported estimate
    (the Firth estimate for a group with < 10 events) at full precision."""
    d, names = dummies(d, col, levels)
    ev = {g: int(d.loc[d[col] == g, y].sum()) for g in levels}
    rows = []
    res = {}
    for kind, cv in (("crude", None), ("adjusted", cov)):
        try:
            o = or_mi_terms(d, imps, y, names, cv, with_imp=with_imp)
        except Exception:
            o = {n: None for n in names}
        sparse = [g for g in levels[1:] if ev[g] < 10]
        of = None
        if sparse and ev[levels[0]] > 0:
            try:
                of = or_mi_terms(d, imps, y, names, cv, firth=True, with_imp=with_imp)
            except Exception:
                of = None
        res[kind] = (o, of)
    for g, nm in zip(levels[1:], names):
        r = dict(analysis=label, outcome=y, group=g, events=nN(d[d[col] == g], y),
                 ref_NE_only=nN(d[d[col] == levels[0]], y), n_model=len(d))
        for kind in ("crude", "adjusted"):
            o, of = res[kind]
            if ev[g] < 10:
                shown = None if ev[g] == 0 or of is None else of[nm]
                r[f"{kind}_OR"] = "not estimable (<10 events)" if shown is None else "Firth " + fmt(shown) + " (<10 events; model-based SE)"
            else:
                shown = o[nm]
                r[f"{kind}_OR"] = fmt(shown) if shown is not None else "fit failed"
            r[f"{kind}_p"] = round(shown["p"], 4) if shown is not None else np.nan
            if shown is not None:       # est, lo, hi: the reported estimate at full precision
                r[f"{kind}_est"], r[f"{kind}_lo"], r[f"{kind}_hi"] = shown["OR"], shown["lo"], shown["hi"]
        rows.append(r)
    return rows


def smd(a, b):
    a, b = pd.Series(a, dtype=float).dropna(), pd.Series(b, dtype=float).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sd = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / sd if sd > 0 else 0.0
