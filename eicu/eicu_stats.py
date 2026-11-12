"""Statistical functions shared by the eICU-CRD scripts.

Covariate set, multiple imputation (chained equations, Rubin's rules), logistic regression
with hospital-clustered standard errors, Firth regression, propensity-score weights for the
average treatment effect in the treated, standardised mean differences, competing-risk times,
the Aalen-Johansen cumulative incidence and the hospital-cluster bootstrap.

MICE_M (number of imputations, default 10) can be set with the environment variable MICE_M.
"""
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from eicu_common import env_int

assert enable_iterative_imputer      # importing this module switches IterativeImputer on

MICE_M = env_int("MICE_M", 10)

COV = ["age10", "male", "cvicu", "kdigo_1", "kdigo_2", "kdigo_3", "kdigo_unk", "na_pre", "map_base",
       "diur_pre24", "apache10", "log_press_h", "ne_peak01", "ne_dose_unknown", "other_press_pre",
       "mv_t0", "steroid_pre24", "inotrope_t0", "sepsis", "card_surg", "elective"]
MISS = ["na_pre", "map_base", "apache10"]      # covariates with missing values (imputed)
LAB = {"vp": "VP vs NE-only", "age10": "Age (+10 y)", "male": "Male", "cvicu": "Cardiac ICU",
       "kdigo_1": "KDIGO 1", "kdigo_2": "KDIGO 2", "kdigo_3": "KDIGO 3", "kdigo_unk": "KDIGO unknown",
       "na_pre": "Na before T0 (+1)", "map_base": "MAP 6 h before T0 (+1)", "diur_pre24": "Loop diuretic 24 h before T0",
       "apache10": "APACHE IV (+10)", "log_press_h": "log(1+vasopressor h before T0)",
       "ne_peak01": "Peak NE before T0 (+0.1 mcg/kg/min)", "ne_dose_unknown": "NE dose not convertible",
       "other_press_pre": "Other vasopressor before T0", "mv_t0": "Mechanical ventilation at T0",
       "steroid_pre24": "Corticosteroid 24 h before T0", "inotrope_t0": "Inotrope at T0", "sepsis": "Sepsis",
       "card_surg": "Cardiac surgery", "elective": "Elective surgical admission"}


def fmt(o, lo, hi):
    """Estimate with interval, two decimals."""
    return f"{o:.2f} ({lo:.2f}–{hi:.2f})"


def rate(x):
    """'events/n (%)' of a 0/1 series."""
    return f"{int(x.sum())}/{len(x)} ({100 * x.mean():.1f}%)"


def impute(df, cols, outcomes, m, seed0=0):
    """m completed copies of df with the columns `cols` imputed by chained equations
    (posterior draws; `outcomes` enter the imputation model). One copy when nothing is missing."""
    X = df[cols + outcomes].astype(float)
    if not X[cols].isna().any().any():
        return [df.copy()] * 1
    outs = []
    for k in range(m):
        imp = IterativeImputer(sample_posterior=True, max_iter=10, random_state=seed0 + k)
        Z = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
        d2 = df.copy()
        for c in cols:
            d2[c] = Z[c]
        outs.append(d2)
    return outs


def logit_cluster(df, y, xs):
    """Logistic regression with hospital-clustered covariance; constant columns are dropped."""
    keep = [c for c in xs if df[c].nunique() > 1]
    X = sm.add_constant(df[keep].astype(float))
    r = sm.Logit(df[y].astype(float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": df.hospitalid}, maxiter=200)
    return r.params, r.cov_params()


def fit_or(df, y, expo, covs, mice=True, m=MICE_M, label="", full=False, miss=MISS):
    """Odds ratio of `expo` from a logistic regression with hospital-clustered SE.

    When `mice` is set and a covariate listed in `miss` has missing values, all covariates are
    imputed (m data sets) and pooled with Rubin's rules; otherwise complete cases are used.
    Returns a result row (and the table of all coefficients when `full`)."""
    d = df.dropna(subset=[y, "hospitalid"]).copy()
    cc = [c for c in covs if c != expo]
    xs = [expo] + cc
    if mice and d[[c for c in miss if c in cc]].isna().any().any():
        sets = impute(d, cc, [y, expo], m)
    else:
        d = d.dropna(subset=xs)
        sets = [d]
    P, U = [], []
    for s in sets:
        p, v = logit_cluster(s, y, xs)
        P.append(p)
        U.append(v)
    P = pd.concat(P, axis=1).T
    Q = P.mean()
    W = sum(U) / len(U)
    B = P.cov() if len(P) > 1 else 0 * W
    T = W + (1 + 1 / len(P)) * B
    se = np.sqrt(np.diag(T))
    se = pd.Series(se, index=Q.index)
    res = pd.DataFrame({"OR": np.exp(Q), "lo": np.exp(Q - 1.96 * se), "hi": np.exp(Q + 1.96 * se),
                        "p": 2 * norm.sf(np.abs(Q / se))}).drop("const")
    n = len(sets[0])
    ev = int(sets[0][y].sum())
    r = res.loc[expo]
    row = dict(analysis=label, outcome=y, exposure=expo, n=n, events=int(ev),
               n_exposed=int(sets[0][expo].sum()), events_exposed=int(sets[0].loc[sets[0][expo] == 1, y].sum()),
               n_hospitals=int(sets[0].hospitalid.nunique()), method=("MICE m=%d" % len(sets)) if len(sets) > 1 else "complete case",
               OR=r.OR, lo=r.lo, hi=r.hi, p=r.p, aOR_95CI=fmt(r.OR, r.lo, r.hi))
    return (row, res) if full else row


def ps_weights(df, covs, trim=0.99):
    """Weights for the average treatment effect in the treated (VP = 1, NE-only = odds of the
    propensity score, truncated at the `trim` quantile). Missing covariates: median + indicator.
    Returns the weights and the design matrix of the propensity model."""
    d = df.copy()
    X = d[covs].astype(float)
    for c in MISS:
        if c in covs:
            d[c + "_miss"] = X[c].isna().astype(int)
            X[c + "_miss"] = d[c + "_miss"]
            X[c] = X[c].fillna(X[c].median())
    X = X[[c for c in X.columns if X[c].nunique() > 1]]
    ps = sm.Logit(d.vp.astype(float), sm.add_constant(X)).fit(disp=0, maxiter=200).predict(sm.add_constant(X))
    w = np.where(d.vp == 1, 1.0, ps / (1 - ps))
    ne = d.vp == 0
    cap = np.quantile(w[ne], trim)
    w = np.where(ne, np.minimum(w, cap), w)
    return pd.Series(w, index=d.index), X


def smd(x, t, w=None):
    """Standardised mean difference between t == 1 and t == 0 (optionally weighted means;
    pooled SD from the unweighted samples)."""
    w = np.ones(len(x)) if w is None else np.asarray(w)
    x = np.asarray(x, float)
    t = np.asarray(t)
    m1 = np.average(x[t == 1], weights=w[t == 1])
    m0 = np.average(x[t == 0], weights=w[t == 0])
    s1 = x[t == 1].var()
    s0 = x[t == 0].var()
    s = np.sqrt((s1 + s0) / 2)
    return (m1 - m0) / s if s > 0 else np.nan


def smd2(xg, xr):
    """Standardised mean difference of two samples (missing values dropped), two decimals."""
    xg = pd.Series(xg).dropna().astype(float)
    xr = pd.Series(xr).dropna().astype(float)
    if len(xg) < 2 or len(xr) < 2:
        return np.nan
    s = np.sqrt((xg.var() + xr.var()) / 2)
    return round((xg.mean() - xr.mean()) / s, 2) if s > 0 else np.nan


def evalue(rr):
    """E-value of a risk ratio."""
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


def cr_times(df, use_restart=True, horizon=72.0):
    """Time (hours) and type of the first event: 1 polyuria, 2 ICU death, 3 ICU discharge alive,
    4 vasopressor restart, 0 none by `horizon`. Ties go to the competing event."""
    tp = df.uo_onset_h.fillna(np.inf).values
    to = df.obs_after_t0_h.values
    tdeath = np.where(df.died_icu.values == 1, to, np.inf)
    tdis = np.where(df.died_icu.values == 0, to, np.inf)
    trs = df.restart_h.fillna(np.inf).values if use_restart else np.full(len(df), np.inf)
    T = np.vstack([tdeath, tdis, trs, tp]).T
    code = np.array([2, 3, 4, 1])
    j = np.argmin(T, axis=1)
    t = T[np.arange(len(df)), j]
    s = code[j]
    s = np.where(t > horizon, 0, s)
    t = np.minimum(t, horizon)
    return t, s


def aj_cif(t, s, w=None, grid=np.arange(0, 73)):
    """Aalen-Johansen cumulative incidence of event type 1 on `grid` (optionally weighted)."""
    w = np.ones(len(t)) if w is None else np.asarray(w, float)
    o = np.argsort(t)
    t, s, w = t[o], s[o], w[o]
    ut = np.unique(t)
    idx = np.searchsorted(t, ut, side="left")
    atrisk = w[::-1].cumsum()[::-1][idx]
    d1 = np.bincount(np.searchsorted(ut, t), weights=w * (s == 1), minlength=len(ut))
    da = np.bincount(np.searchsorted(ut, t), weights=w * (s > 0), minlength=len(ut))
    S = np.concatenate([[1.0], np.cumprod(1 - da / atrisk)[:-1]])
    cif = np.cumsum(S * d1 / atrisk)
    k = np.searchsorted(ut, grid, side="right") - 1
    return np.where(k >= 0, cif[np.clip(k, 0, None)], 0.0)


def hosp_boot(df, fn, B, seed=2026):
    """Hospital-cluster bootstrap: fn(resampled data) for B resamples of whole hospitals;
    replicates in which fn fails are dropped."""
    rng = np.random.default_rng(seed)
    hs = df.hospitalid.unique()
    g = {h: ix for h, ix in df.groupby("hospitalid").indices.items()}
    out = []
    for _ in range(B):
        pick = rng.choice(hs, len(hs), replace=True)
        ix = np.concatenate([g[h] for h in pick])
        try:
            out.append(fn(df.iloc[ix].reset_index(drop=True)))
        except Exception:
            pass
    return np.array(out)


def firth_fit(X, y, maxit=200, tol=1e-8):
    """Firth penalised logistic regression (Newton steps with step halving). Returns the
    coefficients and the model-based covariance matrix."""
    b = np.zeros(X.shape[1])

    def pll(bb):
        e = X @ bb
        p = 1 / (1 + np.exp(-e))
        W = p * (1 - p)
        ld = np.linalg.slogdet(X.T @ (X * W[:, None]))[1]
        return np.sum(y * e - np.logaddexp(0, e)) + 0.5 * ld

    cur = pll(b)
    for _ in range(maxit):
        e = X @ b
        p = 1 / (1 + np.exp(-e))
        W = p * (1 - p)
        info = X.T @ (X * W[:, None])
        Ii = np.linalg.inv(info)
        A = X * np.sqrt(W)[:, None]
        h = np.einsum("ij,jk,ik->i", A, Ii, A)
        step = Ii @ (X.T @ (y - p + h * (0.5 - p)))
        lam = 1.0
        while lam > 1e-4:
            nb = b + lam * step
            nv = pll(nb)
            if nv >= cur - 1e-10:
                break
            lam /= 2
        b = nb
        cur = nv
        if np.max(np.abs(lam * step)) < tol:
            break
    e = X @ b
    p = 1 / (1 + np.exp(-e))
    W = p * (1 - p)
    return b, np.linalg.inv(X.T @ (X * W[:, None]))


def fit_groups(df, y, gcol, groups, ref="NE-only", covs=COV, mice=True, m=MICE_M, label="", force_firth=None, miss=MISS):
    """Odds ratios of each group in `groups` versus `ref` from one logistic model.

    Hospital-clustered SE; Firth regression (model-based SE) when any group in the model has
    fewer than 10 events (or as forced). Covariates are imputed (m data sets, Rubin's rules)
    when `mice` is set and a covariate listed in `miss` has missing values."""
    d = df[df[gcol].isin([ref] + groups)].dropna(subset=[y, "hospitalid"]).copy()
    dums = []
    for g in groups:
        k = "G_" + re.sub(r"[^A-Za-z]", "", g)
        d[k] = (d[gcol] == g).astype(int)
        dums.append(k)
    cc = [c for c in covs]
    evs = {g: int(d.loc[d[gcol] == g, y].sum()) for g in [ref] + groups}
    firth = (min(evs.values()) < 10) if force_firth is None else force_firth
    if mice and len(cc) and d[[c for c in miss if c in cc]].isna().any().any():
        sets = impute(d, cc, [y] + dums, m)
    else:
        d = d.dropna(subset=dums + cc)
        sets = [d]
    keep = [c for c in dums + cc if sets[0][c].nunique() > 1]
    P, Uu = [], []
    for s in sets:
        X = sm.add_constant(s[keep].astype(float), has_constant="add")
        if firth:
            b, V = firth_fit(X.values, s[y].astype(float).values)
            P.append(pd.Series(b, index=X.columns))
            Uu.append(pd.DataFrame(V, index=X.columns, columns=X.columns))
        else:
            r = sm.Logit(s[y].astype(float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": s.hospitalid}, maxiter=300)
            P.append(r.params)
            Uu.append(r.cov_params())
    P = pd.concat(P, axis=1).T
    Q = P.mean()
    W = sum(Uu) / len(Uu)
    B = P.cov() if len(P) > 1 else 0 * W
    se = pd.Series(np.sqrt(np.diag(W + (1 + 1 / len(P)) * B)), index=Q.index)
    out = []
    for g, k in zip(groups, dums):
        x = d[d[gcol] == g]
        base = dict(analysis=label, outcome=y, group=g, ref=ref, n_model=len(sets[0]), events_group=f"{evs[g]}/{len(x)} ({100*evs[g]/max(len(x),1):.1f}%)",
                    events_ref=f"{evs[ref]}/{int((d[gcol]==ref).sum())} ({100*evs[ref]/max((d[gcol]==ref).sum(),1):.1f}%)",
                    method=("Firth (model-based SE)" if firth else "logistic, hospital-clustered SE") + (f", MICE m={len(sets)}" if len(sets) > 1 else ", complete case"))
        if k not in Q.index:
            base.update(OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan, OR_95CI="not estimable (no variation)")
        else:
            o, lo, hi = np.exp(Q[k]), np.exp(Q[k] - 1.96 * se[k]), np.exp(Q[k] + 1.96 * se[k])
            base.update(OR=o, lo=lo, hi=hi, p=2 * norm.sf(abs(Q[k] / se[k])), OR_95CI=fmt(o, lo, hi))
        out.append(base)
    return out
