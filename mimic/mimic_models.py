"""
mimic_models.py - MIMIC-IV v3.1: multiple imputation and estimation tools shared by the
numbered MIMIC scripts.

Contents
  covariate sets    COVS / COV_MI (models on imputed data), COV_CC (complete case)
  impute            MICE (statsmodels MICEData, predictive mean matching k = 5, fixed seed);
                    all extracted stays are imputed once, analysis populations are subsets
  with_imp          adds the imputed covariates of one imputation to an analysis table
  glm_fit, rubin, or_mi   binomial GLM with standard errors clustered on subject_id, pooled
                    with Rubin's rules
  ps_weights        propensity score and ATT weights (NE weights truncated)
  event_data, aj    competing-risk event times and the weighted Aalen-Johansen estimator

Functions that fit models on imputed data take the function that merges an imputation into
the analysis table as the argument with_imp. The default is with_imp of this module; analyses
that use additional imputed variables pass their own function (for example with_imp_fo of
mimic_fluid.py, which adds the imputed fluid overload %).

Optional environment variable: M_IMP (number of imputations, default 10).
"""
import os

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.imputation.mice import MICEData
from scipy import stats

M_IMP = int(os.environ.get("M_IMP", 10))
SEED = 2026

IMPVARS = ["kdigo_pre", "na_pre", "map_base", "log_lact"]
AUXV = ["vp", "age10", "male", "cvicu", "diur_pre24", "fluid_pre24_l", "log_press_h", "ne_peak_c", "other_vaso_pre",
        "vent_t0", "steroid_pre24", "inotrope_t0", "sepsis", "card_surg", "emergency", "vaso_at_t0", "poly24", "poly8",
        "eval12", "pre_poly"]
COVS = ["age10", "male", "cvicu", "C(kdigo_i)", "na_i", "map_i", "diur_pre24", "fluid_pre24_l", "lact_i",
        "log_press_h", "ne_peak_c", "other_vaso_pre", "vent_t0", "steroid_pre24", "inotrope_t0", "sepsis",
        "card_surg", "emergency"]
COV_MI = "+".join(COVS)
COV_CC = COV_MI.replace("C(kdigo_i)", "C(kdigo_cat)").replace("na_i", "na_pre").replace("map_i", "map_base") \
    .replace("lact_i", "lactate_pre24")
YS = ["poly24", "poly8", "poly_8_24"]


def cov_without(*drop):
    return "+".join(c for c in COVS if c not in drop)


def impute(d, m=M_IMP, n_burn=10, seed=SEED, impvars=None, auxv=None):
    """MICE (statsmodels MICEData, predictive mean matching k=5). All extracted stays are imputed once;
    analysis populations are subsets. impvars = variables to impute (default IMPVARS), auxv = auxiliary
    variables of the imputation model (default AUXV). Returns m data frames with the imputed variables."""
    impvars = IMPVARS if impvars is None else impvars
    auxv = AUXV if auxv is None else auxv
    X = d[impvars + auxv].astype(float).reset_index(drop=True)
    np.random.seed(seed)
    md = MICEData(X, k_pmm=5, rng=np.random.RandomState(seed))   # fixed random state for reproducibility
    md.update_all(n_burn)
    out = []
    for _ in range(m):
        md.update_all(5)
        out.append(md.data[impvars].copy().set_index(d.index))
    return out


def with_imp(d, imp):
    """Copy of d with the imputed covariates of one imputation (kdigo_i, na_i, map_i, lact_i)."""
    x = d.copy()
    x["kdigo_i"] = imp.loc[d.index, "kdigo_pre"].round().clip(0, 3).astype(int).astype(str)
    x["na_i"] = imp.loc[d.index, "na_pre"]
    x["map_i"] = imp.loc[d.index, "map_base"]
    x["lact_i"] = np.exp(imp.loc[d.index, "log_lact"])
    return x


def glm_fit(formula, df, w=None):
    mod = smf.glm(formula, df, family=sm.families.Binomial(), var_weights=w)
    return mod.fit(cov_type="cluster", cov_kwds={"groups": df.subject_id.values})


def rubin(ests, vars_):
    ests, vars_ = np.asarray(ests, float), np.asarray(vars_, float)
    m = len(ests)
    qb, ub = ests.mean(), vars_.mean()
    b = ests.var(ddof=1) if m > 1 else 0.0
    t = ub + (1 + 1 / m) * b
    dfv = (m - 1) * (1 + ub / ((1 + 1 / m) * b)) ** 2 if b > 0 else 1e6
    q = stats.t.ppf(0.975, dfv)
    p = 2 * stats.t.sf(abs(qb / np.sqrt(t)), dfv)
    return qb, np.sqrt(t), qb - q * np.sqrt(t), qb + q * np.sqrt(t), p


def or_mi(d, imps, y, expo="vp", cov=COV_MI, terms=None, weights=None, with_imp=with_imp):
    """Odds ratios of the model terms, pooled over the imputations (logistic regression, clustered SE)."""
    terms = terms or [expo]
    acc = {t: ([], []) for t in terms}
    for imp in imps:
        x = with_imp(d, imp)
        w = None if weights is None else weights(x)
        r = glm_fit(f"{y}~{expo}+{cov}" if cov else f"{y}~{expo}", x, w)
        for t in terms:
            acc[t][0].append(r.params[t])
            acc[t][1].append(r.cov_params().loc[t, t])
    out = {}
    for t in terms:
        q, se, lo, hi, p = rubin(*acc[t])
        out[t] = dict(OR=np.exp(q), lo=np.exp(lo), hi=np.exp(hi), p=p)
    return out


def fmt(o, k="OR"):
    return "%.2f (%.2f–%.2f)" % (o[k], o["lo"], o["hi"])


# --- propensity score and ATT weights ------------------------------------------------
def ps_weights(x, trunc=0.99, cov=COV_MI):
    r = smf.glm(f"vp~{cov}", x, family=sm.families.Binomial()).fit()
    ps = r.predict(x).values.clip(1e-6, 1 - 1e-6)
    w = np.where(x.vp.values == 1, 1.0, ps / (1 - ps))
    ne = x.vp.values == 0
    cap = np.quantile(w[ne], trunc)
    return np.where(ne, np.minimum(w, cap), w), ps


# --- competing risks -----------------------------------------------------------------
def event_data(x, comp=("death", "icuout", "restart"), horizon=72.0, poly_col="onset", extra_censor=None):
    """1 = polyuria, 2 = death, 3 = ICU discharge, 4 = vasopressor restart, 0 = censored; ties go to polyuria."""
    on = x[poly_col].values.astype(float)
    times = [np.where(np.isnan(on), np.inf, on)]
    codes = [1]
    cand = {"death": x.death_t.values, "icuout": x.icuout_t.values, "restart": x.restart_t.values}
    for k, c in zip(["death", "icuout", "restart"], [2, 3, 4]):
        if k in comp:
            tt = np.asarray(cand[k], float)
            times.append(np.where(np.isnan(tt) | (tt < 0), np.inf, tt))
            codes.append(c)
    cens = np.full(len(x), horizon)
    if extra_censor is not None:
        ec = np.asarray(extra_censor, float)
        cens = np.minimum(cens, np.where(np.isnan(ec), np.inf, ec))
    times.append(cens)
    codes.append(0)
    M = np.vstack(times)
    idx = np.argmin(M, axis=0)
    return M[idx, np.arange(len(x))], np.array(codes)[idx]


def aj(T, Cc, w=None, grid=np.arange(0, 73), causes=(1, 2, 3, 4)):
    """Weighted Aalen-Johansen estimator (survival reduced by all causes; cumulative incidence of the requested causes)."""
    w = np.ones(len(T)) if w is None else np.asarray(w, float)
    o = np.argsort(T, kind="stable")
    T, Cc, w = T[o], Cc[o], w[o]
    ut = np.unique(T[Cc > 0])
    allc = np.unique(Cc[Cc > 0])
    Wc = np.concatenate([[0], np.cumsum(w)])
    S = 1.0
    cif = {c: 0.0 for c in allc}
    times, vals = [], {c: [] for c in causes}
    for t in ut:
        lo, hi = np.searchsorted(T, t, "left"), np.searchsorted(T, t, "right")
        nr = Wc[-1] - Wc[lo]
        if nr <= 0:
            break
        ww, cc = w[lo:hi], Cc[lo:hi]
        ds = 0.0
        for c in allc:
            dc = ww[cc == c].sum()
            cif[c] += S * dc / nr
            ds += dc
        S *= 1 - ds / nr
        times.append(t)
        for c in causes:
            vals[c].append(cif.get(c, 0.0))
    times = np.array(times)
    out = {}
    for c in causes:
        a = np.array(vals[c])
        out[c] = np.array([a[times <= g][-1] if (times <= g).any() else 0.0 for g in grid])
    out["nrisk_w"] = np.array([w[T >= g].sum() for g in grid])
    out["nrisk"] = np.array([(T >= g).sum() for g in grid])
    return out
