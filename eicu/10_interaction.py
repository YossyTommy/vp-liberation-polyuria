#!/usr/bin/env python3
"""
10_interaction.py - eICU-CRD v2.0, one additional analysis for manuscript v16 (post hoc,
exploratory; outcome, model and sensitivity definition were fixed before the model was fitted): is the link between polyuria and a rise in serum sodium stronger in
the VP group than in the NE group?

The analysis is that of mimic/12_interaction.py.

  population   sodium denominator of analysis B (08_early_late.py): primary population, every
               sodium of the 24 h before T0 < 150 mmol/L and >= 1 sodium within 72 h after T0
  outcome      sodium rise within 72 h after T0
               main        : rise >= 5 mmol/L from baseline or highest value >= 150 mmol/L
               sensitivity : rise >= 3 mmol/L from baseline or highest value >= 150 mmol/L
  model        logistic regression: outcome ~ group + polyuria within 24 h + group x polyuria
               + covariates of the v15 primary model (eicu_stats.COV); chained-equation
               imputation of the covariates with the outcome and exposure terms in the imputation
               model (eicu_stats.impute), hospital-clustered SE, Rubin's rules with a normal
               reference distribution (conventions of eicu_stats.fit_or)
  reported     multiplicative scale: odds ratio of the interaction term and the four conditional
                 odds ratios it is built from
               additive scale: standardised proportions of the four group x polyuria cells, the
                 difference polyuria - no polyuria within each group and the difference between
                 these two differences; SE by the delta method from the covariance matrix of
                 the coefficients (covariates treated as fixed), Rubin's rules
  SE types     every estimate is written twice: with the hospital-clustered covariance
               (convention of the v15 analyses) and with the model-based covariance of the same
               fit. The second was added after the first run: the hospital-clustered SE of the
               interaction term was about one third smaller than the model-based SE, because the
               VP stays with polyuria come from few hospitals (their number is written to the
               output as n_hospitals_vp_polyuria). A clustered SE below the model-based SE with
               few, unbalanced clusters is treated as too small, and the wider interval is the
               one to be used for interpretation.

Nothing else is estimated here. Definitions are taken unchanged from v15 and from
08_early_late.py; the stay table written by script 08 is read, not rebuilt. The script first
recomputes the v15 key values and the counts of analysis B and stops when any of them differs.

Outputs  OUT_DIR/eicu_10/reproduction_checks.csv, interaction_cells.csv, interaction_estimates.csv
         (aggregates only; cells < 10 shown as "<10"; no row-level output)
Run:     python eicu/10_interaction.py   (after eicu/08_early_late.py)
Optional environment variable: MICE_M (number of imputations, default 10).
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eicu_common import COHORT_PRIMARY, cohort_path, out_dir  # noqa: E402
import config  # noqa: E402  (importable once eicu_common has put the repository root on sys.path)
from eicu_stats import COV, MICE_M, impute  # noqa: E402
import early_late_helpers as H  # noqa: E402

warnings.filterwarnings("ignore")
T_START = time.time()
SID = "patientunitstayid"
SE_TYPES = ["clustered on hospital (v15 convention)", "model-based"]
CELLS = [(1, 1), (1, 0), (0, 1), (0, 0)]                 # (vp, polyuria)
CELL_LAB = {(1, 1): "VP, polyuria", (1, 0): "VP, no polyuria", (0, 1): "NE, polyuria", (0, 0): "NE, no polyuria"}
LOGOR = {"interaction: ratio of polyuria ORs, VP / NE": {"vp_poly": 1},
         "polyuria vs no polyuria, NE group": {"poly": 1},
         "polyuria vs no polyuria, VP group": {"poly": 1, "vp_poly": 1},
         "VP vs NE, stays without polyuria": {"vp": 1},
         "VP vs NE, stays with polyuria": {"vp": 1, "vp_poly": 1}}
ADD = {"standardised proportion: VP, polyuria": [1, 0, 0, 0],
       "standardised proportion: VP, no polyuria": [0, 1, 0, 0],
       "standardised proportion: NE, polyuria": [0, 0, 1, 0],
       "standardised proportion: NE, no polyuria": [0, 0, 0, 1],
       "difference polyuria - no polyuria, VP group": [1, -1, 0, 0],
       "difference polyuria - no polyuria, NE group": [0, 0, 1, -1],
       "difference in differences (VP - NE)": [1, -1, -1, 1]}


def say(*a):
    print("%5.0fs" % (time.time() - T_START), *a, flush=True)


def fit_one(s, y, xs):
    """One completed data set: log odds ratios, the four standardised proportions and the
    Jacobian of both with respect to the coefficients; covariance of the estimates for each
    SE type (hospital-clustered and model-based covariance of the same fit)."""
    keep = [c for c in xs if s[c].nunique() > 1]
    X0 = sm.add_constant(s[keep].astype(float))
    r = sm.Logit(s[y].astype(float), X0).fit(disp=0, cov_type="cluster", cov_kwds={"groups": s.hospitalid}, maxiter=200)
    r0 = sm.Logit(s[y].astype(float), X0).fit(disp=0, maxiter=200)
    names = list(X0.columns)
    b = r.params.values
    X0 = X0.values
    iv, ip, ii = names.index("vp"), names.index("poly"), names.index("vp_poly")
    est, jac = [], []
    for k, w in LOGOR.items():
        L = np.zeros(len(names))
        for n, c in w.items():
            L[names.index(n)] = c
        est.append(L @ b)
        jac.append(L)
    S, J = [], []
    for a, p in CELLS:
        X = X0.copy()                                   # covariates of every stay kept, group and polyuria set
        X[:, iv], X[:, ip], X[:, ii] = a, p, a * p
        pr = expit(X @ b)
        S.append(pr.mean())
        J.append(((pr * (1 - pr))[:, None] * X).mean(axis=0))
    S, J = np.array(S), np.vstack(J)
    for k, w in ADD.items():
        w = np.asarray(w, float)
        est.append(w @ S)
        jac.append(w @ J)
    G = np.vstack(jac)
    return np.array(est), {SE_TYPES[0]: G @ r.cov_params().values @ G.T, SE_TYPES[1]: G @ r0.cov_params().values @ G.T}


def estimate(d, y, label):
    d = d.dropna(subset=[y, "hospitalid"]).copy()
    sets = impute(d, list(COV), [y, "vp", "poly", "vp_poly"], MICE_M)
    Ps, Us = [], {t: [] for t in SE_TYPES}
    for s in sets:
        e, U = fit_one(s, y, ["vp", "poly", "vp_poly"] + COV)
        Ps.append(e)
        for t in SE_TYPES:
            Us[t].append(U[t])
    Ps = np.asarray(Ps)
    m = len(Ps)
    q = Ps.mean(axis=0)
    rows = []
    for t in SE_TYPES:
        T = sum(Us[t]) / m + ((1 + 1 / m) * np.cov(Ps.T, ddof=1) if m > 1 else 0)
        se = np.sqrt(np.diag(T))
        for j, k in enumerate(list(LOGOR) + list(ADD)):
            lo, hi, p = q[j] - 1.96 * se[j], q[j] + 1.96 * se[j], 2 * norm.sf(abs(q[j] / se[j]))
            if k in LOGOR:
                rows.append(dict(definition=label, se_type=t, scale="multiplicative (odds ratio)", term=k, est=np.exp(q[j]), lo=np.exp(lo),
                                 hi=np.exp(hi), p=p, est_raw=q[j], se_raw=se[j], raw_unit="log odds ratio",
                                 text="%.2f (%.2f–%.2f)" % (np.exp(q[j]), np.exp(lo), np.exp(hi)), n=len(d)))
            else:
                prop = k.startswith("standardised proportion")
                rows.append(dict(definition=label, se_type=t,
                                 scale="additive (standardised proportion, %)" if prop else "additive (percentage points)",
                                 term=k, est=100 * q[j], lo=100 * lo, hi=100 * hi, p=np.nan if prop else p, est_raw=q[j], se_raw=se[j],
                                 raw_unit="proportion",
                                 text=("%.1f%% (%.1f–%.1f)" if prop else "%.1f (%.1f to %.1f)") % (100 * q[j], 100 * lo, 100 * hi), n=len(d)))
    vpp = d[(d.vp == 1) & (d.poly == 1)]
    for r in rows:
        r["n_hospitals"] = int(d.hospitalid.nunique())
        r["n_hospitals_vp_polyuria"] = int(vpp.hospitalid.nunique())
        r["method"] = "logistic regression with group x polyuria term, MICE m=%d, delta method for standardised proportions" % len(sets)
    return rows


def main():
    out = out_dir("10")
    wk = config.work_dir("eicu", "early_late")
    stored = pd.read_parquet(cohort_path(COHORT_PRIMARY))
    ST = pd.read_parquet(os.path.join(wk, "eicu_early_late_stays.parquet"))      # written by 08_early_late.py

    # ---- 0. reproduction of v15 values and of analysis B ---------------------------------------
    PR = stored[stored.primary == 1]
    S = PR[(PR.na_pre_max < 150) & (PR.na_post72_n >= 1)]
    rep = []

    def chk(item, val, exp):
        rep.append(dict(check=item, value=val, expected=exp, match=val == exp))
    chk("VP stays, primary", str(int((PR.grp == "VP").sum())), "370")
    chk("NE stays, primary", "{:,}".format(int((PR.grp == "NE").sum())), "3,422")
    for g, e in (("VP", "21.9"), ("NE", "16.7")):
        chk(f"24-h polyuria % {g}", "%.1f" % (100 * PR.loc[PR.grp == g, "polyuria_24h"].mean()), e)
    for g, e in (("VP", "14.7"), ("NE", "5.6")):
        chk(f"Na>=150 within 72 h % {g}", "%.1f" % (100 * (S.loc[S.grp == g, "na_post72_max"] >= 150).mean()), e)
    chk("stays of script 08 = primary population", str(set(ST[SID]) == set(PR[SID])), "True")
    B = ST[ST.na_set == 1].copy().reset_index(drop=True)
    chk("sodium denominator VP / NE (analysis B)", "%d / %s" % ((B.grp == "VP").sum(), "{:,}".format(int((B.grp == "NE").sum()))), "326 / 3,089")
    rise = B.na_post72_max - B.na_pre
    B["na_up"] = ((rise >= 5) | (B.na_post72_max >= 150)).astype(int)
    B["na_up3"] = ((rise >= 3) | (B.na_post72_max >= 150)).astype(int)
    B["poly"] = B.polyuria_24h.astype(int)
    B["vp_poly"] = B.vp * B.poly
    cat = np.where(B.poly == 0, 0, np.where(B.na_up == 1, 2, 1))
    for g, e in (("VP", "255/27/44"), ("NE", "2573/340/176")):
        chk(f"analysis B categories 0/1/2, {g}", "/".join(str(int(((B.grp == g) & (cat == c)).sum())) for c in (0, 1, 2)), e)
    REP = pd.DataFrame(rep)
    REP.to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
    print(REP.to_string(index=False))
    if not REP.match.all():
        say("REPRODUCTION FAILED - analysis stopped")
        sys.exit(1)

    # ---- 1. crude cells ------------------------------------------------------------------------
    crow = []
    for dn, col in (("main: rise >= 5 or peak >= 150", "na_up"), ("sens: rise >= 3 or peak >= 150", "na_up3")):
        for a, p in CELLS:
            s = B[(B.vp == a) & (B.poly == p)]
            txt, pct, lo, hi = H.fmt_prop(int(s[col].sum()), len(s))
            crow.append(dict(definition=dn, cell=CELL_LAB[(a, p)], group="VP" if a else "NE", polyuria24=p,
                             n=len(s) if len(s) >= 10 else "<10", events=int(s[col].sum()) if not H.small(int(s[col].sum()), len(s)) else "<10",
                             text=txt, pct=pct, lo=lo, hi=hi))
    pd.DataFrame(crow).to_csv(os.path.join(out, "interaction_cells.csv"), index=False)

    # ---- 2. models -----------------------------------------------------------------------------
    rows = []
    for dn, col in (("main: rise >= 5 or peak >= 150", "na_up"), ("sens: rise >= 3 or peak >= 150", "na_up3")):
        rows += estimate(B, col, dn)
        say("estimated", dn)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(out, "interaction_estimates.csv"), index=False)
    print(E[["definition", "se_type", "term", "text", "p"]].to_string(index=False))
    say("finished")


if __name__ == "__main__":
    main()
