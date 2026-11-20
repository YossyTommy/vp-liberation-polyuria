#!/usr/bin/env python3
"""
12_interaction.py - MIMIC-IV v3.1, one pre-specified additional analysis for manuscript v16
(post hoc, exploratory): is the link between polyuria and a rise in serum sodium stronger in
the VP group than in the NE group?

  population   sodium denominator of analysis B (10_early_late.py): primary population, every
               sodium of the 24 h before T0 < 150 mmol/L and >= 1 sodium within 72 h after T0
  outcome      sodium rise within 72 h after T0
               main        : rise >= 5 mmol/L from baseline or highest value >= 150 mmol/L
               sensitivity : rise >= 3 mmol/L from baseline or highest value >= 150 mmol/L
  model        logistic regression: outcome ~ group + polyuria within 24 h + group x polyuria
               + covariates of the v15 primary model (mimic_models.COV_MI); multiple imputation
               (mimic_models.impute), SE clustered on subject_id, Rubin's rules (mimic_models.rubin)
  reported     multiplicative scale: odds ratio of the interaction term (ratio of the polyuria
                 odds ratios, VP / NE) and the four conditional odds ratios it is built from
               additive scale: standardised proportions of the four group x polyuria cells
                 (every stay of the analysis population is given each of the four combinations
                 in turn, its own covariates kept, and the predicted probabilities are averaged),
                 the difference polyuria - no polyuria within each group and the difference
                 between these two differences; SE by the delta method from the covariance
                 matrix of the coefficients (covariates treated as fixed), Rubin's rules
  SE types     every estimate is written twice: with the covariance clustered on subject_id
               (convention of the v15 analyses) and with the model-based covariance of the same
               fit. The second was added because, in eICU-CRD, the hospital-clustered SE of the
               interaction term was about one third smaller than the model-based SE (VP stays
               with polyuria come from few hospitals); the two are reported side by side in both
               databases so that the wider interval can be used.

Nothing else is estimated here. Definitions (cohort, groups, T0, polyuria, covariates,
imputation, sodium handling) are taken unchanged from v15 and from 10_early_late.py; the
stay table written by script 10 is read, not rebuilt.

The script first recomputes the v15 key values and the counts of analysis B and stops when
any of them differs.

Outputs  OUT_DIR/mimic_12/reproduction_checks.csv, interaction_cells.csv, interaction_estimates.csv
         (aggregates only; cells < 10 shown as "<10"; no row-level output)
Run:     python mimic/12_interaction.py   (after mimic/10_early_late.py)
Optional environment variable: M_IMP (number of imputations, default 10).
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mimic_common as mc  # noqa: E402
import mimic_models as mm  # noqa: E402
from mimic_common import config  # noqa: E402
import early_late_helpers as H  # noqa: E402  (repository root; on sys.path once mimic_common is imported)

warnings.filterwarnings("ignore")
T_START = time.time()
SE_TYPES = ["clustered on patient (v15 convention)", "model-based"]
CELLS = [(1, 1), (1, 0), (0, 1), (0, 0)]                 # (vp, polyuria)
CELL_LAB = {(1, 1): "VP, polyuria", (1, 0): "VP, no polyuria", (0, 1): "NE, polyuria", (0, 0): "NE, no polyuria"}
# log odds ratios (linear combinations of the coefficients)
LOGOR = {"interaction: ratio of polyuria ORs, VP / NE": {"vp:poly24": 1},
         "polyuria vs no polyuria, NE group": {"poly24": 1},
         "polyuria vs no polyuria, VP group": {"poly24": 1, "vp:poly24": 1},
         "VP vs NE, stays without polyuria": {"vp": 1},
         "VP vs NE, stays with polyuria": {"vp": 1, "vp:poly24": 1}}
# linear combinations of the four standardised proportions (order of CELLS)
ADD = {"standardised proportion: VP, polyuria": [1, 0, 0, 0],
       "standardised proportion: VP, no polyuria": [0, 1, 0, 0],
       "standardised proportion: NE, polyuria": [0, 0, 1, 0],
       "standardised proportion: NE, no polyuria": [0, 0, 0, 1],
       "difference polyuria - no polyuria, VP group": [1, -1, 0, 0],
       "difference polyuria - no polyuria, NE group": [0, 0, 1, -1],
       "difference in differences (VP - NE)": [1, -1, -1, 1]}


def say(*a):
    print("%5.0fs" % (time.time() - T_START), *a, flush=True)


def fit_one(x, y):
    """One completed data set: coefficients, the covariance matrix of each SE type, the four
    standardised proportions and their Jacobian with respect to the coefficients."""
    f = f"{y} ~ vp * poly24 + {mm.COV_MI}"
    r = mm.glm_fit(f, x)                                                    # SE clustered on subject_id (v15)
    r0 = smf.glm(f, x, family=sm.families.Binomial()).fit()                 # same fit, model-based covariance
    b = r.params
    names = list(r.model.exog_names)
    X0 = np.asarray(r.model.exog, float)
    iv, ip, ii = names.index("vp"), names.index("poly24"), names.index("vp:poly24")
    S, J = [], []
    for a, p in CELLS:
        X = X0.copy()                                   # covariates of every stay kept, group and polyuria set
        X[:, iv], X[:, ip], X[:, ii] = a, p, a * p
        pr = expit(X @ b.values)
        S.append(pr.mean())
        J.append(((pr * (1 - pr))[:, None] * X).mean(axis=0))
    return b, {SE_TYPES[0]: r.cov_params(), SE_TYPES[1]: r0.cov_params()}, np.array(S), np.vstack(J)


def estimate(d, imps, y, label):
    d = d.dropna(subset=[y]).copy()
    acc = {(t, k): ([], []) for t in SE_TYPES for k in list(LOGOR) + list(ADD)}
    for imp in imps:
        b, Vs, S, J = fit_one(mm.with_imp(d, imp), y)
        for t, V in Vs.items():
            for k, w in LOGOR.items():
                L = pd.Series(0.0, index=b.index)
                for n, c in w.items():
                    L[n] = c
                acc[(t, k)][0].append(float(L.values @ b.values))
                acc[(t, k)][1].append(float(L.values @ V.values @ L.values))
            VS = J @ V.values @ J.T
            for k, w in ADD.items():
                w = np.asarray(w, float)
                acc[(t, k)][0].append(float(w @ S))
                acc[(t, k)][1].append(float(w @ VS @ w))
    rows = []
    for t in SE_TYPES:
        for k in LOGOR:
            q, se, lo, hi, p = mm.rubin(*acc[(t, k)])
            rows.append(dict(definition=label, se_type=t, scale="multiplicative (odds ratio)", term=k, est=np.exp(q), lo=np.exp(lo),
                             hi=np.exp(hi), p=p, est_raw=q, se_raw=se, raw_unit="log odds ratio",
                             text="%.2f (%.2f–%.2f)" % (np.exp(q), np.exp(lo), np.exp(hi)), n=len(d)))
        for k in ADD:
            q, se, lo, hi, p = mm.rubin(*acc[(t, k)])
            prop = k.startswith("standardised proportion")
            rows.append(dict(definition=label, se_type=t,
                             scale="additive (standardised proportion, %)" if prop else "additive (percentage points)",
                             term=k, est=100 * q, lo=100 * lo, hi=100 * hi, p=np.nan if prop else p, est_raw=q, se_raw=se,
                             raw_unit="proportion",
                             text=("%.1f%% (%.1f–%.1f)" if prop else "%.1f (%.1f to %.1f)") % (100 * q, 100 * lo, 100 * hi), n=len(d)))
    for r in rows:
        r["n_patients"] = d.subject_id.nunique()
        r["method"] = "logistic regression with group x polyuria term, MICE m=%d, delta method for standardised proportions" % len(imps)
    return rows


def main():
    out = config.out_dir("mimic_12")
    wk = config.work_dir("mimic", "early_late")
    D0 = pd.read_parquet(mc.cohort_file(mc.PRIMARY_TABLE))
    ST = pd.read_parquet(os.path.join(wk, "mimic_early_late_stays.parquet"))     # written by 10_early_late.py

    # ---- 0. reproduction of v15 values and of analysis B ---------------------------------------
    P0 = D0[D0.pop_primary == 1]
    S = P0[(P0.na_max_pre24 < 150) & (P0.na_n72 >= 1)]
    rep = []

    def chk(item, val, exp):
        rep.append(dict(check=item, value=val, expected=exp, match=val == exp))
    chk("VP stays, primary", str(int((P0.grp == "VP").sum())), "743")
    chk("NE stays, primary", "{:,}".format(int((P0.grp == "NE").sum())), "4,667")
    for g, e in (("VP", "30.1"), ("NE", "19.4")):
        chk(f"24-h polyuria % {g}", "%.1f" % (100 * P0.loc[P0.grp == g, "poly24"].mean()), e)
    for g, e in (("VP", "9.5"), ("NE", "4.8")):
        chk(f"Na>=150 within 72 h % {g} (harmonized)", "%.1f" % (100 * S.loc[S.grp == g, "na150_72"].mean()), e)
    chk("stays of script 10 = primary population", str(bool(ST.index.equals(P0.index))), "True")
    B = ST[ST.na_set == 1].copy()
    chk("sodium denominator VP / NE (analysis B)", "%d / %s" % ((B.grp == "VP").sum(), "{:,}".format(int((B.grp == "NE").sum()))), "698 / 4,212")
    rise = B.na_peak_72 - B.na_pre
    B["na_up"] = ((rise >= 5) | (B.na_peak_72 >= 150)).astype(int)
    B["na_up3"] = ((rise >= 3) | (B.na_peak_72 >= 150)).astype(int)
    cat = np.where(B.poly24 == 0, 0, np.where(B.na_up == 1, 2, 1))
    for g, e in (("VP", "486/86/126"), ("NE", "3394/531/287")):
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
            s = B[(B.vp == a) & (B.poly24 == p)]
            txt, pct, lo, hi = H.fmt_prop(int(s[col].sum()), len(s))
            crow.append(dict(definition=dn, cell=CELL_LAB[(a, p)], group="VP" if a else "NE", polyuria24=p,
                             n=len(s) if len(s) >= 10 else "<10", events=int(s[col].sum()) if not H.small(int(s[col].sum()), len(s)) else "<10",
                             text=txt, pct=pct, lo=lo, hi=hi))
    pd.DataFrame(crow).to_csv(os.path.join(out, "interaction_cells.csv"), index=False)

    # ---- 2. models -----------------------------------------------------------------------------
    IMPS = mm.impute(D0)
    say("imputed")
    rows = []
    for dn, col in (("main: rise >= 5 or peak >= 150", "na_up"), ("sens: rise >= 3 or peak >= 150", "na_up3")):
        rows += estimate(B, IMPS, col, dn)
        say("estimated", dn)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(out, "interaction_estimates.csv"), index=False)
    print(E[["definition", "se_type", "term", "text", "p"]].to_string(index=False))
    say("finished")


if __name__ == "__main__":
    main()
