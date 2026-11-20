#!/usr/bin/env python3
"""
11_se_check.py - eICU-CRD v2.0, manuscript v17: standard errors of the post hoc sodium-rise models.

No new analysis: the three-category model of 08_early_late.py (analysis B; main definition and
its two sensitivity definitions) and the five-category model of 09_early_late_extra.py (E1; main
definition and the >= 3 mmol/L definition) are fitted again unchanged (same imputation calls). For
every estimate the script writes two intervals from the same fit:
  clustered     hospital-clustered covariance (convention of the v15 analyses; reported in v16)
  model-based   model-based covariance
so that pooled/05_se_check.py can apply the rule already used for the interaction analysis
(10_interaction.py): in each database, the wider of the two intervals is reported. In eICU-CRD
the vasopressin stays with polyuria come from few hospitals, and a clustered SE below the
model-based SE is treated as too small.

The script stops before writing any estimate when a v15 key value or a count of analysis B
differs, or when a point estimate (or its clustered interval) differs from the output of
scripts 08 / 09. It also writes the number of stays with sodium >= 150 mmol/L within 72 h per
group in the sodium analysis population (counts of the v15 secondary outcome; used for a figure).

Outputs  OUT_DIR/eicu_11/reproduction_checks.csv, se_check_estimates.csv, na150_counts.csv
         (aggregates only; no row-level output)
Run:     python eicu/11_se_check.py   (after eicu/08_early_late.py and eicu/09_early_late_extra.py)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eicu_common import COHORT_PRIMARY, cohort_path, out_dir  # noqa: E402
import config  # noqa: E402
from eicu_stats import COV, MICE_M, impute  # noqa: E402
import early_late_helpers as H  # noqa: E402

warnings.filterwarnings("ignore")
T_START = time.time()
SID = "patientunitstayid"
SE_TYPES = ["clustered", "model-based"]
C3 = {"polyuria without Na rise": {1: 1}, "polyuria with Na rise": {2: 1}}
C5 = {"early with Na rise": {2: 1}, "early without Na rise": {1: 1}, "late with Na rise": {4: 1}, "late without Na rise": {3: 1},
      "ratio: early with / late with": {2: 1, 4: -1}, "ratio: early with / early without": {2: 1, 1: -1},
      "ratio: late with / late without": {4: 1, 3: -1}, "ratio: early without / late without": {1: 1, 3: -1}}


def say(*a):
    print("%5.0fs" % (time.time() - T_START), *a, flush=True)


def fit_both(s, ycol, cats):
    xs = ["vp"] + COV
    keep = [c for c in xs if s[c].nunique() > 1]
    X = sm.add_constant(s[keep].astype(float))
    out = {}
    for t in SE_TYPES:
        kw = dict(cov_type="cluster", cov_kwds={"groups": s.hospitalid}) if t == "clustered" else {}
        r = sm.MNLogit(s[ycol].astype(int), X).fit(disp=0, maxiter=300, method="newton", **kw)
        V = r.cov_params()
        k = [i for i in V.index if i[1] == "vp"]
        out[t] = (r.params.loc["vp"].values, V.loc[k, k].values)
    return out


def estimate(d, ycol, cats, model_vars, combos, model, label):
    sets = impute(d, list(COV), model_vars, MICE_M)
    Ps, Us = {t: [] for t in SE_TYPES}, {t: [] for t in SE_TYPES}
    for s in sets:
        res = fit_both(s, ycol, cats)
        for t in SE_TYPES:
            Ps[t].append(res[t][0])
            Us[t].append(res[t][1])
    rows = []
    for t in SE_TYPES:
        P = np.asarray(Ps[t])
        m = len(P)
        q = P.mean(axis=0)
        T = sum(Us[t]) / m + ((1 + 1 / m) * np.cov(P.T, ddof=1) if m > 1 else 0)
        for k, w in combos.items():
            L = np.zeros(len(cats))
            for n, c in w.items():
                L[cats.index(n)] = c
            est, se = float(L @ q), float(np.sqrt(L @ T @ L))
            lo, hi = est - 1.96 * se, est + 1.96 * se
            rows.append(dict(model=model, definition=label, term=k, se_type=t, OR=np.exp(est), lo=np.exp(lo), hi=np.exp(hi),
                             p=2 * norm.sf(abs(est / se)), log_or=est, se=se, text="%.2f (%.2f–%.2f)" % (np.exp(est), np.exp(lo), np.exp(hi)),
                             n=len(d)))
    return rows


def main():
    out = out_dir("11")
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
    P = ST.copy().reset_index(drop=True)
    P["tim3"] = P.timing.map({"none": 0, "early": 1, "late": 2}).astype(int)
    B = P[P.na_set == 1].copy()
    chk("sodium denominator VP / NE (analysis B)", "%d / %s" % ((B.grp == "VP").sum(), "{:,}".format(int((B.grp == "NE").sum()))), "326 / 3,089")
    rise = B.na_post72_max - B.na_pre
    defs3 = {"main: rise >= 5 or peak >= 150": (rise >= 5) | (B.na_post72_max >= 150),
             "sens: rise >= 3 or peak >= 150": (rise >= 3) | (B.na_post72_max >= 150),
             "sens: peak >= 150 only": B.na_post72_max >= 150}
    cat = np.where(B.polyuria_24h == 0, 0, np.where(defs3["main: rise >= 5 or peak >= 150"], 2, 1))
    for g, e in (("VP", "255/27/44"), ("NE", "2573/340/176")):
        chk(f"analysis B categories 0/1/2, {g}", "/".join(str(int(((B.grp == g) & (cat == c)).sum())) for c in (0, 1, 2)), e)
    REP = pd.DataFrame(rep)
    if not REP.match.all():
        REP.to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
        print(REP.to_string(index=False))
        say("REPRODUCTION FAILED - stopped")
        sys.exit(1)

    # ---- 1. models (both SE types; imputation calls of scripts 08 and 09) ------------------------
    rows = []
    for dn, m in defs3.items():
        d = B.copy()
        d["cat"] = np.where(d.polyuria_24h == 0, 0, np.where(m, 2, 1))
        d = d.dropna(subset=["cat", "hospitalid"]).copy()
        d["c1"] = (d["cat"] == 1).astype(int)
        d["c2"] = (d["cat"] == 2).astype(int)
        rows += estimate(d, "cat", [1, 2], ["c1", "c2", "vp"], C3, "three-category", dn)
        say("three-category", dn)
    for dn in ("main: rise >= 5 or peak >= 150", "sens: rise >= 3 or peak >= 150"):
        d = B.copy()
        up = defs3[dn]
        d["cat5"] = np.select([d.tim3 == 0, (d.tim3 == 1) & ~up, (d.tim3 == 1) & up, (d.tim3 == 2) & ~up], [0, 1, 2, 3], 4)
        for c in (1, 2, 3, 4):
            d[f"k{c}"] = (d.cat5 == c).astype(int)
        rows += estimate(d, "cat5", [1, 2, 3, 4], ["k1", "k2", "k3", "k4", "vp"], C5, "five-category", dn)
        say("five-category", dn)
    E = pd.DataFrame(rows)

    # ---- 2. point estimates and clustered intervals must equal the output of scripts 08 and 09 --
    M08 = pd.read_csv(os.path.join(config.OUT_DIR, "eicu_08", "B_multinomial.csv"))
    M09 = pd.read_csv(os.path.join(config.OUT_DIR, "eicu_09", "E1_multinomial5.csv"))
    worst = 0.0
    for _, r in E[E.se_type == "clustered"].iterrows():
        if r.model == "three-category":
            c = {"polyuria without Na rise": 1, "polyuria with Na rise": 2}[r.term]
            ref = M08[(M08.analysis == r.definition) & (M08.category == c)].iloc[0]
        else:
            ref = M09[(M09.analysis == r.definition) & (M09.term == r.term)].iloc[0]
        worst = max(worst, abs(np.log(r.OR / ref.OR)), abs(np.log(r.lo / ref.lo)), abs(np.log(r.hi / ref.hi)))
    mb = E[E.se_type == "model-based"].set_index(["model", "definition", "term"]).OR
    cl = E[E.se_type == "clustered"].set_index(["model", "definition", "term"]).OR
    chk("point estimates and clustered intervals equal scripts 08 / 09 (largest difference on the log scale < 1e-6)", str(bool(worst < 1e-6)), "True")
    chk("point estimates identical for the two SE types", str(bool(np.allclose(mb.values, cl.loc[mb.index].values, rtol=1e-9))), "True")
    REP = pd.DataFrame(rep)
    REP.to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
    print(REP.to_string(index=False))
    say("largest difference from scripts 08 / 09 (log scale): %.2e" % worst)
    if not REP.match.all():
        say("POINT ESTIMATES DIFFER FROM v16 - stopped, no estimates written")
        sys.exit(1)
    vpp = B[(B.vp == 1) & (B.polyuria_24h == 1)]
    E["n_hospitals"] = int(B.hospitalid.nunique())
    E["n_hospitals_vp_polyuria"] = int(vpp.hospitalid.nunique())
    E.to_csv(os.path.join(out, "se_check_estimates.csv"), index=False)

    # ---- 3. counts of the v15 secondary outcome in the sodium analysis population ---------------
    crow = []
    for g in ("VP", "NE"):
        s = S[S.grp == g]
        k = int((s.na_post72_max >= 150).sum())
        txt, pct, lo, hi = H.fmt_prop(k, len(s))
        crow.append(dict(group=g, n=len(s), events=k if not H.small(k, len(s)) else "<10", text=txt, pct=pct, lo=lo, hi=hi))
    pd.DataFrame(crow).to_csv(os.path.join(out, "na150_counts.csv"), index=False)
    print(E[["model", "definition", "term", "se_type", "text"]].to_string(index=False))
    say("finished")


if __name__ == "__main__":
    main()
