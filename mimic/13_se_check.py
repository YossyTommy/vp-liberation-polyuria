#!/usr/bin/env python3
"""
13_se_check.py - MIMIC-IV v3.1, manuscript v17: standard errors of the post hoc sodium-rise models.

No new analysis: the three-category model of 10_early_late.py (analysis B; main definition and its
two sensitivity definitions) and the five-category model of 11_early_late_extra.py (E1; main
definition and the >= 3 mmol/L definition) are fitted again unchanged. For every estimate the
script writes two intervals from the same fit:
  clustered     covariance clustered on subject_id (convention of the v15 analyses; this is the
                interval reported in v16)
  model-based   model-based covariance
so that pooled/05_se_check.py can apply the rule already used for the interaction analysis
(12_interaction.py): in each database, the wider of the two intervals is reported.

The script stops before writing any estimate when
  - a v15 key value or a count of analysis B differs, or
  - a point estimate (or its clustered interval) differs from the output of scripts 10 / 11.

Definitions (cohort, groups, T0, polyuria, covariates, imputation, sodium handling) are taken
unchanged from v15 and from scripts 10 and 11; the stay table written by script 10 is read.
The script also writes the number of stays with sodium >= 150 mmol/L within 72 h per group in
the sodium analysis population (the counts of the v15 secondary outcome; used for a figure).

Outputs  OUT_DIR/mimic_13/reproduction_checks.csv, se_check_estimates.csv, na150_counts.csv
         (aggregates only; no row-level output)
Run:     python mimic/13_se_check.py   (after mimic/10_early_late.py and mimic/11_early_late_extra.py)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mimic_common as mc  # noqa: E402
import mimic_models as mm  # noqa: E402
from mimic_common import config  # noqa: E402
import early_late_helpers as H  # noqa: E402

warnings.filterwarnings("ignore")
T_START = time.time()
SE_TYPES = ["clustered", "model-based"]
C3 = {"polyuria without Na rise": {1: 1}, "polyuria with Na rise": {2: 1}}
C5 = {"early with Na rise": {2: 1}, "early without Na rise": {1: 1}, "late with Na rise": {4: 1}, "late without Na rise": {3: 1},
      "ratio: early with / late with": {2: 1, 4: -1}, "ratio: early with / early without": {2: 1, 1: -1},
      "ratio: late with / late without": {4: 1, 3: -1}, "ratio: early without / late without": {1: 1, 3: -1}}


def say(*a):
    print("%5.0fs" % (time.time() - T_START), *a, flush=True)


def fit_both(x, ycol, cats):
    """VP coefficients of the non-reference categories and their covariance, for both SE types."""
    f = f"{ycol} ~ vp + {mm.COV_MI}"
    out = {}
    for t in SE_TYPES:
        kw = dict(cov_type="cluster", cov_kwds={"groups": x.subject_id.values}) if t == "clustered" else {}
        r = smf.mnlogit(f, x).fit(disp=0, maxiter=300, method="newton", **kw)
        V = r.cov_params()
        k = [i for i in V.index if i[1] == "vp"]
        out[t] = (pd.Series(r.params.loc["vp"].values, index=cats), pd.DataFrame(V.loc[k, k].values, index=cats, columns=cats))
    return out


def estimate(d, imps, ycol, cats, combos, model, label):
    acc = {(t, k): ([], []) for t in SE_TYPES for k in combos}
    for imp in imps:
        res = fit_both(mm.with_imp(d, imp), ycol, cats)
        for t, (b, V) in res.items():
            for k, w in combos.items():
                L = pd.Series(0.0, index=b.index)
                for n, c in w.items():
                    L[n] = c
                acc[(t, k)][0].append(float(L.values @ b.values))
                acc[(t, k)][1].append(float(L.values @ V.values @ L.values))
    rows = []
    for t in SE_TYPES:
        for k in combos:
            q, se, lo, hi, p = mm.rubin(*acc[(t, k)])
            rows.append(dict(model=model, definition=label, term=k, se_type=t, OR=np.exp(q), lo=np.exp(lo), hi=np.exp(hi), p=p,
                             log_or=q, se=se, text="%.2f (%.2f–%.2f)" % (np.exp(q), np.exp(lo), np.exp(hi)), n=len(d)))
    return rows


def main():
    out = config.out_dir("mimic_13")
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
    defs3 = {"main: rise >= 5 or peak >= 150": (rise >= 5) | (B.na_peak_72 >= 150),
             "sens: rise >= 3 or peak >= 150": (rise >= 3) | (B.na_peak_72 >= 150),
             "sens: peak >= 150 only": B.na_peak_72 >= 150}
    cat = np.where(B.poly24 == 0, 0, np.where(defs3["main: rise >= 5 or peak >= 150"], 2, 1))
    for g, e in (("VP", "486/86/126"), ("NE", "3394/531/287")):
        chk(f"analysis B categories 0/1/2, {g}", "/".join(str(int(((B.grp == g) & (cat == c)).sum())) for c in (0, 1, 2)), e)
    REP = pd.DataFrame(rep)
    if not REP.match.all():
        REP.to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
        print(REP.to_string(index=False))
        say("REPRODUCTION FAILED - stopped")
        sys.exit(1)

    # ---- 1. models (both SE types) -------------------------------------------------------------
    IMPS = mm.impute(D0)
    say("imputed")
    rows = []
    for dn, m in defs3.items():
        B["cat"] = np.where(B.poly24 == 0, 0, np.where(m, 2, 1))
        rows += estimate(B, IMPS, "cat", [1, 2], C3, "three-category", dn)
        say("three-category", dn)
    B["tim3"] = B.timing.map({"none": 0, "early": 1, "late": 2}).astype(int)
    for dn in ("main: rise >= 5 or peak >= 150", "sens: rise >= 3 or peak >= 150"):
        up = defs3[dn]
        B["cat5"] = np.select([B.tim3 == 0, (B.tim3 == 1) & ~up, (B.tim3 == 1) & up, (B.tim3 == 2) & ~up], [0, 1, 2, 3], 4)
        rows += estimate(B, IMPS, "cat5", [1, 2, 3, 4], C5, "five-category", dn)
        say("five-category", dn)
    E = pd.DataFrame(rows)

    # ---- 2. point estimates and clustered intervals must equal the output of scripts 10 and 11 --
    M10 = pd.read_csv(os.path.join(config.OUT_DIR, "mimic_10", "B_multinomial.csv"))
    M11 = pd.read_csv(os.path.join(config.OUT_DIR, "mimic_11", "E1_multinomial5.csv"))
    worst = 0.0
    for _, r in E[E.se_type == "clustered"].iterrows():
        if r.model == "three-category":
            c = {"polyuria without Na rise": 1, "polyuria with Na rise": 2}[r.term]
            ref = M10[(M10.analysis == r.definition) & (M10.category == c)].iloc[0]
        else:
            ref = M11[(M11.analysis == r.definition) & (M11.term == r.term)].iloc[0]
        d = max(abs(np.log(r.OR / ref.OR)), abs(np.log(r.lo / ref.lo)), abs(np.log(r.hi / ref.hi)))
        worst = max(worst, d)
    mb = E[E.se_type == "model-based"].set_index(["model", "definition", "term"]).OR
    cl = E[E.se_type == "clustered"].set_index(["model", "definition", "term"]).OR
    same = bool(np.allclose(mb.values, cl.loc[mb.index].values, rtol=1e-9))
    chk("point estimates and clustered intervals equal scripts 10 / 11 (largest difference on the log scale < 1e-6)", str(bool(worst < 1e-6)), "True")
    chk("point estimates identical for the two SE types", str(same), "True")
    REP = pd.DataFrame(rep)
    REP.to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
    print(REP.to_string(index=False))
    say("largest difference from scripts 10 / 11 (log scale): %.2e" % worst)
    if not REP.match.all():
        say("POINT ESTIMATES DIFFER FROM v16 - stopped, no estimates written")
        sys.exit(1)
    E.to_csv(os.path.join(out, "se_check_estimates.csv"), index=False)

    # ---- 3. counts of the v15 secondary outcome in the sodium analysis population ---------------
    crow = []
    for g in ("VP", "NE"):
        s = S[S.grp == g]
        k = int(s.na150_72.sum())
        txt, pct, lo, hi = H.fmt_prop(k, len(s))
        crow.append(dict(group=g, n=len(s), events=k if not H.small(k, len(s)) else "<10", text=txt, pct=pct, lo=lo, hi=hi))
    pd.DataFrame(crow).to_csv(os.path.join(out, "na150_counts.csv"), index=False)
    print(E[["model", "definition", "term", "se_type", "text"]].to_string(index=False))
    say("finished")


if __name__ == "__main__":
    main()
