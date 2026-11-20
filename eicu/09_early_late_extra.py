#!/usr/bin/env python3
"""
09_early_late_extra.py - eICU-CRD v2.0, second set of exploratory analyses on early (0-8 h) and
late (>8-24 h) polyuria in the VP and NE-only groups (reported in the manuscript as post hoc exploratory analyses).

The five analyses (E1-E5) and their definitions are those of mimic/11_early_late_extra.py; they
were fixed before any of them was run and all of them are reported whatever their direction.

Definitions taken unchanged from v15 and from 08_early_late.py: cohort, groups, T0, polyuria and
its onset, covariates (eicu_stats.COV), imputation (eicu_stats.impute), hospital-clustered SE,
sodium filter (lab 'sodium', 100 < v < 200), sodium denominator and the sodium-rise definition.

eICU-specific points
  urine bins        eicu_cohort.urine_outcomes (bins -24..71 h relative to T0)
  vasopressin rate  positive infusionDrug records of the index episode with the unit "units/min"
                    and 0 < rate <= 0.2 (rule of eicu_cohort.build_usage_cohort); other units = missing
  urine labs        lab names 'urinary osmolality', 'urinary sodium', 'urinary specific gravity',
                    filtered once from lab.csv.gz into WORK_DIR/eicu/early_late/x_lab_urine.csv

Outputs  OUT_DIR/eicu_09/*.csv (aggregates only; cells < 10 shown as "<10")
         WORK_DIR/eicu/early_late/eicu_early_late_extra_stays.parquet, x_lab_urine.csv (row level)
Run:     python eicu/09_early_late_extra.py   (after eicu/08_early_late.py)
Optional environment variables: NBOOT_EL (bootstrap replicates, default 2000), MICE_M (10).
"""
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eicu_common import COHORT_PRIMARY, cohort_path, ex_dir, out_dir, raw  # noqa: E402
import config  # noqa: E402  (importable once eicu_common has put the repository root on sys.path)
from eicu_extract import load_infusions  # noqa: E402
from eicu_cohort import urine_and_intake_records, urine_outcomes  # noqa: E402
from eicu_stats import COV, MICE_M, impute  # noqa: E402
import early_late_helpers as H  # noqa: E402

warnings.filterwarnings("ignore")
G4 = ["VP early", "VP late", "NE early", "NE late"]
PAIRS = [("VP early", "VP late"), ("VP early", "NE early"), ("VP late", "NE late")]
URINE_LABS = {"urinary osmolality": "uosm", "urinary sodium": "una", "urinary specific gravity": "usg"}
T_START = time.time()
SID = "patientunitstayid"


def say(*a):
    print("%5.0fs" % (time.time() - T_START), *a, flush=True)


# ----------------------------------------------------------------------------------------
# estimation (conventions of eicu_stats.fit_or: chained-equation imputation of the covariates with
# the outcome and exposure terms in the imputation model, hospital-clustered SE, Rubin's rules
# with a normal reference distribution)
def mi_linear_combos(d, model_vars, fit, combos, covs=COV):
    """fit(s) -> (params Series, covariance DataFrame) for one completed data set."""
    sets = impute(d, list(covs), model_vars, MICE_M)
    names = None
    Ps, Us = [], []
    for s in sets:
        b, V = fit(s)
        names = list(b.index) if names is None else names
        Ps.append(b.reindex(names).fillna(0).values)
        Us.append(V.reindex(index=names, columns=names).fillna(0).values)
    Ps = np.asarray(Ps)
    m = len(Ps)
    q = Ps.mean(axis=0)
    T = sum(Us) / m + ((1 + 1 / m) * np.cov(Ps.T, ddof=1) if m > 1 else 0)
    out = {}
    for k, w in combos.items():
        L = np.zeros(len(names))
        for n, c in w.items():
            L[names.index(n)] = c
        est, se = float(L @ q), float(np.sqrt(L @ T @ L))
        out[k] = dict(est=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se, p=2 * norm.sf(abs(est / se)))
    return out


def design(s, xs):
    keep = [c for c in xs if s[c].nunique() > 1]
    return sm.add_constant(s[keep].astype(float))


def fit_ols(y, xs):
    def f(s):
        r = sm.OLS(s[y].astype(float), design(s, xs)).fit(cov_type="cluster", cov_kwds={"groups": s.hospitalid})
        return r.params, r.cov_params()
    return f


def fit_logit(y, xs):
    def f(s):
        r = sm.Logit(s[y].astype(float), design(s, xs)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": s.hospitalid}, maxiter=200)
        return r.params, r.cov_params()
    return f


def fit_mnlogit(y, xs, cats):
    def f(s):
        r = sm.MNLogit(s[y].astype(int), design(s, xs)).fit(disp=0, maxiter=300, method="newton", cov_type="cluster",
                                                            cov_kwds={"groups": s.hospitalid})
        V = r.cov_params()
        k = [i for i in V.index if i[1] == "vp"]
        return pd.Series(r.params.loc["vp"].values, index=cats), pd.DataFrame(V.loc[k, k].values, index=cats, columns=cats)
    return f


def or_rows(res, label, extra=None):
    return [dict(analysis=label, term=k, OR=np.exp(r["est"]), lo=np.exp(r["lo"]), hi=np.exp(r["hi"]), p=r["p"],
                 log_or=r["est"], se=r["se"], **(extra or {})) for k, r in res.items()]


def cell(n):
    return n if n >= 10 else "<10"


def urine_labs():
    """Urine osmolality, sodium and specific gravity of the stays of interest (cached below WORK_DIR)."""
    f = os.path.join(config.work_dir("eicu", "early_late"), "x_lab_urine.csv")
    if not os.path.exists(f):
        say("extracting urine laboratory values from lab.csv.gz (one pass)")
        idf = os.path.join(ex_dir(), "ids.txt")
        cond = "||".join('$5=="%s"' % k for k in URINE_LABS)
        cmd = (f"gunzip -c '{raw('lab.csv.gz')}' | LC_ALL=C awk -v idf='{idf}' -F',' "
               "'BEGIN{while((getline l<idf)>0)k[l]=1} NR==1{print;next} ($2 in k) && (" + cond + "){print}'" + f" > '{f}.tmp' && mv '{f}.tmp' '{f}'")
        subprocess.run(cmd, shell=True, check=True)
    x = pd.read_csv(f, low_memory=False, usecols=[SID, "labresultoffset", "labname", "labresult"])
    x["v"] = pd.to_numeric(x.labresult, errors="coerce")
    x["kind"] = x.labname.astype(str).str.lower().str.strip().map(URINE_LABS)
    return x.dropna(subset=["v", "kind"])


# ----------------------------------------------------------------------------------------
def main():
    out = out_dir("09")
    wk = config.work_dir("eicu", "early_late")
    stored = pd.read_parquet(cohort_path(COHORT_PRIMARY))
    ST = pd.read_parquet(os.path.join(wk, "eicu_early_late_stays.parquet"))      # written by 08_early_late.py

    # ---- 0. reproduction of v15 values and of the groups of script 08 ------------------------
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
    chk("groups of script 08 (VP early / VP late / NE early / NE late)",
        "/".join(str(int((ST.grp4 == g).sum())) for g in G4), "46/35/239/332")
    REP = pd.DataFrame(rep)
    REP.to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
    print(REP.to_string(index=False))
    if not REP.match.all():
        say("REPRODUCTION FAILED - analysis stopped")
        sys.exit(1)

    P = ST.copy().reset_index(drop=True)
    P["tim3"] = P.timing.map({"none": 0, "early": 1, "late": 2}).astype(int)
    rise = P.na_post72_max - P.na_pre
    P["na_up"] = ((rise >= 5) | (P.na_post72_max >= 150)).astype(float).where(P.na_set == 1)
    P["na_up3"] = ((rise >= 3) | (P.na_post72_max >= 150)).astype(float).where(P.na_set == 1)
    P["death"] = P.hosp_death.astype(int)
    P["icu_days_post"] = (P.obs_after_t0_h / 24).clip(lower=0)
    P["grpl"] = P.grp.map({"VP": "VP", "NE": "NE"})

    # ---- E1. five-level outcome ---------------------------------------------------------------
    B = P[P.na_set == 1].copy()
    CATS = {1: "early polyuria without Na rise", 2: "early polyuria with Na rise",
            3: "late polyuria without Na rise", 4: "late polyuria with Na rise"}
    CONTR = {"early with Na rise": {2: 1}, "early without Na rise": {1: 1}, "late with Na rise": {4: 1}, "late without Na rise": {3: 1},
             "ratio: early with / late with": {2: 1, 4: -1}, "ratio: early with / early without": {2: 1, 1: -1},
             "ratio: late with / late without": {4: 1, 3: -1}, "ratio: early without / late without": {1: 1, 3: -1}}
    crow, mrow = [], []
    for dn, col in (("main: rise >= 5 or peak >= 150", "na_up"), ("sens: rise >= 3 or peak >= 150", "na_up3")):
        up = B[col] == 1
        B["cat5"] = np.select([B.tim3 == 0, (B.tim3 == 1) & ~up, (B.tim3 == 1) & up, (B.tim3 == 2) & ~up], [0, 1, 2, 3], 4)
        for c in (1, 2, 3, 4):
            B[f"k{c}"] = (B.cat5 == c).astype(int)
        for g in ("VP", "NE"):
            s = B[B.grp == g]
            for c, cl in [(0, "no polyuria")] + list(CATS.items()):
                txt, p, lo, hi = H.fmt_prop(int((s.cat5 == c).sum()), len(s))
                crow.append(dict(definition=dn, group=g, category=c, label=cl, text=txt, pct=p, lo=lo, hi=hi, n_group=cell(len(s))))
        try:
            res = mi_linear_combos(B, ["k1", "k2", "k3", "k4", "vp"], fit_mnlogit("cat5", ["vp"] + COV, [1, 2, 3, 4]), CONTR)
            mrow += or_rows(res, dn, dict(n=len(B)))
        except Exception as e:
            mrow.append(dict(analysis=dn, term="not estimable (%s)" % type(e).__name__, n=len(B)))
        say("E1", dn)
    pd.DataFrame(crow).to_csv(os.path.join(out, "E1_category5_counts.csv"), index=False)
    pd.DataFrame(mrow).to_csv(os.path.join(out, "E1_multinomial5.csv"), index=False)

    # ---- E2. quantities anchored at the onset of polyuria --------------------------------------
    A = P[P.polyuria_24h == 1].copy()
    T0, ON = A.set_index(SID).t0, A.set_index(SID).uo_onset_h
    lab = pd.read_csv(os.path.join(ex_dir(), "x_lab.csv"), low_memory=False, usecols=[SID, "labresultoffset", "labname", "labresult"])
    lab = lab[lab[SID].isin(set(A[SID]))]
    lab["labname"] = lab.labname.astype(str).str.lower().str.strip()
    lab["v"] = pd.to_numeric(lab.labresult, errors="coerce")
    na = lab[(lab.labname == "sodium") & (lab.v > 100) & (lab.v < 200)].copy()
    na["ho"] = (na.labresultoffset - na[SID].map(T0)) / 60 - na[SID].map(ON)
    na = na.sort_values([SID, "ho"])
    pre = na[(na.ho > -24) & (na.ho <= 0)].groupby(SID).v.agg(["last", "max"])
    post = na[(na.ho > 0) & (na.ho <= 48)].groupby(SID).v.agg(["max", "count"])
    A["na_on_pre"] = A[SID].map(pre["last"])
    A["na_on_premax"] = A[SID].map(pre["max"])
    A["na_on_max48"] = A[SID].map(post["max"])
    A["na_on_set"] = ((A.na_on_premax < 150) & A.na_on_max48.notna()).astype(int)
    for c in ("na_on_pre", "na_on_max48"):
        A[c] = A[c].where(A.na_on_set == 1)
    A["na_on_rise"] = A.na_on_max48 - A.na_on_pre
    A["na_on_150"] = (A.na_on_max48 >= 150).astype(float).where(A.na_on_set == 1)
    A["na_on_rise5"] = ((A.na_on_rise >= 5) | (A.na_on_max48 >= 150)).astype(float).where(A.na_on_set == 1)
    A["na_t0_to_onset"] = (A[SID].map(pre["last"]) - A.na_pre).where(A.na_set == 1)
    inf, _ = load_infusions()
    say("infusionDrug loaded")
    ur, _ = urine_and_intake_records()
    Cc = A[[SID, "t0", "admissionweight", "riskset"]].copy()
    hr = urine_outcomes(Cc, inf, ur)
    assert np.allclose(Cc.uo_onset_h.fillna(-1).values, A.uo_onset_h.fillna(-1).values), "urine onset differs from v15"
    series = {sid: dict(zip(g.h.astype(int), g.rate)) for sid, g in hr.groupby(SID)}
    rows = []
    for sid, on in zip(A[SID], A.uo_onset_h):
        rr, st = series.get(sid, {}), int(on) - 2
        vals = [rr[h] for h in range(st, st + 24) if h in rr]
        run = 0
        for h in range(st, st + 72):
            if h in rr and rr[h] > 3.0:
                run += 1
            else:
                break
        rows.append((sid, np.mean(vals) if vals else np.nan, sum(v > 3.0 for v in vals), len(vals), run))
    U = pd.DataFrame(rows, columns=[SID, "uo_on24_rate", "poly_hours_on24", "obs_on24", "run_len_h"]).set_index(SID)
    for c in U.columns:
        A[c] = A[SID].map(U[c])
    A["run_ge6"] = (A.run_len_h >= 6).astype(int)
    SPEC2 = [("na_on_pre", "Na before onset (last value, 24 h before onset), mmol/L", "cont", 0),
             ("na_on_max48", "highest Na within 48 h after onset, mmol/L", "cont", 0),
             ("na_on_rise", "Na rise after onset (highest 48 h - before onset), mmol/L", "cont", 1),
             ("na_on_rise5", "Na rise >= 5 mmol/L or >= 150 mmol/L within 48 h after onset", "bin", 1),
             ("na_on_150", "Na >= 150 mmol/L within 48 h after onset", "bin", 1),
             ("na_t0_to_onset", "Na change from T0 baseline to onset, mmol/L", "cont", 1),
             ("uo_on24_rate", "mean urine output in the 24 h from the start of the run, mL/kg/h", "cont", 2),
             ("poly_hours_on24", "hours > 3 mL/kg/h in the 24 h from the start of the run", "cont", 0),
             ("run_len_h", "length of the first run > 3 mL/kg/h, h", "cont", 0),
             ("run_ge6", "first run >= 6 h", "bin", 1),
             ("obs_on24", "recorded urine bins in the 24 h from the start of the run", "cont", 0)]
    H.describe(A, "grp4", SPEC2, G4).to_csv(os.path.join(out, "E2_summary.csv"), index=False)
    H.boot_contrasts(A, "grp4", SPEC2, PAIRS).to_csv(os.path.join(out, "E2_contrasts_bootstrap.csv"), index=False)
    A["vp_early"] = A.vp * A.early
    COMB = {"interaction (VP x early)": {"vp_early": 1}, "VP early - VP late": {"early": 1, "vp_early": 1},
            "VP early - NE early": {"vp": 1, "vp_early": 1}, "NE early - NE late": {"early": 1}, "VP late - NE late": {"vp": 1}}
    rrow = []
    for y, lab_ in (("na_on_rise", "Na rise after onset (48 h), mmol/L"), ("uo_on24_rate", "mean urine output 24 h from run start, mL/kg/h"),
                    ("poly_hours_on24", "hours > 3 mL/kg/h in 24 h from run start")):
        d = A.dropna(subset=[y, "hospitalid"])
        res = mi_linear_combos(d, [y, "vp", "early", "vp_early"], fit_ols(y, ["vp", "early", "vp_early"] + COV), COMB)
        for k, r in res.items():
            rrow.append(dict(outcome=lab_, variable=y, term=k, n=len(d), **r))
    pd.DataFrame(rrow).to_csv(os.path.join(out, "E2_regression_onset_anchored.csv"), index=False)
    say("E2 done")

    # ---- E3. outcomes -------------------------------------------------------------------------
    SPEC3 = [("death", "in-hospital death", "bin", 1), ("icu_days_post", "ICU days after T0", "cont", 1)]
    P["g_tim"] = P.grpl + " | " + P.timing.map({"none": "no polyuria", "early": "early polyuria", "late": "late polyuria"})
    B = P[P.na_set == 1].copy()
    up = B.na_up == 1
    B["cat3"] = np.where(B.tim3 == 0, 0, np.where(up, 2, 1))
    B["g_cat3"] = B.grpl + " | " + B.cat3.map({0: "no polyuria", 1: "polyuria without Na rise", 2: "polyuria with Na rise"})
    B["cat5"] = np.select([B.tim3 == 0, (B.tim3 == 1) & ~up, (B.tim3 == 1) & up, (B.tim3 == 2) & ~up], [0, 1, 2, 3], 4)
    B["g_cat5"] = B.grpl + " | " + B.cat5.map({0: "no polyuria", **CATS})
    B["g_up"] = B.grpl + " | " + pd.Series(np.where(B.tim3 == 0, "no polyuria, ", "polyuria, "), index=B.index) + \
        pd.Series(np.where(up, "Na rise", "no Na rise"), index=B.index)
    tabs = []
    for d, gcol, pop in ((P, "g_tim", "primary population, by timing of polyuria"),
                         (B, "g_cat3", "sodium denominator, by polyuria and Na rise"),
                         (B, "g_up", "sodium denominator, Na rise by polyuria (2 x 2)"),
                         (B, "g_cat5", "sodium denominator, by timing and Na rise")):
        gs = sorted(d[gcol].unique())
        t = H.describe(d, gcol, SPEC3, gs)
        long = [dict(population=pop, group=g, variable=r["variable"], label=r["label"], value=r[g], n=r[g + "_n"])
                for _, r in t.iterrows() for g in gs]
        tabs.append(pd.DataFrame(long))
    pd.concat(tabs).to_csv(os.path.join(out, "E3_outcomes_by_category.csv"), index=False)
    for d in (B, P):
        for c in (1, 2):
            if "cat3" in d.columns:
                d[f"c{c}"] = (d.cat3 == c).astype(int)
            d[f"t{c}"] = (d.tim3 == c).astype(int)
    C3 = {"polyuria without Na rise vs no polyuria": {"c1": 1}, "polyuria with Na rise vs no polyuria": {"c2": 1},
          "polyuria with vs without Na rise": {"c2": 1, "c1": -1}}
    C2 = {"early polyuria vs no polyuria": {"t1": 1}, "late polyuria vs no polyuria": {"t2": 1}, "early vs late polyuria": {"t1": 1, "t2": -1}}
    drow = []
    for pop, g in (("both groups (adjusted for group)", None), ("VP group", "VP"), ("NE group", "NE")):
        gx = ["vp"] if g is None else []
        for lab_, d, terms, combos, catcol in (("death by polyuria and Na rise (sodium denominator)", B, ["c1", "c2"], C3, "cat3"),
                                              ("death by timing of polyuria (primary population)", P, ["t1", "t2"], C2, "tim3")):
            dd = (d if g is None else d[d.grp == g]).dropna(subset=["hospitalid"])
            extra = dict(population=pop, n=len(dd), deaths=cell(int(dd.death.sum())),
                         smallest_category_deaths=cell(int(dd.groupby(catcol).death.sum().min())))
            try:
                res = mi_linear_combos(dd, ["death"] + terms + gx, fit_logit("death", terms + gx + COV), combos)
                drow += or_rows(res, lab_, extra)
            except Exception as e:
                drow.append(dict(analysis=lab_, term="not estimable (%s)" % type(e).__name__, **extra))
    pd.DataFrame(drow).to_csv(os.path.join(out, "E3_death_models.csv"), index=False)
    say("E3 done")

    # ---- E4. vasopressin rate and duration within the VP group ----------------------------------
    V = P[P.grp == "VP"].copy()
    r = inf[inf.c_vp & (inf.rate > 0) & inf[SID].isin(set(V[SID]))].merge(V[[SID, "t0", "idx_start"]], on=SID)
    r = r[(r.infusionoffset >= r.idx_start) & (r.infusionoffset <= r.t0)].copy()        # records of the index episode
    r["umin"] = np.where((r.unit.fillna("") == "units/min") & (r.rate > 0) & (r.rate <= 0.2), r.rate, np.nan)
    r = r.sort_values([SID, "infusionoffset"])
    V["vp_last_umin"] = V[SID].map(r.groupby(SID).tail(1).set_index(SID).umin)           # rate of the last record itself
    V["vp_peak_umin"] = V[SID].map(r.groupby(SID).umin.max())
    V["vp_tapered"] = (V.vp_last_umin < V.vp_peak_umin - 1e-9).astype(float).where(V.vp_last_umin.notna())
    V["vp_last_ge003"] = (V.vp_last_umin >= 0.03 - 1e-9).astype(float).where(V.vp_last_umin.notna())
    V["vp_dur_h"] = V.idx_dur_h
    V["vp_dur_ge24"] = (V.vp_dur_h >= 24).astype(int)
    V["g_tim"] = V.timing.map({"none": "VP no polyuria", "early": "VP early", "late": "VP late"})
    SPEC4 = [("vp_last_umin", "last vasopressin rate before T0, units/min", "cont", 3),
             ("vp_last_ge003", "last vasopressin rate >= 0.03 units/min", "bin", 1),
             ("vp_peak_umin", "highest vasopressin rate of the index episode, units/min", "cont", 3),
             ("vp_tapered", "rate lowered before the stop (last < highest)", "bin", 1),
             ("vp_dur_h", "duration of the index vasopressin episode, h", "cont", 1),
             ("vp_dur_ge24", "index vasopressin episode >= 24 h", "bin", 1)]
    GV = ["VP no polyuria", "VP early", "VP late"]
    H.describe(V, "g_tim", SPEC4, GV).to_csv(os.path.join(out, "E4_vp_dose_summary.csv"), index=False)
    H.boot_contrasts(V, "g_tim", SPEC4, [("VP early", "VP no polyuria"), ("VP late", "VP no polyuria"), ("VP early", "VP late")]) \
        .to_csv(os.path.join(out, "E4_vp_dose_contrasts.csv"), index=False)
    V["vp_last_c"] = V.vp_last_umin / 0.01
    V["vp_dur_log2"] = np.log2(V.vp_dur_h.clip(lower=0.25))
    cov_int = [c for c in COV if c != "ne_dose_unknown"]                                 # as internal_vp of 01_primary_analysis.py
    erow = []
    for expo, lab_ in (("vp_last_c", "last vasopressin rate, per 0.01 units/min"), ("vp_tapered", "rate lowered before the stop vs not"),
                       ("vp_dur_log2", "duration of the vasopressin episode, per doubling")):
        d = V.dropna(subset=[expo, "hospitalid"])
        for y in ("poly8", "poly_8_24"):
            for model, covs in (("crude", []), ("adjusted (v15 covariates)", cov_int)):
                base = dict(exposure=lab_, outcome=y, model=model, n=len(d), events=cell(int(d[y].sum())))
                try:
                    res = mi_linear_combos(d, [y, expo], fit_logit(y, [expo] + covs), {expo: {expo: 1}}, covs=covs)[expo]
                    erow.append(dict(base, OR=np.exp(res["est"]), lo=np.exp(res["lo"]), hi=np.exp(res["hi"]), p=res["p"]))
                except Exception as e:
                    erow.append(dict(base, note="not estimable (%s)" % type(e).__name__))
    pd.DataFrame(erow).to_csv(os.path.join(out, "E4_vp_dose_models.csv"), index=False)
    say("E4 done")

    # ---- E5. urine laboratory values around the onset -------------------------------------------
    ul = urine_labs()
    ul = ul[ul[SID].isin(set(A[SID]))].copy()
    ul["ho"] = (ul.labresultoffset - ul[SID].map(T0)) / 60 - ul[SID].map(ON)
    ul = ul[(ul.ho > -6) & (ul.ho <= 24)]
    ul = ul[((ul.kind == "uosm") & ul.v.between(30, 1500)) | ((ul.kind == "una") & ul.v.between(0, 400)) |
            ((ul.kind == "usg") & ul.v.between(1.000, 1.060))]
    for k in ("uosm", "una", "usg"):
        A[k + "_v"] = A[SID].map(ul[ul.kind == k].groupby(SID).v.min())
        A[k + "_meas"] = A[k + "_v"].notna().astype(int)
    A["uosm_lt300"] = (A.uosm_v < 300).astype(float).where(A.uosm_meas == 1)
    A["usg_le1005"] = (A.usg_v <= 1.005).astype(float).where(A.usg_meas == 1)
    SPEC5 = [("uosm_meas", "urine osmolality measured (6 h before to 24 h after onset)", "bin", 1),
             ("uosm_v", "lowest urine osmolality, mOsm/kg (measured stays)", "cont", 0),
             ("uosm_lt300", "lowest urine osmolality < 300 mOsm/kg (measured stays)", "bin", 1),
             ("una_meas", "urine sodium measured", "bin", 1),
             ("una_v", "lowest urine sodium, mmol/L (measured stays)", "cont", 0),
             ("usg_meas", "urine specific gravity measured", "bin", 1),
             ("usg_v", "lowest urine specific gravity (measured stays)", "cont", 3),
             ("usg_le1005", "lowest urine specific gravity <= 1.005 (measured stays)", "bin", 1)]
    H.describe(A, "grp4", SPEC5, G4).to_csv(os.path.join(out, "E5_urine_labs.csv"), index=False)
    H.boot_contrasts(A, "grp4", SPEC5, PAIRS).to_csv(os.path.join(out, "E5_urine_labs_contrasts.csv"), index=False)
    A["vpg"] = np.where(A.vp == 1, "VP polyuria", "NE polyuria")
    H.describe(A, "vpg", SPEC5, ["VP polyuria", "NE polyuria"]).to_csv(os.path.join(out, "E5_urine_labs_by_group.csv"), index=False)
    H.boot_contrasts(A, "vpg", SPEC5, [("VP polyuria", "NE polyuria")]).to_csv(os.path.join(out, "E5_urine_labs_by_group_contrasts.csv"), index=False)

    keep = [SID, "grp4"] + [s[0] for s in SPEC2 + SPEC5] + ["na_on_set"]
    A[keep].to_parquet(os.path.join(wk, "eicu_early_late_extra_stays.parquet"), index=False)
    say("finished")


if __name__ == "__main__":
    main()
