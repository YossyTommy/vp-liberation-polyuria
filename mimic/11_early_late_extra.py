#!/usr/bin/env python3
"""
11_early_late_extra.py - MIMIC-IV v3.1, second set of exploratory analyses on early (0-8 h) and
late (>8-24 h) polyuria in the VP and NE groups (reported in the manuscript as post hoc exploratory analyses).

The five analyses were fixed before any of them was run; all of them are reported whatever
their direction.

  E1  five-level outcome: no polyuria / early without Na rise / early with Na rise / late without
      Na rise / late with Na rise; multinomial logistic regression (reference = no polyuria)
  E2  quantities anchored at the onset of polyuria instead of T0 (same observation time after
      onset for early and late polyuria): sodium before onset, highest sodium within 48 h after
      onset, rise; urine output, polyuric hours and length of the run in the 24 h from the start
      of the run
  E3  outcomes: in-hospital death and ICU days after T0 by timing of polyuria and by sodium rise;
      logistic regression of death
  E4  within the VP group: last and highest vasopressin rate of the index episode, taper and
      duration by timing of polyuria; logistic regression of early and late polyuria
  E5  urine osmolality, urine sodium and urine specific gravity around the onset of polyuria
      (how often measured, values when measured)

Definitions taken unchanged from v15 and from 10_early_late.py: cohort, groups, T0, polyuria and
its onset, covariates (mimic_models.COV_MI), imputation (mimic_models.impute), sodium denominator
and the sodium-rise definition (rise >= 5 mmol/L or peak >= 150 mmol/L within 72 h of T0).

New variables
  onset time        T0 + onset (h); the run starts 2 h earlier (start bin = onset - 2)
  na_on_pre         last sodium in (onset - 24 h, onset]
  na_on_max48       highest sodium in (onset, onset + 48 h]
  na_on_rise        na_on_max48 - na_on_pre, in stays with every sodium of (onset - 24 h, onset]
                    < 150 mmol/L and >= 1 sodium in (onset, onset + 48 h]  (na_on_set)
  uo_on24_rate      mean of the recorded hourly bins (mL/kg/h) in the 24 bins from the start bin
  poly_hours_on24   number of those bins > 3 mL/kg/h;  run_len_h = consecutive bins > 3 mL/kg/h
                    from the start bin (an unrecorded bin ends the run)
  icu_days_post     (ICU discharge - T0) / 24 h
  vp_last_umin      rate of the last vasopressin row of the index episode (units/hour / 60 or
                    units/min; values outside (0, 0.2] units/min are missing); vp_peak_umin = highest
                    rate of the episode; vp_tapered = last < peak; vp_dur_h = idx_ep1_h of v15
  urine labs        osmolality 51093 / 52044 and sodium 51100 (labs.csv of 01_extract.sh); specific
                    gravity 51498 (extracted here from hosp/labevents.csv.gz into
                    WORK_DIR/mimic/early_late/labs_urine_sg.csv); window (onset - 6 h, onset + 24 h]

Outputs  OUT_DIR/mimic_11/*.csv (aggregates only; cells < 10 shown as "<10")
         WORK_DIR/mimic/early_late/mimic_early_late_extra_stays.parquet, labs_urine_sg.csv (row level)
Run:     python mimic/11_early_late_extra.py   (after mimic/10_early_late.py)
Optional environment variables: NBOOT_EL (bootstrap replicates, default 2000), M_IMP (10).
"""
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mimic_common as mc  # noqa: E402
import mimic_models as mm  # noqa: E402
from mimic_common import config, NA_ITEMS, UOSM_ITEMS, VP, hrs  # noqa: E402
import early_late_helpers as H  # noqa: E402  (repository root; on sys.path once mimic_common is imported)

warnings.filterwarnings("ignore")
UNA_ITEM, USG_ITEM = 51100, 51498
G4 = ["VP early", "VP late", "NE early", "NE late"]
PAIRS = [("VP early", "VP late"), ("VP early", "NE early"), ("VP late", "NE late")]
T_START = time.time()


def say(*a):
    print("%5.0fs" % (time.time() - T_START), *a, flush=True)


# ----------------------------------------------------------------------------------------
# estimation (conventions of mimic_models: mm.impute / mm.with_imp, SE clustered on subject_id,
# Rubin's rules of mm.rubin)
def mi_linear_combos(d, imps, fit, combos):
    """fit(x) -> (params, covariance) for one completed data set; combos = {name: {term: weight}}.
    Returns the pooled estimate, SE, 95% CI and p of every linear combination."""
    acc = {k: ([], []) for k in combos}
    for imp in imps:
        b, V = fit(mm.with_imp(d, imp))
        for k, w in combos.items():
            L = pd.Series(0.0, index=b.index)
            for n, c in w.items():
                L[n] = c
            acc[k][0].append(float(L.values @ b.values))
            acc[k][1].append(float(L.values @ V.values @ L.values))
    out = {}
    for k in combos:
        q, se, lo, hi, p = mm.rubin(*acc[k])
        out[k] = dict(est=q, se=se, lo=lo, hi=hi, p=p)
    return out


def fit_ols(formula):
    def f(x):
        r = smf.ols(formula, x).fit(cov_type="cluster", cov_kwds={"groups": x.subject_id.values})
        return r.params, r.cov_params()
    return f


def fit_logit(formula):
    def f(x):
        r = mm.glm_fit(formula, x)
        return r.params, r.cov_params()
    return f


def fit_mnlogit(formula, cats):
    """Multinomial logit; returns the VP coefficients of the non-reference categories as a
    Series indexed by category, with their covariance."""
    def f(x):
        r = smf.mnlogit(formula, x).fit(disp=0, maxiter=300, method="newton", cov_type="cluster",
                                        cov_kwds={"groups": x.subject_id.values})
        V = r.cov_params()
        k = [i for i in V.index if i[1] == "vp"]
        b = pd.Series(r.params.loc["vp"].values, index=cats)
        return b, pd.DataFrame(V.loc[k, k].values, index=cats, columns=cats)
    return f


def or_rows(res, label, extra=None):
    rows = []
    for k, r in res.items():
        rows.append(dict(analysis=label, term=k, OR=np.exp(r["est"]), lo=np.exp(r["lo"]), hi=np.exp(r["hi"]), p=r["p"],
                         log_or=r["est"], se=r["se"], **(extra or {})))
    return rows


def cell(n):
    return n if n >= 10 else "<10"


def ensure_sg():
    """Urine specific gravity rows of labevents (cached below WORK_DIR)."""
    f = os.path.join(config.work_dir("mimic", "early_late"), "labs_urine_sg.csv")
    if not os.path.exists(f):
        say("extracting urine specific gravity from labevents (one pass)")
        src = os.path.join(mc.MIMIC, "hosp", "labevents.csv.gz")
        cmd = (f"gunzip -c '{src}' | LC_ALL=C awk -F, 'NR>1 && $5=={USG_ITEM} && $10!=\"\" "
               "{print $2\",\"$3\",\"$5\",\"$7\",\"$10}'" + f" > '{f}.tmp' && mv '{f}.tmp' '{f}'")
        subprocess.run(cmd, shell=True, check=True)
    x = pd.read_csv(f, header=None, names=["subject_id", "hadm_id", "itemid", "charttime", "v"], low_memory=False)
    x["v"] = pd.to_numeric(x.v, errors="coerce")
    x["hadm_id"] = pd.to_numeric(x.hadm_id, errors="coerce")
    x = x.dropna(subset=["hadm_id", "v"])
    x["hadm_id"] = x.hadm_id.astype("int64")
    x["charttime"] = pd.to_datetime(x.charttime, errors="coerce")
    return x.dropna(subset=["charttime"])


# ----------------------------------------------------------------------------------------
def main():
    out = config.out_dir("mimic_11")
    wk = config.work_dir("mimic", "early_late")
    D0 = pd.read_parquet(mc.cohort_file(mc.PRIMARY_TABLE))
    ST = pd.read_parquet(os.path.join(wk, "mimic_early_late_stays.parquet"))     # written by 10_early_late.py

    # ---- 0. reproduction of v15 values and of the groups of script 10 ------------------------
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
    chk("groups of script 10 (VP early / VP late / NE early / NE late)",
        "/".join(str(int((ST.grp4 == g).sum())) for g in G4), "98/126/337/567")
    REP = pd.DataFrame(rep)
    REP.to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
    print(REP.to_string(index=False))
    if not REP.match.all():
        say("REPRODUCTION FAILED - analysis stopped")
        sys.exit(1)

    data = mc.load_all(say)
    IMPS = mm.impute(D0)
    say("imputed")
    P = ST.copy()
    P["tim3"] = P.timing.map({"none": 0, "early": 1, "late": 2}).astype(int)
    rise = P.na_peak_72 - P.na_pre
    P["na_up"] = ((rise >= 5) | (P.na_peak_72 >= 150)).astype(float).where(P.na_set == 1)
    P["na_up3"] = ((rise >= 3) | (P.na_peak_72 >= 150)).astype(float).where(P.na_set == 1)
    P["death"] = P.hospital_expire_flag.astype(int)
    P["icu_days_post"] = (P.icuout_t / 24).clip(lower=0)

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
        for g in ("VP", "NE"):
            s = B[B.grp == g]
            for c, cl in [(0, "no polyuria")] + list(CATS.items()):
                txt, p, lo, hi = H.fmt_prop(int((s.cat5 == c).sum()), len(s))
                crow.append(dict(definition=dn, group=g, category=c, label=cl, text=txt, pct=p, lo=lo, hi=hi, n_group=cell(len(s))))
        res = mi_linear_combos(B, IMPS, fit_mnlogit(f"cat5 ~ vp + {mm.COV_MI}", [1, 2, 3, 4]), CONTR)
        mrow += or_rows(res, dn, dict(n=len(B)))
        say("E1", dn)
    pd.DataFrame(crow).to_csv(os.path.join(out, "E1_category5_counts.csv"), index=False)
    pd.DataFrame(mrow).to_csv(os.path.join(out, "E1_multinomial5.csv"), index=False)

    # ---- E2. quantities anchored at the onset of polyuria --------------------------------------
    A = P[P.poly24 == 1].copy()
    L = data["lb"][data["lb"].itemid.isin(NA_ITEMS)].merge(A[["hadm_id", "stay_id", "t0", "onset"]], on="hadm_id")
    L["ho"] = hrs(L.charttime, L.t0) - L.onset
    L = L.sort_values("charttime")
    pre = L[(L.ho > -24) & (L.ho <= 0)].groupby("stay_id").v.agg(["last", "max"])
    post = L[(L.ho > 0) & (L.ho <= 48)].groupby("stay_id").v.agg(["max", "count"])
    A["na_on_pre"] = A.stay_id.map(pre["last"])
    A["na_on_premax"] = A.stay_id.map(pre["max"])
    A["na_on_max48"] = A.stay_id.map(post["max"])
    A["na_on_set"] = ((A.na_on_premax < 150) & A.na_on_max48.notna()).astype(int)
    for c in ("na_on_pre", "na_on_max48"):
        A[c] = A[c].where(A.na_on_set == 1)
    A["na_on_rise"] = A.na_on_max48 - A.na_on_pre
    A["na_on_150"] = (A.na_on_max48 >= 150).astype(float).where(A.na_on_set == 1)
    A["na_on_rise5"] = ((A.na_on_rise >= 5) | (A.na_on_max48 >= 150)).astype(float).where(A.na_on_set == 1)
    A["na_t0_to_onset"] = (A.stay_id.map(pre["last"]) - A.na_pre).where(A.na_set == 1)    # change between T0 and onset
    hr = mc.hourly_bins(data["u"], A[["stay_id", "t0", "weight"]])
    series = {sid: dict(zip(g.h.astype(int), g.rate)) for sid, g in hr.groupby("stay_id")}
    rows = []
    for sid, on in zip(A.stay_id, A.onset):
        rr, st = series.get(sid, {}), int(on) - 2
        vals = [rr[h] for h in range(st, st + 24) if h in rr]
        run = 0
        for h in range(st, st + 72):
            if h in rr and rr[h] > mc.POLY_RATE:
                run += 1
            else:
                break
        rows.append((sid, np.mean(vals) if vals else np.nan, sum(v > mc.POLY_RATE for v in vals), len(vals), run))
    U = pd.DataFrame(rows, columns=["stay_id", "uo_on24_rate", "poly_hours_on24", "obs_on24", "run_len_h"]).set_index("stay_id")
    for c in U.columns:
        A[c] = A.stay_id.map(U[c])
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
    COMB = {"interaction (VP x early)": {"vp:early": 1}, "VP early - VP late": {"early": 1, "vp:early": 1},
            "VP early - NE early": {"vp": 1, "vp:early": 1}, "NE early - NE late": {"early": 1}, "VP late - NE late": {"vp": 1}}
    rrow = []
    for y, lab in (("na_on_rise", "Na rise after onset (48 h), mmol/L"), ("uo_on24_rate", "mean urine output 24 h from run start, mL/kg/h"),
                   ("poly_hours_on24", "hours > 3 mL/kg/h in 24 h from run start")):
        d = A.dropna(subset=[y])
        res = mi_linear_combos(d, IMPS, fit_ols(f"{y} ~ vp * early + {mm.COV_MI}"), COMB)
        for k, r in res.items():
            rrow.append(dict(outcome=lab, variable=y, term=k, n=len(d), **r))
    pd.DataFrame(rrow).to_csv(os.path.join(out, "E2_regression_onset_anchored.csv"), index=False)
    say("E2 done")

    # ---- E3. outcomes -------------------------------------------------------------------------
    SPEC3 = [("death", "in-hospital death", "bin", 1), ("icu_days_post", "ICU days after T0", "cont", 1)]
    P["g_tim"] = P.grp + " | " + P.timing.map({"none": "no polyuria", "early": "early polyuria", "late": "late polyuria"})
    B = P[P.na_set == 1].copy()
    B["cat3"] = np.where(B.tim3 == 0, 0, np.where(B.na_up == 1, 2, 1))
    B["g_cat3"] = B.grp + " | " + B.cat3.map({0: "no polyuria", 1: "polyuria without Na rise", 2: "polyuria with Na rise"})
    up = B.na_up == 1
    B["cat5"] = np.select([B.tim3 == 0, (B.tim3 == 1) & ~up, (B.tim3 == 1) & up, (B.tim3 == 2) & ~up], [0, 1, 2, 3], 4)
    B["g_cat5"] = B.grp + " | " + B.cat5.map({0: "no polyuria", **CATS})
    B["g_up"] = B.grp + " | " + pd.Series(np.where(B.tim3 == 0, "no polyuria, ", "polyuria, "), index=B.index) + \
        pd.Series(np.where(up, "Na rise", "no Na rise"), index=B.index)
    tabs = []
    for d, gcol, pop in ((P, "g_tim", "primary population, by timing of polyuria"),
                         (B, "g_cat3", "sodium denominator, by polyuria and Na rise"),
                         (B, "g_up", "sodium denominator, Na rise by polyuria (2 x 2)"),
                         (B, "g_cat5", "sodium denominator, by timing and Na rise")):
        t = H.describe(d, gcol, SPEC3, sorted(d[gcol].unique()))
        long = []
        for _, r in t.iterrows():
            for g in sorted(d[gcol].unique()):
                long.append(dict(population=pop, group=g, variable=r["variable"], label=r["label"], value=r[g], n=r[g + "_n"]))
        tabs.append(pd.DataFrame(long))
    pd.concat(tabs).to_csv(os.path.join(out, "E3_outcomes_by_category.csv"), index=False)
    drow = []
    T1, T2 = "C(cat3)[T.1]", "C(cat3)[T.2]"
    C3 = {"polyuria without Na rise vs no polyuria": {T1: 1}, "polyuria with Na rise vs no polyuria": {T2: 1},
          "polyuria with vs without Na rise": {T2: 1, T1: -1}}
    M1, M2 = "C(tim3)[T.1]", "C(tim3)[T.2]"
    C2 = {"early polyuria vs no polyuria": {M1: 1}, "late polyuria vs no polyuria": {M2: 1}, "early vs late polyuria": {M1: 1, M2: -1}}
    covg = mm.cov_without()                               # all covariates
    for pop, d, f3, f2 in (("both groups (adjusted for group)", None, f"death ~ C(cat3) + vp + {covg}", f"death ~ C(tim3) + vp + {covg}"),
                           ("VP group", "VP", f"death ~ C(cat3) + {covg}", f"death ~ C(tim3) + {covg}"),
                           ("NE group", "NE", f"death ~ C(cat3) + {covg}", f"death ~ C(tim3) + {covg}")):
        b_ = B if d is None else B[B.grp == d]
        p_ = P if d is None else P[P.grp == d]
        ev3 = int(b_.groupby("cat3").death.sum().min())
        ev2 = int(p_.groupby("tim3").death.sum().min())
        drow += or_rows(mi_linear_combos(b_, IMPS, fit_logit(f3), C3), "death by polyuria and Na rise (sodium denominator)",
                        dict(population=pop, n=len(b_), deaths=cell(int(b_.death.sum())), smallest_category_deaths=cell(ev3)))
        drow += or_rows(mi_linear_combos(p_, IMPS, fit_logit(f2), C2), "death by timing of polyuria (primary population)",
                        dict(population=pop, n=len(p_), deaths=cell(int(p_.death.sum())), smallest_category_deaths=cell(ev2)))
    pd.DataFrame(drow).to_csv(os.path.join(out, "E3_death_models.csv"), index=False)
    say("E3 done")

    # ---- E4. vasopressin rate and duration within the VP group ----------------------------------
    V = P[P.grp == "VP"].copy()
    v = data["v"]
    v = v[(v.itemid == VP) & v.stay_id.isin(set(V.stay_id)) & v.rate.notna() & (v.rate > 0)].merge(
        V[["stay_id", "t0", "idx_start"]], on="stay_id")
    v = v[(v.starttime >= v.idx_start) & (v.endtime <= v.t0)].copy()            # rows of the index episode
    v["umin"] = np.where(v.rateuom == "units/hour", v.rate / 60, np.where(v.rateuom == "units/min", v.rate, np.nan))
    v.loc[~((v.umin > 0) & (v.umin <= 0.2)), "umin"] = np.nan
    v = v.sort_values(["stay_id", "endtime", "starttime"])
    V["vp_last_umin"] = V.stay_id.map(v.groupby("stay_id").tail(1).set_index("stay_id").umin)   # rate of the last row itself
    V["vp_peak_umin"] = V.stay_id.map(v.groupby("stay_id").umin.max())
    V["vp_tapered"] = (V.vp_last_umin < V.vp_peak_umin - 1e-9).astype(float).where(V.vp_last_umin.notna())
    V["vp_last_ge003"] = (V.vp_last_umin >= 0.03 - 1e-9).astype(float).where(V.vp_last_umin.notna())
    V["vp_dur_h"] = V.idx_ep1_h
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
    covv = mm.cov_without()
    erow = []
    for expo, lab in (("vp_last_c", "last vasopressin rate, per 0.01 units/min"), ("vp_tapered", "rate lowered before the stop vs not"),
                      ("vp_dur_log2", "duration of the vasopressin episode, per doubling")):
        d = V.dropna(subset=[expo])
        for y in ("poly8", "poly_8_24"):
            for model, f in (("crude", f"{y} ~ {expo}"), ("adjusted (v15 covariates)", f"{y} ~ {expo} + {covv}")):
                try:
                    r = mi_linear_combos(d, IMPS, fit_logit(f), {expo: {expo: 1}})[expo]
                    erow.append(dict(exposure=lab, outcome=y, model=model, OR=np.exp(r["est"]), lo=np.exp(r["lo"]), hi=np.exp(r["hi"]),
                                     p=r["p"], n=len(d), events=cell(int(d[y].sum()))))
                except Exception as e:
                    erow.append(dict(exposure=lab, outcome=y, model=model, n=len(d), note="not estimable (%s)" % type(e).__name__))
    pd.DataFrame(erow).to_csv(os.path.join(out, "E4_vp_dose_models.csv"), index=False)
    say("E4 done")

    # ---- E5. urine laboratory values around the onset -------------------------------------------
    lb = data["lb"]
    ul = pd.concat([lb[lb.itemid.isin(UOSM_ITEMS)].assign(kind="uosm"), lb[lb.itemid == UNA_ITEM].assign(kind="una"),
                    ensure_sg().assign(kind="usg")])
    ul = ul.merge(A[["hadm_id", "stay_id", "t0", "onset"]], on="hadm_id")
    ul["ho"] = hrs(ul.charttime, ul.t0) - ul.onset
    ul = ul[(ul.ho > -6) & (ul.ho <= 24)]
    ul = ul[((ul.kind == "uosm") & ul.v.between(30, 1500)) | ((ul.kind == "una") & ul.v.between(0, 400)) |
            ((ul.kind == "usg") & ul.v.between(1.000, 1.060))]
    for k, how in (("uosm", "min"), ("una", "min"), ("usg", "min")):
        A[k + "_v"] = A.stay_id.map(ul[ul.kind == k].groupby("stay_id").v.agg(how))
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

    keep = ["stay_id", "grp4"] + [s[0] for s in SPEC2 + SPEC5] + ["na_on_set"]
    A[keep].to_parquet(os.path.join(wk, "mimic_early_late_extra_stays.parquet"))
    say("finished")


if __name__ == "__main__":
    main()
