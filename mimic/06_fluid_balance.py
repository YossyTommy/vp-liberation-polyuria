#!/usr/bin/env python3
"""
06_fluid_balance.py - MIMIC-IV v3.1, cumulative fluid balance.

Builds the cumulative fluid balance and fluid overload % (FO%) from ICU admission to the anchor
(definitions: mimic_fluid.py) and examines whether polyuria after liberation is explained by
excretion of retained resuscitation fluid. Post hoc, exploratory.

MICE (PMM k=5, seed 2026), the covariate set COV_MI, the binomial GLM with subject-clustered
SE, Rubin pooling and the group-dummy models (Firth for groups with < 10 events) are those of
mimic_models.py and mimic_groups.py. Every model of this script merges the imputations with
with_imp_fo (mimic_fluid.py), which adds the imputed FO% when the imputation contains it and
is otherwise identical to with_imp. Analysis populations are the flags stored in the
analysis-ready cohort tables:
  two-group primary   pop_primary of 02_primary_analysis.py (T0 = end of the index episode,
                      no other vasopressor running at T0)
  five groups         pop_main and ugrp of 05_usage_groups.py (T0b = liberation from all
                      vasopressors)
New variables are computed at the anchor of each analysis; NE-only stays are identical in both.

Analyses
  (a) description by group: median [IQR], SMD against NE-only, coverage
  (b) two-group primary and NE-first vs NE-only: COV_MI vs COV_MI + FO% (continuous, imputed),
      on the same imputations; FO% natural spline (df 4) and observed-FO% only as references;
      the five-group model of 05_usage_groups.py is reproduced
  (c) odds ratios by FO% tertile (cut on observed FO% of NE-only stays) and interaction tests
      (D1 Wald, 2 df; trend, 1 df)
  (d) within NE-only: polyuria by FO% quartile (Wilson CI) and adjusted OR per 5 % FO%
  (e) vasopressor >= 24 h and FO% >= NE-only median
  (f) overlap weighting (ATO; PS = COV_MI + FO%) of the two-group primary analysis: estimated
      per imputation, log OR averaged; percentile CI from a subject-level cluster bootstrap
      (imputations fixed, all imputations re-estimated in each replicate); SMD before/after
  (g) no loop diuretic in the 24 h after T0; NE-first vs VP-first; Na >= 150
  Na >= 150 = within 72 h, new onset (denominator: pre-T0 24 h maximum Na < 150).

Paths come from config.py. Optional environment variables: NBOOT (default 200), M_IMP.
Inputs
  WORK_DIR/mimic/cohort/   analysis_ready_mimic_primary.parquet, analysis_ready_mimic_usage_groups.parquet
  MIMIC_DIR/icu/inputevents.csv.gz, outputevents.csv.gz (cached in WORK_DIR/mimic/ext as
  ie_cohort.parquet and oe_cohort.parquet)
Outputs
  OUT_DIR/mimic_06/   results_mimic_fluid_balance.xlsx, key_results.json, run_log.txt
  WORK_DIR/mimic/cohort/   mimic_fluid_balance_2grp_anchor.parquet, mimic_fluid_balance_T0b_anchor.parquet

Run:  python mimic/06_fluid_balance.py   (after 02_primary_analysis.py and 05_usage_groups.py)
"""
import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # folder of the shared modules
import mimic_common as mc  # noqa: E402
import mimic_models as mm  # noqa: E402
from mimic_common import config  # noqa: E402
from mimic_models import YS  # noqa: E402
from mimic_groups import G4, group_models, nN, smd  # noqa: E402
from mimic_fluid import COV_FO, prep_io, fluid_vars, with_imp_fo, impute_fo, ato_analysis, wsmd  # noqa: E402

warnings.filterwarnings("ignore")

NBOOT = int(os.environ.get("NBOOT", 200))
YLAB = {"poly24": "polyuria ≤24 h", "poly8": "polyuria 0–8 h", "poly_8_24": "polyuria >8–24 h", "na150_72": "Na ≥150 ≤72 h (new-onset)"}
COV = mm.COV_MI
say = mc.RunLog()


# ===========================================================================
# Extraction (cached)
def extract(sids):
    """inputevents and outputevents of the cohort stays, cached as parquet in WORK_DIR/mimic/ext."""
    os.makedirs(mc.EXT, exist_ok=True)
    fi, fo = os.path.join(mc.EXT, "ie_cohort.parquet"), os.path.join(mc.EXT, "oe_cohort.parquet")
    if not os.path.exists(fi):
        say("extract inputevents")
        cols = ["stay_id", "starttime", "endtime", "itemid", "amount", "amountuom", "statusdescription", "ordercategoryname"]
        out = [ch[ch.stay_id.isin(sids)] for ch in
               pd.read_csv(os.path.join(mc.MIMIC, "icu", "inputevents.csv.gz"), usecols=cols, chunksize=2_000_000)]
        ie = pd.concat(out)
        for c in ("starttime", "endtime"):
            ie[c] = pd.to_datetime(ie[c])
        ie.to_parquet(fi)
    if not os.path.exists(fo):
        say("extract outputevents")
        oe = pd.read_csv(os.path.join(mc.MIMIC, "icu", "outputevents.csv.gz"),
                         usecols=["subject_id", "stay_id", "charttime", "itemid", "value", "valueuom"])
        oe = oe[oe.stay_id.isin(sids)].copy()
        oe["charttime"] = pd.to_datetime(oe.charttime)
        oe.to_parquet(fo)
    return pd.read_parquet(fi), pd.read_parquet(fo)


# ===========================================================================
# Estimation tools used only by this script
def wilson(k, n):
    if n == 0:
        return "0/0"
    lo, hi = proportion_confint(k, n, method="wilson")
    return "%d/%d (%.1f%%; %.1f–%.1f)" % (k, n, 100 * k / n, 100 * lo, 100 * hi)


def gm(d, imps, y, col, levels, cov, label, model):
    """group_models (crude / adjusted, MICE, clustered SE, Firth below 10 events) with a model label."""
    rows = group_models(d, imps, y, col, levels, cov=cov, label=label, with_imp=with_imp_fo)
    for r in rows:
        r["model"] = model
        r["ref_group"] = levels[0]
        r["outcome_label"] = YLAB.get(y, y)
    return rows


def na_set(d):
    return d[d.na_max_pre24 < 150]


def d1_test(Qs, Us):
    """Multi-parameter Wald test D1 (Li, Raghunathan and Rubin 1991)."""
    Qs, Us = np.asarray(Qs), np.asarray(Us)
    m, k = Qs.shape
    Qb, Ub = Qs.mean(0), Us.mean(0)
    B = np.cov(Qs.T, ddof=1).reshape(k, k)
    Ui = np.linalg.inv(Ub)
    r = (1 + 1 / m) * np.trace(B @ Ui) / k
    D = float(Qb @ Ui @ Qb) / (k * (1 + r))
    t = k * (m - 1)
    v = 4 + (t - 4) * (1 + (1 - 2 / t) / r) ** 2 if (t > 4 and r > 0) else t * (1 + 1 / k) * (1 + r) ** 2 / 2
    return D, stats.f.sf(D, k, v)


def interaction(d, imps, y, expo, strat, cov):
    """expo x stratum (three categories): D1 (2 df); expo x stratum index (trend, 1 df, Rubin)."""
    Qs, Us, qt, vt = [], [], [], []
    for imp in imps:
        x = with_imp_fo(d, imp)
        r = mm.glm_fit(f"{y}~{expo}*C({strat})+{cov}", x)
        nm = [c for c in r.params.index if c.startswith(f"{expo}:C({strat})")]
        Qs.append(r.params[nm].values)
        Us.append(r.cov_params().loc[nm, nm].values)
        r2 = mm.glm_fit(f"{y}~{expo}*{strat}+{cov}", x)
        t_ = f"{expo}:{strat}"
        qt.append(r2.params[t_])
        vt.append(r2.cov_params().loc[t_, t_])
    D, p = d1_test(Qs, Us)
    q, se, lo, hi, pt = mm.rubin(qt, vt)
    return dict(p_interaction_3cat_D1=p, D1=D, ratio_of_OR_per_tertile=np.exp(q), rOR_lo=np.exp(lo), rOR_hi=np.exp(hi),
                p_interaction_trend=pt)


SMDV = [("age10", "age /10 y"), ("male", "male"), ("cvicu", "CVICU"), ("kdigo_i", "KDIGO pre (imputed; 0–3 as numeric)"),
        ("na_i", "Na pre (imputed)"), ("map_i", "MAP base (imputed)"), ("diur_pre24", "loop diuretic 24 h pre"),
        ("fluid_pre24_l", "fluid 24 h pre, L (existing)"), ("lact_i", "lactate (imputed)"), ("log_press_h", "log(1+pressor h)"),
        ("ne_peak_c", "NE peak /0.1"), ("other_vaso_pre", "other vasopressor pre"), ("vent_t0", "ventilated at T0"),
        ("steroid_pre24", "steroid 24 h pre"), ("inotrope_t0", "inotrope at T0"), ("sepsis", "sepsis"),
        ("card_surg", "cardiac surgery"), ("emergency", "emergency"), ("fo_i", "FO% (imputed)"),
        ("bal_pre24_l", "balance 24 h pre, L (observed; not in PS)"), ("icu_pre_h", "ICU h before T0 (not in PS)"),
        ("in_cum_l", "cum input, L (observed; not in PS)"), ("out_cum_l", "cum output, L (observed; not in PS)")]


# ===========================================================================
DESC = [("icu_pre_h", "ICU time before T0, h"), ("in_cum_l", "cumulative input ICU→T0, L"),
        ("out_cum_l", "cumulative output ICU→T0, L"), ("bal_cum_l", "cumulative balance ICU→T0, L"),
        ("fo_pct", "FO% (balance/weight ×100)"), ("fo_pct_nogu", "FO% excluding GU irrigation in/out"),
        ("in_pre24_l", "input 24 h before T0, L (prorated)"), ("out_pre24_l", "output 24 h before T0, L"),
        ("bal_pre24_l", "balance 24 h before T0, L"), ("bal_pre24_mlkg", "balance 24 h before T0, mL/kg"),
        ("in_post8_l", "input 0–8 h after T0, L (descriptive)"), ("in_post24_l", "input 0–24 h after T0, L (descriptive)"),
        ("fluid_pre24_l", "existing covariate: fluid 24 h before T0, L (unprorated)"), ("press_h", "vasopressor time, h"),
        ("weight", "weight, kg")]


def describe(groups, ref):
    rows = []
    for var, lab in DESC:
        r = dict(variable=var, label=lab)
        for g, s in groups.items():
            x = s[var].dropna()
            r[f"{g}"] = ("%.2f [%.2f–%.2f]" % (x.median(), x.quantile(.25), x.quantile(.75))) if len(x) else "NA"
            r[f"{g} n_obs"] = "%d/%d" % (len(x), len(s))
            if g != ref:
                r[f"{g} SMD vs {ref}"] = round(smd(s[var], groups[ref][var]), 3)
        rows.append(r)
    return pd.DataFrame(rows)


def coverage(groups):
    rows = []
    for g, s in groups.items():
        N = len(s)
        f = lambda m: "%d/%d (%.1f%%)" % (int(m.sum()), N, 100 * m.mean() if N else np.nan)
        rows.append(dict(group=g, N=N,
                         input_recorded_ICU_to_T0=f(s.in_cov_cum == 1), output_recorded_ICU_to_T0=f(s.out_cov_cum == 1),
                         weight_available=f(s.weight > 0), FO_pct_computable=f(s.fo_pct.notna()),
                         FO_pct_missing_imputed=f(s.fo_pct.isna()),
                         input_recorded_pre24=f(s.in_cov_pre24 == 1), output_recorded_pre24=f(s.out_cov_pre24 == 1),
                         balance_pre24_computable=f(s.bal_pre24_l.notna()),
                         pre24_window_truncated_by_ICU_admission=f(s.pre24_truncated == 1),
                         input_recorded_post0_8=f(s.in_cov_post8 == 1), input_recorded_post0_24=f(s.in_cov_post24 == 1),
                         GU_irrigation_before_T0=f(s.gu_irrig_cum == 1),
                         ICU_h_before_T0_when_output_missing=("%.1f [%.1f–%.1f]" % tuple(s.loc[s.out_cov_cum == 0, "icu_pre_h"].quantile([.5, .25, .75])))
                         if (s.out_cov_cum == 0).any() else "—"))
    return pd.DataFrame(rows)


def main():
    t_all = time.time()
    outdir = config.out_dir("mimic_06")
    SH = {}
    D0 = pd.read_parquet(mc.cohort_file(mc.PRIMARY_TABLE))
    DB = pd.read_parquet(mc.cohort_file(mc.USAGE_TABLE))
    sids = set(D0.stay_id.astype("int64")) | set(DB.stay_id.astype("int64"))
    ie, oe = extract(sids)
    I, O, io_counts = prep_io(ie, oe)
    say("io row counts:", io_counts)
    F0 = fluid_vars(I, O, D0[["stay_id", "t0", "intime", "weight"]])
    FB = fluid_vars(I, O, DB[["stay_id", "t0", "intime", "weight"]])
    fcols = [c for c in F0.columns if c != "stay_id"]
    D0 = D0.drop(columns=[c for c in fcols if c in D0.columns]).merge(F0, on="stay_id", how="left")
    DB = DB.drop(columns=[c for c in fcols if c in DB.columns]).merge(FB, on="stay_id", how="left")
    assert len(D0) == 16344 and len(DB) == 16344

    P0 = D0[D0.pop_primary == 1].copy()
    MAIN = DB[(DB.pop_main == 1) & DB.ugrp.isin(G4)].copy()
    # ---- reproduction checks ------------------------------------------------------
    rep = []
    n2 = P0.grp.value_counts().to_dict()
    e2 = P0.groupby("grp").poly24.sum().to_dict()
    n5 = MAIN.ugrp.value_counts().to_dict()
    target = {"NE-only": 4667, "NE-first": 560, "VP-first": 456, "VP-only": 203, "Simultaneous": 190}
    ok = (n2 == {"NE": 4667, "VP": 743}) and (e2 == {"NE": 904, "VP": 224}) and all(n5.get(k) == v for k, v in target.items())
    say("reproduction counts:", n2, e2, n5, "OK" if ok else "MISMATCH")
    if not ok:
        raise SystemExit("population mismatch — stop")
    # new variables of NE-only stays must be identical under the two-group and five-group anchors
    ne_chk = P0[P0.grp == "NE"].set_index("stay_id").fo_pct.sort_index().equals(
        MAIN[MAIN.ugrp == "NE-only"].set_index("stay_id").fo_pct.sort_index())
    say("NE-only FO% identical across 2-group/5-group anchors:", ne_chk)

    t = time.time()
    IMPS0 = mm.impute(D0)
    IMPSB = mm.impute(DB)
    IMPS0f = impute_fo(D0)
    IMPSBf = impute_fo(DB)
    say("imputations %.0fs" % (time.time() - t))
    o0 = {y: mm.or_mi(P0, IMPS0, y, with_imp=with_imp_fo)["vp"] for y in YS}
    rep += [dict(check="2-group primary N (VP/NE)", value="%d / %d" % (n2["VP"], n2["NE"]), expected="743 / 4667"),
            dict(check="2-group primary poly24 events (VP/NE)", value="%d / %d" % (e2["VP"], e2["NE"]), expected="224 / 904"),
            dict(check="2-group primary aOR poly24 (script 02 imputations, COV_MI)", value=mm.fmt(o0["poly24"]), expected="1.51 (1.24–1.84)"),
            dict(check="2-group primary aOR poly8", value=mm.fmt(o0["poly8"]), expected=""),
            dict(check="2-group primary aOR poly_8_24", value=mm.fmt(o0["poly_8_24"]), expected="")]
    for g in G4:
        rep.append(dict(check=f"5-group analysis N {g}", value=str(n5[g]), expected=str(target[g])))
    rep.append(dict(check="NE-only FO% identical in 2-group and 5-group data", value=str(ne_chk), expected="True"))
    r5 = []
    for y in YS:
        r5 += gm(MAIN, IMPSB, y, "ugrp", G4, COV, "reproduction of script 05: 5-group model", "COV_MI (script 05 imputations)")
    SH["01_reproduction"] = pd.concat([pd.DataFrame(rep), pd.DataFrame(r5)], ignore_index=True)
    say(pd.DataFrame(rep).to_string())

    # ---- coverage and description ---------------------------------------------------
    groups = {g: MAIN[MAIN.ugrp == g] for g in G4}
    groups["2grp VP (primary)"] = P0[P0.grp == "VP"]
    groups["2grp NE (primary)"] = P0[P0.grp == "NE"]
    COVT = coverage(groups)
    SH["02_coverage"] = pd.concat([COVT, pd.DataFrame([dict(group="row counts (all 16,344 extracted stays)", **{k: str(v) for k, v in io_counts.items()})])],
                                  ignore_index=True)
    SH["03_descriptive"] = describe(groups, "NE-only")

    # ---- (b) before / after adding FO% ----------------------------------------------
    # natural spline with fixed knots (quartiles of observed FO% in the two-group primary population; boundaries =
    # observed minimum / maximum over all extracted stays, which PMM cannot exceed); constraints="center" (df = 4)
    kn = tuple(round(float(q), 4) for q in P0.fo_pct.dropna().quantile([.25, .5, .75]))
    allfo = pd.concat([D0.fo_pct, DB.fo_pct]).dropna()
    COV_SPL = mm.COV_MI + "+cr(fo_i, knots=%s, lower_bound=%.4f, upper_bound=%.4f, constraints='center')" % (
        str(kn), allfo.min() - 1e-3, allfo.max() + 1e-3)
    say("spline term:", COV_SPL.split("+")[-1])
    MP = MAIN[MAIN.ugrp.isin(["NE-only", "NE-first"])].copy()
    rows = []
    for y in YS + ["na150_72"]:
        p0_, mp_ = (P0, MP) if y != "na150_72" else (na_set(P0), na_set(MP))
        for cov, nm, imps0, impsb in ((COV, "A: COV_MI (FO%-augmented imputations)", IMPS0f, IMPSBf),
                                      (COV_FO, "B: COV_MI + FO% (linear)", IMPS0f, IMPSBf),
                                      (COV_SPL, "C: COV_MI + FO% natural cubic spline (3 fixed knots, centred, 4 df)", IMPS0f, IMPSBf),
                                      (COV, "ref: COV_MI (original imputations of scripts 02/05)", IMPS0, IMPSB)):
            rows += gm(p0_, imps0, y, "grp", ["NE", "VP"], cov, "(b) 2-group primary VP vs NE", nm)
            rows += gm(mp_, impsb, y, "ugrp", ["NE-only", "NE-first"], cov, "(b) NE-first vs NE-only (pairwise)", nm)
        obs0, obsb = p0_[p0_.fo_pct.notna()], mp_[mp_.fo_pct.notna()]
        rows += gm(obs0, IMPS0f, y, "grp", ["NE", "VP"], COV_FO, "(b) 2-group primary, observed FO% only", "B: COV_MI + FO%")
        rows += gm(obsb, IMPSBf, y, "ugrp", ["NE-only", "NE-first"], COV_FO, "(b) NE-first vs NE-only, observed FO% only", "B: COV_MI + FO%")
        if y != "na150_72":
            rows += gm(MAIN, IMPSBf, y, "ugrp", G4, COV_FO, "(b) 5-group model (structure of script 05)", "B: COV_MI + FO%")
        say("(b)", y)
    SH["04_b_FO_adjustment"] = pd.DataFrame(rows)

    # ---- (c) FO% tertiles -----------------------------------------------------------
    ne_fo = MAIN.loc[MAIN.ugrp == "NE-only", "fo_pct"].dropna()
    cut = ne_fo.quantile([1 / 3, 2 / 3]).values
    def tert(s):
        return pd.cut(s, [-np.inf, cut[0], cut[1], np.inf], labels=[0, 1, 2], right=False).astype(float)
    P0["fo_t"], MP["fo_t"] = tert(P0.fo_pct), tert(MP.fo_pct)
    rows, irows = [], []
    for y in ("poly8", "poly24"):
        for lab, d, col, lev in (("2-group primary VP vs NE", P0, "grp", ["NE", "VP"]),
                                 ("NE-first vs NE-only", MP, "ugrp", ["NE-only", "NE-first"])):
            imps = IMPS0f if col == "grp" else IMPSBf
            for k in (0, 1, 2):
                s = d[d.fo_t == k]
                rr = gm(s, imps, y, col, lev, COV_FO, f"(c) {lab}, FO% tertile T{k + 1} [{['<%.2f' % cut[0], '%.2f–%.2f' % (cut[0], cut[1]), '≥%.2f' % cut[1]][k]}%]", "B: COV_MI + FO%")
                for r in rr:
                    r["FO_range_pct"] = "%.2f [%.2f–%.2f]" % (s.fo_pct.median(), s.fo_pct.min(), s.fo_pct.max())
                rows += rr
            dd = d[d.fo_t.notna()].copy()
            dd["expo"] = (dd[col] == lev[1]).astype(int)
            it = interaction(dd, imps, y, "expo", "fo_t", COV_FO)
            irows.append(dict(analysis=f"(c) {lab}", outcome=y, n_model=len(dd), FO_missing_excluded=int(d.fo_t.isna().sum()),
                              cutpoints_NEonly_FO_pct="%.2f / %.2f" % tuple(cut), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in it.items()}))
        say("(c)", y)
    SH["05_c_FO_tertiles"] = pd.DataFrame(rows)
    SH["05b_c_interaction"] = pd.DataFrame(irows)

    # ---- (d) FO% and 0-8 h polyuria within NE-only ----------------------------------
    NEo = P0[P0.grp == "NE"].copy()
    qc = NEo.fo_pct.dropna().quantile([.25, .5, .75]).values
    NEo["fo_q"] = pd.cut(NEo.fo_pct, [-np.inf, *qc, np.inf], labels=[1, 2, 3, 4], right=False)
    rows = []
    for k in (1, 2, 3, 4):
        s = NEo[NEo.fo_q == k]
        rows.append(dict(analysis="(d) NE-only, FO% quartile (observed)", quartile=f"Q{k}",
                         FO_pct_range="%.2f to %.2f" % (s.fo_pct.min(), s.fo_pct.max()),
                         poly8=wilson(int(s.poly8.sum()), len(s)), poly24=wilson(int(s.poly24.sum()), len(s)),
                         poly_8_24=wilson(int(s.poly_8_24.sum()), len(s))))
    s = NEo[NEo.fo_q.isna()]
    rows.append(dict(analysis="(d) NE-only, FO% not computable (no output record)", quartile="missing",
                     poly8=wilson(int(s.poly8.sum()), len(s)), poly24=wilson(int(s.poly24.sum()), len(s)),
                     poly_8_24=wilson(int(s.poly_8_24.sum()), len(s))))
    NQ = pd.DataFrame(rows)
    rows = []
    for y in YS:
        for nm, cv in (("crude", None), ("adjusted COV_MI", COV), ("adjusted COV_MI − fluid_pre24_l", mm.cov_without("fluid_pre24_l"))):
            o = mm.or_mi(NEo, IMPS0f, y, expo="fo5_i", cov=cv, with_imp=with_imp_fo)["fo5_i"]
            rows.append(dict(analysis="(d) NE-only, OR per +5 percentage points FO% (MICE-imputed FO%)", outcome=y,
                             events=nN(NEo, y), model=nm, OR=mm.fmt(o), p=round(o["p"], 4), est=o["OR"], lo=o["lo"], hi=o["hi"]))
    SH["06_d_NEonly_FO"] = pd.concat([NQ, pd.DataFrame(rows)], ignore_index=True)

    # ---- (e) long vasopressor course and high FO% -----------------------------------
    med = ne_fo.median()
    rows = []
    for y in YS + ["na150_72"]:
        for lab, d, col, lev, imps in (("2-group primary VP vs NE", P0, "grp", ["NE", "VP"], IMPS0f),
                                       ("NE-first vs NE-only", MP, "ugrp", ["NE-only", "NE-first"], IMPSBf)):
            s = d[(d.press_h >= 24) & (d.fo_pct >= med)]
            if y == "na150_72":
                s = na_set(s)
            for cov, nm in ((COV, "A: COV_MI"), (COV_FO, "B: COV_MI + FO%")):
                rows += gm(s, imps, y, col, lev, cov, f"(e) {lab}; pressor ≥24 h & FO% ≥{med:.2f}% (NE-only median)", nm)
        say("(e)", y)
    SH["07_e_severe_fluid"] = pd.DataFrame(rows)

    # ---- (f) overlap weighting --------------------------------------------------
    t = time.time()
    ato_rows, smd_rows = [], []
    for lab, d, ys in (("2-group primary", P0, YS), ("2-group primary, Na new-onset risk set", na_set(P0), ["na150_72"])):
        res, pts, _ = ato_analysis(d, IMPS0f, ys, COV_FO, NBOOT, with_imp_fo)
        for y in ys:
            r = res[y]
            ato_rows.append(dict(analysis=f"(f) ATO overlap weighting, {lab}", outcome=y, events_VP=nN(d[d.vp == 1], y),
                                 events_NE=nN(d[d.vp == 0], y), weighted_risk_VP="%.1f%%" % (100 * r["p1"]),
                                 weighted_risk_NE="%.1f%%" % (100 * r["p0"]),
                                 OR_ATO="%.2f (%.2f–%.2f)" % (np.exp(r["lor"]), np.exp(r["lor_lo"]), np.exp(r["lor_hi"])),
                                 RD_ATO_pct="%.1f (%.1f to %.1f)" % (100 * r["rd"], 100 * r["rd_lo"], 100 * r["rd_hi"]),
                                 est=np.exp(r["lor"]), lo=np.exp(r["lor_lo"]), hi=np.exp(r["lor_hi"]), nboot=r["nboot"],
                                 PS_model="vp ~ " + COV_FO))
        if lab == "2-group primary":
            dd = d.reset_index(drop=False).rename(columns={"index": "_ix"}).set_index("_ix")
            z = dd.vp.values
            acc = {v: [] for v, _ in SMDV}
            for imp, p in zip(IMPS0f, pts):
                x = with_imp_fo(dd, imp)
                x["kdigo_i"] = x.kdigo_i.astype(float)
                for v, _ in SMDV:
                    acc[v].append(wsmd(x, z, p[1], v))
            ess = [((p[1][z == 1].sum()) ** 2 / (p[1][z == 1] ** 2).sum(), (p[1][z == 0].sum()) ** 2 / (p[1][z == 0] ** 2).sum()) for p in pts]
            for v, labv in SMDV:
                a = np.array(acc[v])
                smd_rows.append(dict(variable=v, label=labv, SMD_unweighted=round(a[:, 0].mean(), 3), SMD_ATO_weighted=round(a[:, 1].mean(), 3)))
            smd_rows.append(dict(variable="ESS", label="effective sample size VP / NE (mean over imputations)",
                                 SMD_unweighted="%d / %d" % ((z == 1).sum(), (z == 0).sum()),
                                 SMD_ATO_weighted="%.0f / %.0f" % tuple(np.mean(ess, 0))))
    SH["08_f_ATO"] = pd.DataFrame(ato_rows)
    SH["08b_f_ATO_SMD"] = pd.DataFrame(smd_rows)
    say("(f) ATO %.0fs" % (time.time() - t))

    # ---- (g) no loop diuretic; NE-first vs VP-first ----------------------------------
    rows = []
    P0n, MPn = P0[P0.loop_post24 == 0], MP[MP.loop_post24 == 0]
    H2H = MAIN[MAIN.ugrp.isin(["VP-first", "NE-first"])]
    for y in YS + ["na150_72"]:
        for cov, nm in ((COV, "A: COV_MI"), (COV_FO, "B: COV_MI + FO%")):
            a0, ab = (P0n, MPn) if y != "na150_72" else (na_set(P0n), na_set(MPn))
            rows += gm(a0, IMPS0f, y, "grp", ["NE", "VP"], cov, "(g1) no loop diuretic 0–24 h after T0: 2-group primary VP vs NE", nm)
            rows += gm(ab, IMPSBf, y, "ugrp", ["NE-only", "NE-first"], cov, "(g1) no loop diuretic 0–24 h after T0b: NE-first vs NE-only", nm)
            h = H2H if y != "na150_72" else na_set(H2H)
            rows += gm(h, IMPSBf, y, "ugrp", ["VP-first", "NE-first"], cov, "(g2) head-to-head NE-first vs VP-first (ref)", nm)
        say("(g)", y)
    SH["09_g_no_diuretic_h2h"] = pd.DataFrame(rows)

    nfail = sum(int((t_.astype(str) == "fit failed").sum().sum()) for t_ in SH.values())
    say("model cells reporting 'fit failed':", nfail)
    if nfail:
        raise SystemExit("fit failures — stop")
    # ---- README / save --------------------------------------------------------------
    readme = [
        ("script", "06_fluid_balance.py; uses the functions of 02_primary_analysis.py (MICE m=%d PMM k=5 seed 2026, COV_MI, GLM binomial with subject_id-clustered SE, Rubin) and 05_usage_groups.py (group models; Firth only for groups with <10 events)" % mm.M_IMP),
        ("populations", "2-group primary = script 02 pop_primary (VP 743 / NE 4,667; anchor = script 02 t0). 5 groups = script 05 pop_main (anchor T0b). Read from the analysis-ready cohort tables; no definition changed."),
        ("input", "inputevents amountuom=='mL' and amount>0 (all items, incl. drug carriers and enteral feeds); prorated by time over starttime–endtime into [a,b); zero-duration rows = point at starttime. outputevents 227488 (GU irrigant in) added to input."),
        ("output", "all outputevents items (charttime in [a,b)), values as recorded (negatives kept); 227489 GU irrigant/urine out kept at gross volume, as in the urine rule (urine.csv counts 227489 gross, does not subtract 227488)."),
        ("windows", "cum = [ICU intime, T0); pre24 = [max(intime, T0−24 h), T0); post8/post24 = [T0, T0+8/24 h) input only (descriptive)."),
        ("coverage", "a window with no input row (or no output row) is missing, not 0; balance/FO% missing if either side missing; FO% = (input−output) L / weight kg ×100 (weight = weight of script 02)."),
        ("imputation", "FO% added to the imputation model of script 02 (IMPVARS + fo_pct, same AUXV, same seed, m=10). Model A (COV_MI) and B (COV_MI + FO%) use the same FO%-augmented imputations; 'ref' rows use the original imputations of scripts 02/05."),
        ("head-to-head", "in (g2) rows the column ref_NE_only holds the VP-first (reference) events"),
        ("NE-first contrasts", "pairwise models restricted to NE-first + NE-only stays (and VP-first + NE-first for the head-to-head); 5-group model rows reproduce script 05."),
        ("tertiles/quartiles", "cut on observed FO% of NE-only stays; stays without observed FO% are excluded from those strata (counts given). Interaction: D1 multi-parameter Wald across imputations (2 df) and expo × tertile index (1 df)."),
        ("ATO", "propensity vp ~ COV_MI + FO%% per imputation; weights VP 1−ps, NE ps; weighted risks, marginal OR and RD; point = mean over imputations of log OR; CI = 2.5/97.5 percentiles of %d subject-cluster bootstrap replicates, each re-estimating all 10 imputations (imputations held fixed)." % NBOOT),
        ("Na ≥150", "72 h after T0, new-onset risk set (pre-T0 24 h Na max <150), as in scripts 02/05."),
        ("limitation", "Fluids and output before ICU admission (ED, OR, ward) are not recorded in MIMIC-IV ICU tables and are not included; insensible losses not included."),
        ("status", "post hoc, exploratory; no multiplicity correction"),
    ]
    SH = {"00_README": pd.DataFrame(readme, columns=["item", "description"]), **SH}
    xl = os.path.join(outdir, "results_mimic_fluid_balance.xlsx")
    with pd.ExcelWriter(xl) as xw:
        for k, t_ in SH.items():
            t_.to_excel(xw, sheet_name=k[:31], index=False)
    keep = ["stay_id", "subject_id", "hadm_id", "grp", "t0", "intime", "weight"] + fcols
    D0[keep + ["pop_primary"]].to_parquet(mc.cohort_file(mc.FLUID_TABLE_2GRP))
    DB[keep + ["ugrp", "pop_main"]].to_parquet(mc.cohort_file(mc.FLUID_TABLE_T0B))
    with open(os.path.join(outdir, "key_results.json"), "w") as f:
        json.dump(dict(io_row_counts=io_counts, cut_tertile=list(map(float, cut)), ne_median_fo=float(med),
                       quartiles=list(map(float, qc))), f, indent=1, default=str)
    say("total %.0f s" % (time.time() - t_all))
    say.write(os.path.join(outdir, "run_log.txt"))


if __name__ == "__main__":
    main()