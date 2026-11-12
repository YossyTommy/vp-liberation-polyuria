#!/usr/bin/env python3
"""
05_usage_groups.py - MIMIC-IV v3.1, exploratory stop-order groups.

Regroups stays by the pattern of vasopressin (VP) / norepinephrine (NE) use and by the order in
which the two drugs were stopped (classification rules: mimic_groups.py). Anchor = liberation
from all vasopressors (T0b). Post hoc, exploratory.

The extraction, hourly urine bins, run rule, covariates, exclusions, MICE and subject-clustered
standard errors are those of the primary analysis (mimic_common.py, mimic_models.py). Only the
anchor table passed to build() differs: T0 of VP-containing stays becomes T0b; the NE-only
group is the NE group of the primary analysis.

Definitions
  NE-only        no VP record in the stay; T0b = end of the first NE episode; excluded if another
                 vasopressor is running at T0b
  VP-containing stays   VP-only, NE-first, Simultaneous, VP-first (see mimic_groups.py).
     Stays with a VP episode starting between T0a(VP) and T0b ("VP restarted before
     liberation") are reported separately and excluded from the groups.
     VP-first stays with a non-NE vasopressor still running after the VP stop are flagged and
     reported separately (they stay in the group).
  Exclusions are those of the primary analysis applied at T0b: adult, T0b inside the ICU stay,
     alive at T0b, no renal replacement therapy, no pituitary/hypothalamic/diabetes insipidus/
     brain-death/pituitary-surgery code, urine recorded in >= 12 of 24 hourly bins after T0b,
     no polyuria in the 12 h before T0b.

Paths come from config.py. Optional environment variable: M_IMP.
Outputs
  OUT_DIR/mimic_05/   results_mimic_usage_groups.xlsx, key_results.json, run_log.txt
  WORK_DIR/mimic/cohort/   analysis_ready_mimic_usage_groups.parquet,
                           usage_groups_two_group_table.parquet (two-group table rebuilt here),
                           usage_groups_classification.parquet (output of classify())

Run:  python mimic/05_usage_groups.py   (after 01_extract.sh; no output of 02_primary_analysis.py
      is read, and the caches in WORK_DIR/mimic/ext are created here when missing)
"""
import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # folder of the shared modules
import mimic_common as mc  # noqa: E402
import mimic_models as mm  # noqa: E402
from mimic_common import config, hrs, NE  # noqa: E402
from mimic_models import YS, fmt  # noqa: E402
from mimic_groups import (GAP, G4, G3, classify, anchor_cols, shifted_anchors, extra_exposure,  # noqa: E402
                          group_models, nN, smd)

warnings.filterwarnings("ignore")

TWO_GROUP_TABLE = "usage_groups_two_group_table.parquet"
CLASSIFICATION_TABLE = "usage_groups_classification.parquet"
say = mc.RunLog()


# ===========================================================================
# Tools used only by this script
def interval_polyuria(c, D0, u):
    """VP-first: polyuria between T0a(VP) and T0b (hourly bins from T0a(VP); runs completed by T0b)."""
    t0 = c.set_index("stay_id").t0a_vp
    w = D0.set_index("stay_id").weight.reindex(t0.index)
    u = u[u.stay_id.isin(t0.index)]
    m = u.assign(h=np.floor(hrs(u.charttime, u.stay_id.map(t0))), wt=u.stay_id.map(w))
    m = m[(m.h >= 0) & (m.h < 400) & (m.wt > 0)]
    hr = m.groupby(["stay_id", "h"]).agg(ml=("value", "sum"), wt=("wt", "first")).reset_index()
    hr["rate"] = hr.ml / hr.wt
    gap = c.set_index("stay_id").gap_vp_t0b_h
    res = {}
    for sid, g in hr.groupby("stay_id"):
        rr = dict(zip(g.h.astype(int), g.rate))
        hi = int(np.floor(gap.get(sid, 0)))
        res[sid] = mc.run_onset(rr, 0, hi, 3.0) if hi >= 2 else None
    s = pd.Series({k: (np.nan if v_ is None else v_) for k, v_ in res.items()}).reindex(t0.index)
    return s


def wilson(k, n):
    if n == 0:
        return "NA"
    lo, hi = proportion_confint(k, n, method="wilson")
    return "%d/%d (%.1f%%; 95%% CI %.1f–%.1f)" % (k, n, 100 * k / n, 100 * lo, 100 * hi)


CONT = [("anchor_age", "age, y"), ("weight", "weight, kg"), ("lactate_pre24", "lactate max 24 h before T0b, mmol/L"),
        ("press_h", "vasopressor time first start→T0b, h"), ("vp_dur_h", "VP infusion time before T0b, h"),
        ("idx_ep1_h", "first index episode duration, h"), ("ne_peak", "NE peak before T0b, mcg/kg/min (0 if none)"),
        ("ne_peak_users", "NE peak among NE users, mcg/kg/min"), ("vp_peak_umin", "VP peak before T0b, U/min"),
        ("fluid_pre24_l", "fluid 24 h before T0b, L"), ("uo_pre6_mlkgh", "urine 6 h before T0b, mL/kg/h"),
        ("na_pre", "serum Na last 24 h before T0b, mmol/L"), ("map_base", "MAP median 6 h before T0b, mmHg"),
        ("obs24", "recorded urine bins 0–24 h")]
BIN = [("male", "male sex"), ("sepsis", "sepsis (ICD)"), ("card_surg", "cardiac surgery"), ("cvicu", "cardiac ICU (CVICU)"),
       ("other_vaso_pre", "other vasopressor before T0b"), ("vent_t0", "invasive ventilation at T0b"),
       ("steroid_pre24", "systemic steroid 24 h before T0b"), ("inotrope_t0", "inotrope at T0b (dobutamine/milrinone)"),
       ("diur_pre24", "loop diuretic 24 h before T0b"), ("emergency", "emergency admission"),
       ("hospital_expire_flag", "hospital death")]


def table1(P, col, levels):
    rows = []
    ref = P[P[col] == levels[0]]
    for c, lab in CONT:
        r = dict(variable=lab)
        for g in levels:
            s = P[P[col] == g][c]
            r[g] = "%.2f [%.2f–%.2f]" % (s.median(), s.quantile(.25), s.quantile(.75)) if s.notna().any() else "NA"
            r[f"missing_{g}"] = "%d/%d" % (s.isna().sum(), len(s))
            if g != levels[0]:
                r[f"SMD_{g}_vs_NEonly"] = round(smd(s, ref[c]), 3)
        rows.append(r)
    for c, lab in BIN:
        r = dict(variable=lab)
        for g in levels:
            s = P[P[col] == g][c]
            r[g] = "%d/%d (%.1f%%)" % (s.sum(), len(s), 100 * s.mean()) if len(s) else "NA"
            r[f"missing_{g}"] = "0/%d" % len(s)
            if g != levels[0]:
                r[f"SMD_{g}_vs_NEonly"] = round(smd(s, ref[c]), 3)
        rows.append(r)
    for k in ("0", "1", "2", "3", "unk"):
        r = dict(variable=f"KDIGO stage before T0b = {k} (creatinine)")
        for g in levels:
            s = P[P[col] == g].kdigo_cat == k
            r[g] = "%d/%d (%.1f%%)" % (s.sum(), len(s), 100 * s.mean()) if len(s) else "NA"
            r[f"missing_{g}"] = ""
            if g != levels[0]:
                r[f"SMD_{g}_vs_NEonly"] = round(smd(s.astype(int), (ref.kdigo_cat == k).astype(int)), 3)
        rows.append(r)
    rows.append(dict(variable="APACHE IV", **{g: "not available in MIMIC-IV" for g in levels}))
    return pd.DataFrame(rows)


def q(s, ps=(.1, .2, .3, .4, .5, .6, .7, .8, .9)):
    s = s.dropna()
    return {f"p{int(p * 100)}": round(s.quantile(p), 2) for p in ps} if len(s) else {}



# ===========================================================================
def main():
    t_all = time.time()
    outdir = config.out_dir("mimic_05")
    mc.check_extracts()
    mc.build_caches(say)
    data = mc.load_all(say)
    v = data["v"]
    SH = {}

    # --- A. Reproduction of the two-group primary analysis (same functions as script 02) -----
    say("build D0 (current anchors)")
    D0, _, _ = mc.build(data, 1.0)
    PP0 = mc.populations(D0)
    P0 = D0[PP0["primary"]]
    IMPS0 = mm.impute(D0)
    o0 = {y: mm.or_mi(P0, IMPS0, y)["vp"] for y in YS}
    rep = [dict(check="current primary (script 02, T0 = end of first index episode)", group=g,
                n_stays=int((P0.grp == g).sum()), n_patients=P0[P0.grp == g].subject_id.nunique(),
                poly24=int(P0[P0.grp == g].poly24.sum()), poly8=int(P0[P0.grp == g].poly8.sum()),
                poly_8_24=int(P0[P0.grp == g].poly_8_24.sum())) for g in ("VP", "NE")]
    for y in YS:
        rep.append(dict(check=f"current primary aOR VP vs NE, {y} (MICE m={mm.M_IMP})", group="VP vs NE", estimate=fmt(o0[y])))
    say("reproduction:", rep[0], rep[1], fmt(o0["poly24"]))
    D0.to_parquet(mc.cohort_file(TWO_GROUP_TABLE))

    # --- B. Classification and T0b -------------------------------------------------
    c, ep = classify(v)
    c.to_parquet(mc.cohort_file(CLASSIFICATION_TABLE))
    say("classification:", c.ugrp.value_counts().to_dict())
    d_orig = mc.anchors(v, GAP).set_index("stay_id")
    # check: anchor_cols at t0 = T0a(VP) must reproduce anchors()
    chk = anchor_cols(v, ep, d_orig[d_orig.grp == "VP"][["t0"]])
    mism = {k: int((chk[k].fillna(-9) != d_orig.loc[chk.index, k].fillna(-9)).sum())
            for k in ["vaso_at_t0", "restart_t", "press_h", "ne_peak", "other_vaso_pre", "weight", "idx_restart_t"]}
    say("anchor_cols vs anchors() at T0a (mismatches):", mism)
    dB = shifted_anchors(v, ep, d_orig, c.set_index("stay_id").t0b)
    say("build DB (T0b anchors)")
    DB, _, _ = mc.build(data, 1.0, anchor_table=dB)
    DB = DB.merge(c, on="stay_id", how="left")
    DB.loc[DB.grp == "NE", "ugrp"] = "NE-only"
    DB = DB.merge(extra_exposure(v, data["u"], DB), on="stay_id", how="left")
    DB["ne_peak_users"] = DB.ne_peak.where(DB.ne_peak > 0)
    DB["ugrp3"] = DB.ugrp.replace({"NE-first": "NE+VP", "Simultaneous": "NE+VP", "VP-first": "NE+VP"})
    say("vasopressor running at T0b (raw rows, all drugs; must be 0):",
        int(DB.loc[DB.grp == "VP", "vaso_any_at_t0b_check"].sum()))
    ipoly = interval_polyuria(c[c.ugrp == "VP-first"], D0, data["u"])
    DB["poly_interval_onset"] = DB.stay_id.map(ipoly)
    IMPSB = mm.impute(DB)

    # populations
    base = (DB.ex_age == 0) & (DB.ex_notinicu == 0) & (DB.ex_deadpre == 0) & (DB.rrt_any == 0) & (DB.ex_neuroendo == 0)
    ne_ok = (DB.grp != "NE") | (DB.vaso_at_t0 == 0)
    DB["pop_base"] = (base & ne_ok).astype(int)
    DB["pop_eval"] = (DB.pop_base.astype(bool) & (DB.obs24 >= 12)).astype(int)
    DB["pop_main"] = (DB.pop_eval.astype(bool) & (DB.pre_poly == 0)).astype(int)
    DB.to_parquet(mc.cohort_file(mc.USAGE_TABLE))
    MAIN = DB[(DB.pop_main == 1) & DB.ugrp.isin(G4)].copy()
    ALLG = G4 + ["VP restarted before liberation"]

    # --- 1. Flow -------------------------------------------------------------------
    rows = []
    for g in ALLG:
        s = DB[DB.ugrp == g]
        m = pd.Series(True, index=s.index)
        steps = [("stays in group (classified at T0b)", None), ("age >= 18", s.ex_age == 0),
                 ("T0b within the ICU stay", s.ex_notinicu == 0), ("alive at T0b", s.ex_deadpre == 0),
                 ("no RRT during the stay", s.rrt_any == 0),
                 ("no pituitary/hypothalamic/DI dx, brain death or pituitary surgery", s.ex_neuroendo == 0)]
        if g == "NE-only":
            steps.append(("no vasopressor running at T0b (current NE-only rule)", s.vaso_at_t0 == 0))
        steps += [("urine recorded in >=12 of 24 hourly bins after T0b", s.obs24 >= 12),
                  ("new-onset: no polyuria −12..−1 h before T0b  [= analysis population]", s.pre_poly == 0)]
        prev = len(s)
        for lab, cnd in steps:
            if cnd is not None:
                m &= cnd
            rows.append(dict(group=g, step=lab, n_stays=int(m.sum()), n_patients=int(s[m].subject_id.nunique()),
                             excluded=prev - int(m.sum())))
            prev = int(m.sum())
    for g in ("VP-only", "NE+VP"):
        s = MAIN[MAIN.ugrp3 == g]
        rows.append(dict(group=g + " (3-group view)", step="analysis population", n_stays=len(s),
                         n_patients=int(s.subject_id.nunique())))
    s = MAIN[MAIN.ugrp.isin(["VP-only", "NE-first", "Simultaneous"])]
    rows.append(dict(group="union VP-only+NE-first+Simultaneous", step="analysis population", n_stays=len(s),
                     n_patients=int(s.subject_id.nunique())))
    s = MAIN[MAIN.grp == "VP"]
    rows.append(dict(group="all VP-containing (4 groups)", step="analysis population", n_stays=len(s),
                     n_patients=int(s.subject_id.nunique())))
    rows.append(dict(group="VP-first with a non-NE vasopressor also running after VP stop", step="analysis population",
                     n_stays=int(((MAIN.ugrp == "VP-first") & (MAIN.other_after_vp == 1)).sum()),
                     n_patients=int(MAIN[(MAIN.ugrp == "VP-first") & (MAIN.other_after_vp == 1)].subject_id.nunique())))
    FLOW = pd.DataFrame(rows)
    SH["01_flow"] = FLOW
    SH["01b_classification_all"] = pd.DataFrame(
        [dict(ugrp=g, n_stays=int((DB.ugrp == g).sum()),
              T0b_later_than_VPstop=int(((DB.ugrp == g) & (DB.gap_vp_t0b_h > 0)).sum()),
              other_nonNE_running_after_VPstop=int(((DB.ugrp == g) & (DB.other_after_vp == 1)).sum()))
         for g in ALLG])

    # reproduction: union of groups vs the VP group of the primary analysis
    old = set(P0.loc[P0.grp == "VP", "stay_id"])
    uni = MAIN[MAIN.ugrp.isin(["VP-only", "NE-first", "Simultaneous"])]
    new = set(uni.stay_id)
    only_new = uni[~uni.stay_id.isin(old)]
    only_old = DB[DB.stay_id.isin(old - new)]
    rr = []
    rr.append(dict(check="current primary VP stays", n=len(old)))
    rr.append(dict(check="union VP-only+NE-first+Simultaneous (T0b)", n=len(new)))
    rr.append(dict(check="in both", n=len(old & new)))
    rr.append(dict(check="union only: T0b later than VP stop (NE stopped <=1 h after VP, or other vasopressor ran on)",
                   n=int((only_new.gap_vp_t0b_h > 0).sum())))
    rr.append(dict(check="union only: T0b = VP stop (other reason)", n=int((only_new.gap_vp_t0b_h <= 0).sum())))
    for g in ("Simultaneous", "VP-only", "NE-first"):
        rr.append(dict(check=f"  union only, group {g}", n=int((only_new.ugrp == g).sum())))
    for g in only_old.ugrp.value_counts().index:
        rr.append(dict(check=f"current only: now classified {g}", n=int((only_old.ugrp == g).sum())))
    rr.append(dict(check="current only: of which excluded at T0b by new-onset/urine/ICU/alive rules",
                   n=int((~only_old.stay_id.isin(set(MAIN.stay_id))).sum())))
    for y in YS:
        rr.append(dict(check=f"union events {y}", n=int(uni[y].sum())))
    u2 = MAIN[MAIN.ugrp.isin(["NE-only", "VP-only", "NE-first", "Simultaneous"])].copy()
    u2["vpu"] = (u2.grp == "VP").astype(int)
    for y in YS:
        o = mm.or_mi(u2.assign(vp=u2.vpu), IMPSB, y)["vp"]
        rr.append(dict(check=f"union vs NE-only aOR {y} (MICE on T0b data)", n=len(u2), estimate=fmt(o)))
    ss = MAIN[(MAIN.grp == "VP") & (MAIN.gap_vp_t0b_h <= 0) & MAIN.ugrp.isin(["VP-only", "NE-first", "Simultaneous"])]
    rr.append(dict(check="subset of union with T0b = VP stop (should equal current 743)", n=len(ss),
                   estimate="events 24 h %d" % ss.poly24.sum()))
    SH["02_reproduction"] = pd.concat([pd.DataFrame(rep), pd.DataFrame(rr)], ignore_index=True)
    say(SH["02_reproduction"].to_string())

    # --- 2. Characteristics --------------------------------------------------------
    SH["03_table1_5groups"] = table1(MAIN, "ugrp", G4)
    SH["03b_table1_3groups"] = table1(MAIN, "ugrp3", G3)

    # --- 3. T0a to T0b -------------------------------------------------------------
    rows = []
    for g in ("NE-first", "Simultaneous", "VP-first", "VP-only"):
        for popn, s in (("analysis population", MAIN[MAIN.ugrp == g]), ("all classified stays", DB[DB.ugrp == g])):
            for col, lab in (("gap_t0a_t0b_h", "T0b − T0a (T0a = earlier of last NE end and VP stop)"),
                             ("gap_vp_t0b_h", "T0b − T0a(VP)"), ("diff_ne_vp_h", "last NE end − VP stop")):
                x = s[col].dropna()
                if not len(x):
                    continue
                rows.append(dict(group=g, population=popn, quantity=lab, n=len(x),
                                 median_IQR="%.2f [%.2f–%.2f]" % (x.median(), x.quantile(.25), x.quantile(.75)),
                                 **q(x)))
    SH["04_gap_T0a_T0b"] = pd.DataFrame(rows)
    # VP-first: polyuria in the interval and exclusions by the new-onset rule
    vf_all = DB[(DB.ugrp == "VP-first")]
    vf_ev = DB[(DB.ugrp == "VP-first") & (DB.pop_eval == 1)]
    k_int = int(vf_ev.poly_interval_onset.notna().sum())
    k_pre = int((vf_ev.pre_poly == 1).sum())
    k_both = int(((vf_ev.pre_poly == 1) & vf_ev.poly_interval_onset.notna()).sum())
    vfrows = [dict(item="VP-first stays passing base exclusions and urine rule at T0b (before new-onset rule)", value=len(vf_ev)),
              dict(item="  polyuria run completed between T0a(VP) and T0b", value=wilson(k_int, len(vf_ev))),
              dict(item="  excluded by new-onset rule (polyuria −12..−1 h before T0b)", value=wilson(k_pre, len(vf_ev))),
              dict(item="  both", value="%d/%d" % (k_both, len(vf_ev))),
              dict(item="  VP-first stays with T0a(VP)–T0b gap <2 h (interval polyuria not assessable)",
                   value="%d/%d" % (int((vf_ev.gap_vp_t0b_h < 2).sum()), len(vf_ev))),
              dict(item="all classified VP-first stays: interval polyuria", value=wilson(int(DB.loc[DB.ugrp == "VP-first",
                                                                                                    "poly_interval_onset"].notna().sum()), len(vf_all)))]
    for g in ("NE-first", "Simultaneous", "VP-only", "NE-only"):
        s = DB[(DB.ugrp == g) & (DB.pop_eval == 1)]
        vfrows.append(dict(item=f"{g}: excluded by new-onset rule among evaluable", value=wilson(int((s.pre_poly == 1).sum()), len(s))))
    SH["04b_VPfirst_interval"] = pd.DataFrame(vfrows)

    # --- 4. Completeness of urine charting -----------------------------------------
    rows = []
    for g in ALLG:
        rs = DB[(DB.ugrp == g) & (DB.pop_base == 1)]
        ev = rs[rs.obs24 >= 12]
        mn = MAIN[MAIN.ugrp == g]
        rows.append(dict(group=g, risk_set_before_urine_rule=len(rs), evaluable=wilson(len(ev), len(rs)),
                         bins_0_24_all="%.0f [%.0f–%.0f]" % (rs.obs24.median(), rs.obs24.quantile(.25), rs.obs24.quantile(.75)) if len(rs) else "",
                         bins_0_24_analysis="%.0f [%.0f–%.0f]" % (mn.obs24.median(), mn.obs24.quantile(.25), mn.obs24.quantile(.75)) if len(mn) else "",
                         analysis_24_of_24_bins="%d/%d" % ((mn.obs24 == 24).sum(), len(mn)),
                         analysis_ge21_bins="%d/%d" % ((mn.obs24 >= 21).sum(), len(mn))))
    SH["05_urine_completeness"] = pd.DataFrame(rows)

    # --- 5. Primary outcomes -------------------------------------------------------
    rows = []
    for y in YS:
        rows += group_models(MAIN, IMPSB, y, "ugrp", G4, label="5 groups, main risk set (all stays; earlier onset = non-event)")
    for y in YS:
        rows += group_models(MAIN, IMPSB, y, "ugrp3", G3, label="3 groups, main risk set")
    cond = MAIN[((MAIN.onset.isna()) | (MAIN.onset > 8)) & (MAIN.icuout_t > 8) & ((MAIN.death_t.isna()) | (MAIN.death_t > 8))]
    rows += group_models(cond, IMPSB, "poly_8_24", "ugrp", G4, label="5 groups, conditional >8–24 h (free of polyuria, alive and in ICU at 8 h)")
    rows += group_models(cond, IMPSB, "poly_8_24", "ugrp3", G3, label="3 groups, conditional >8–24 h")
    OUT = pd.DataFrame(rows)
    SH["06_outcomes"] = OUT
    rs_rows = []
    for g in G4 + ["NE+VP"]:
        col = "ugrp3" if g == "NE+VP" else "ugrp"
        rs_rows.append(dict(group=g, main_risk_set_N=int((MAIN[col] == g).sum()), conditional_8h_risk_set_N=int((cond[col] == g).sum()),
                            poly24=nN(MAIN[MAIN[col] == g], "poly24"), poly8=nN(MAIN[MAIN[col] == g], "poly8"),
                            poly_8_24=nN(MAIN[MAIN[col] == g], "poly_8_24"),
                            poly_8_24_conditional=nN(cond[cond[col] == g], "poly_8_24")))
    SH["06b_risk_sets"] = pd.DataFrame(rs_rows)
    say("outcomes done")

    # --- 6. Na >= 150 --------------------------------------------------------------
    rows, nrows = [], []
    for g in G4 + ["NE+VP"]:
        col = "ugrp3" if g == "NE+VP" else "ugrp"
        s = MAIN[MAIN[col] == g]
        nrows.append(dict(group=g, analysis_N=len(s), excluded_pre_Na_ge150=int((s.na_max_pre24 >= 150).sum()),
                          excluded_no_pre_Na=int(s.na_max_pre24.isna().sum()),
                          Na_new_onset_risk_set=int((s.na_max_pre24 < 150).sum()),
                          Na_tests_0_72h_median_IQR="%.0f [%.0f–%.0f]" % (s.na_n72.median(), s.na_n72.quantile(.25), s.na_n72.quantile(.75)),
                          Na_tests_mean=round(s.na_n72.mean(), 2)))
    SH["07a_Na_counts"] = pd.DataFrame(nrows)
    NA = MAIN[MAIN.na_max_pre24 < 150].copy()
    NA["log_na_n72"] = np.log1p(NA.na_n72)
    rows += group_models(NA, IMPSB, "na150_72", "ugrp", G4, label="Na>=150 72 h, 5 groups")
    rows += group_models(NA, IMPSB, "na150_72", "ugrp", G4, cov=mm.COV_MI + "+log_na_n72", label="Na>=150 72 h, 5 groups, + log(1+Na tests)")
    rows += group_models(NA, IMPSB, "na150_72", "ugrp3", G3, label="Na>=150 72 h, 3 groups")
    rows += group_models(NA, IMPSB, "na150_72", "ugrp3", G3, cov=mm.COV_MI + "+log_na_n72", label="Na>=150 72 h, 3 groups, + log(1+Na tests)")
    SH["07b_Na150_models"] = pd.DataFrame(rows)

    # --- 7. Restart ----------------------------------------------------------------
    rows = []
    for g in G4 + ["NE+VP"]:
        col = "ugrp3" if g == "NE+VP" else "ugrp"
        s = MAIN[MAIN[col] == g]
        rows.append(dict(group=g, restart24_any_vasopressor=nN(s, "restart24"),
                         restart2h="%d/%d" % ((s.restart_t <= 2).sum(), len(s)),
                         VP_restart24=("%d/%d" % ((s.idx_restart_t <= 24).sum(), len(s))) if g != "NE-only" else "",
                         poly24_prerestart=nN(s, "poly24_prerestart"), poly8_prerestart=nN(s, "poly8_prerestart"),
                         poly_8_24_prerestart=nN(s, "poly_8_24_prerestart")))
    SH["08a_restart"] = pd.DataFrame(rows)
    rows = []
    for y in ("poly24_prerestart", "poly8_prerestart", "poly_8_24_prerestart"):
        rows += group_models(MAIN, IMPSB, y, "ugrp", G4, label="sensitivity: polyuria counted only before vasopressor restart, 5 groups")
        rows += group_models(MAIN, IMPSB, y, "ugrp3", G3, label="sensitivity: polyuria before restart, 3 groups")
    SH["08b_prerestart_models"] = pd.DataFrame(rows)
    say("Na/restart done")

    # --- 8. VP-stop anchor (exploratory, descriptive only) -------------------------
    rows = []
    gmap = c.set_index("stay_id").ugrp
    D0["ugrp"] = D0.stay_id.map(gmap)
    for g in ("VP-first", "NE-first", "Simultaneous", "VP-only"):
        s0 = D0[D0.ugrp == g]
        b0 = s0[(s0.ex_age == 0) & (s0.ex_notinicu == 0) & (s0.ex_deadpre == 0) & (s0.rrt_any == 0) & (s0.ex_neuroendo == 0)]
        e0 = b0[b0.obs24 >= 12]
        a0 = e0[e0.pre_poly == 0]
        r = dict(group=g, anchor="VP stop T0a(VP)", classified=len(s0), base_exclusions=len(b0), evaluable=len(e0),
                 excluded_new_onset=int((e0.pre_poly == 1).sum()), analysis_N=len(a0),
                 patients=int(a0.subject_id.nunique()))
        for y in YS:
            r[y] = wilson(int(a0[y].sum()), len(a0))
        rows.append(r)
    SH["09_VPstop_anchor"] = pd.DataFrame(rows)
    vfm = MAIN[MAIN.ugrp == "VP-first"]
    SH["09_VPstop_anchor"] = pd.concat([SH["09_VPstop_anchor"], pd.DataFrame([
        dict(group="VP-first (T0b analysis population)", anchor="interval T0a(VP)–T0b",
             analysis_N=len(vfm), interval_polyuria_T0aVP_to_T0b=wilson(int(vfm.poly_interval_onset.notna().sum()), len(vfm)))])],
        ignore_index=True)
    # reported separately: VP-first with a non-NE vasopressor continuing / VP restarted
    rows = []
    for lab, s in (("VP-first, non-NE vasopressor also running after VP stop", MAIN[(MAIN.ugrp == "VP-first") & (MAIN.other_after_vp == 1)]),
                   ("VP-first, NE only after VP stop", MAIN[(MAIN.ugrp == "VP-first") & (MAIN.other_after_vp == 0)]),
                   ("VP restarted before liberation (excluded from 4 groups)", DB[(DB.ugrp == "VP restarted before liberation") & (DB.pop_main == 1)]),
                   ("VP-only with T0b > VP stop (other vasopressor ran on)", MAIN[(MAIN.ugrp == "VP-only") & (MAIN.gap_vp_t0b_h > 0)])):
        r = dict(subset=lab, N=len(s), patients=int(s.subject_id.nunique()))
        for y in YS + ["na150_72", "restart24"]:
            r[y] = nN(s, y) if len(s) else "0/0"
        rows.append(r)
    SH["10_separate_subsets"] = pd.DataFrame(rows)

    # difference between the "running" rules (two-group analysis: raw rate > 0 rows; T0b: per-drug episodes joined at 1 h)
    epx = ep.merge(DB[["stay_id", "t0", "grp"]], on="stay_id")
    epx = epx[(epx.start <= epx.t0) & (epx.end > epx.t0) & (epx.itemid != NE)]
    ne_ep_run = set(epx.loc[epx.grp == "NE", "stay_id"])
    notes = [dict(item="current primary VP stays whose T0b is later than VP stop (another vasopressor's joined episode spans VP stop across a <=1 h gap in records)",
                  n="%d/%d" % (int((DB.stay_id.isin(old) & (DB.gap_vp_t0b_h > 0)).sum()), len(old))),
             dict(item="NE-only analysis stays with another vasopressor's joined episode spanning T0 (kept, current raw-row rule)",
                  n="%d/%d" % (int(MAIN[MAIN.ugrp == "NE-only"].stay_id.isin(ne_ep_run).sum()), int((MAIN.ugrp == "NE-only").sum()))),
             dict(item="NE-only stays excluded because another vasopressor was running at T0 (not shifted to a later liberation time; asymmetric with VP-containing stays)",
                  n=str(int(FLOW.loc[(FLOW.group == "NE-only") & FLOW.step.str.startswith("no vasopressor running"), "excluded"].iloc[0]))),
             dict(item="Simultaneous, base exclusions passed: died within 24 h of T0b",
                  n="%d/%d" % (int(((DB.ugrp == "Simultaneous") & (DB.pop_base == 1) & (DB.death_t <= 24)).sum()),
                               int(((DB.ugrp == "Simultaneous") & (DB.pop_base == 1)).sum()))),
             dict(item="Simultaneous, base exclusions passed: ICU discharge within 24 h of T0b",
                  n="%d/%d" % (int(((DB.ugrp == "Simultaneous") & (DB.pop_base == 1) & (DB.icuout_t <= 24)).sum()),
                               int(((DB.ugrp == "Simultaneous") & (DB.pop_base == 1)).sum())))]
    for g in G4:
        b_ = DB[(DB.ugrp == g) & (DB.pop_base == 1)]
        notes.append(dict(item=f"{g}, base exclusions passed: died within 24 h of T0b", n="%d/%d" % (int((b_.death_t <= 24).sum()), len(b_))))
    SH["11_rule_notes"] = pd.DataFrame(notes)

    # README
    SH = {"00_README": pd.DataFrame([
        ("script", "05_usage_groups.py; same functions as 02_primary_analysis.py (episodes 1 h, run rule, covariates, MICE m=%d, GLM binomial with subject-clustered SE)" % mm.M_IMP),
        ("anchor", "T0b = liberation from ALL vasopressors (NE, VP, phenylephrine, epinephrine, dopamine, angiotensin II); running at t = start <= t < end of a per-drug joined episode"),
        ("groups", "NE-only (= NE group of script 02); VP-only; NE+VP split by last NE end − VP stop: NE-first < −1 h, Simultaneous |·| <= 1 h, VP-first > +1 h; VP restarted before T0b excluded"),
        ("population", "adult, T0b in ICU, alive at T0b, no RRT in stay, no pituitary/hypothalamic/DI/brain-death/pituitary-surgery codes, urine >=12/24 bins after T0b, no polyuria −12..−1 h before T0b; NE-only additionally no vasopressor running at T0b (unchanged rule)"),
        ("covariates", mm.COV_MI),
        ("sparse groups", "when a group has <10 events, its OR is from Firth penalised logistic regression (MI-pooled, model-based SE) or 'not estimable' if 0 events"),
        ("status", "post hoc, exploratory"),
    ], columns=["item", "description"]), **SH}
    xl = os.path.join(outdir, "results_mimic_usage_groups.xlsx")
    with pd.ExcelWriter(xl) as xw:
        for k, t in SH.items():
            t.to_excel(xw, sheet_name=k[:31], index=False)
    DB.to_parquet(mc.cohort_file(mc.USAGE_TABLE))
    with open(os.path.join(outdir, "key_results.json"), "w") as f:
        json.dump({"flow": FLOW.to_dict("records")}, f, default=str, indent=1)
    say("total %.0f s" % (time.time() - t_all))
    say.write(os.path.join(outdir, "run_log.txt"))


if __name__ == "__main__":
    main()