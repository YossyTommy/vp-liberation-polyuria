#!/usr/bin/env python3
"""
08_sodium_harmonized.py - MIMIC-IV v3.1, serum sodium >= 150 mmol/L within 72 h on the
denominator harmonized with eICU-CRD.

  unharmonized   all sodium values in the 24 h before T0 < 150 mmol/L (stays without a sodium
                 measurement in the 72 h after T0 count as non-events)
  harmonized     the above and >= 1 sodium measurement in the 72 h after T0 (the definition used
                 for eICU-CRD); this is the reported analysis

Models, covariates, imputation (impute of mimic_models.py, seed 2026) and clustered standard
errors are those of scripts 02, 05 and 06; only the denominator changes. As in
06_fluid_balance.py, every model merges the imputations with with_imp_fo (mimic_fluid.py): it
adds the imputed FO% for the analyses of part (2), which use the FO%-augmented imputations,
and is identical to with_imp for all other analyses. Each analysis is first estimated on the
unharmonized denominator and checked against its expected value, then estimated on the
harmonized denominator.

Inputs   analysis-ready cohort tables of scripts 02 and 05, the FO% table of script 06
         (two-group anchor) and vaso2.csv (switch flag, as switch_flags in 07_vpstop_timing.py)
Outputs  OUT_DIR/mimic_08/sodium_harmonized.csv, reproduction_checks.csv
Optional environment variables: NBOOT (default 200), M_IMP (10).

Run:  python mimic/08_sodium_harmonized.py   (after scripts 02, 05 and 06)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # folder of the shared modules
import mimic_common as mc  # noqa: E402
import mimic_models as mm  # noqa: E402
from mimic_common import config, VP  # noqa: E402
from mimic_groups import G4, group_models  # noqa: E402
from mimic_fluid import COV_FO, with_imp_fo, impute_fo, ato_analysis  # noqa: E402

warnings.filterwarnings("ignore")

NBOOT = int(os.environ.get("NBOOT", 200))
ROWS, REP = [], []


def unharmonized_set(d):
    return d[d.na_max_pre24 < 150]


def harmonized_set(d):
    return d[(d.na_max_pre24 < 150) & (d.na_n72 >= 1)]


def nN(d, y, col, g):
    s = d[d[col] == g]
    return int(s[y].sum()), len(s)


def add(analysis, denom, d, y, col, g1, g0, o, note=""):
    k1, n1 = nN(d, y, col, g1)
    k0, n0 = nN(d, y, col, g0)
    ROWS.append(dict(analysis=analysis, denominator=denom, outcome=y, group=g1, ref=g0, k1=k1, n1=n1, k0=k0, n0=n0,
                     aOR="%.2f (%.2f–%.2f)" % (o["OR"], o["lo"], o["hi"]), est=o["OR"], lo=o["lo"], hi=o["hi"], note=note))


def rep(check, value, expected):
    REP.append(dict(check=check, value=value, expected=expected, match=str(value).startswith(expected)))


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    outdir = config.out_dir("mimic_08")
    D0 = pd.read_parquet(mc.cohort_file(mc.PRIMARY_TABLE))
    DB = pd.read_parquet(mc.cohort_file(mc.USAGE_TABLE))
    IMPS0 = mm.impute(D0)                      # as in scripts 02 / 06 / 07
    IMPSB = mm.impute(DB)                      # as in scripts 05 / 07
    say("imputed")
    P0 = D0[D0.pop_primary == 1].copy()
    P0["ugrp"] = P0.stay_id.map(DB.set_index("stay_id").ugrp)
    P0["t0a"] = P0.stay_id.map(DB.set_index("stay_id").t0a)
    MAINB = DB[(DB.pop_main == 1) & DB.ugrp.isin(G4)].copy()
    for lab, fn in (("unharmonized", unharmonized_set), ("harmonized", harmonized_set)):
        s = fn(P0)
        say(lab, "2-group Na set VP/NE:", nN(s, "na150_72", "grp", "VP"), nN(s, "na150_72", "grp", "NE"))

    # ---- (1) two-group primary, detection bias, Na rise, early polyuria ----------------
    for lab, fn in (("unharmonized", unharmonized_set), ("harmonized", harmonized_set)):
        d = fn(P0).copy()
        d["log_na_n72"] = np.log1p(d.na_n72)
        d["na150_fixed"] = ((d.na_fix24 >= 150) | (d.na_fix48 >= 150)).astype(int)
        d["na150_chem"] = (d.na_peak_72_chem >= 150).astype(int)
        specs = [("primary", d, "na150_72", mm.COV_MI),
                 ("+ log(1 + number of Na tests 0–72 h)", d, "na150_72", mm.COV_MI + "+log_na_n72"),
                 ("restricted to ≥3 Na tests 0–72 h", d[d.na_n72 >= 3], "na150_72", mm.COV_MI),
                 ("fixed-time windows (18–30 h, 42–54 h; both measured)", d[d.na_fix24.notna() & d.na_fix48.notna()], "na150_fixed", mm.COV_MI),
                 ("chemistry (indirect ISE) Na only", d, "na150_chem", mm.COV_MI),
                 ("Na rise ≥8 mmol/L within 72 h (within Na set)", d[d.na_pre.notna()], "na_rise8_72", mm.COV_MI)]
        for nm, x, y, cov in specs:
            o = mm.or_mi(x, IMPS0, y, cov=cov, with_imp=with_imp_fo)["vp"]
            add("2grp: " + nm, lab, x, y, "grp", "VP", "NE", o)
        o2 = mm.or_mi(d, IMPS0, "na150_72", expo="vp+poly8", terms=["vp", "poly8"], with_imp=with_imp_fo)
        add("2grp: + onset 0–8 h (VP term)", lab, d, "na150_72", "grp", "VP", "NE", o2["vp"])
        add("2grp: + onset 0–8 h (poly8 term)", lab, d, "na150_72", "grp", "VP", "NE", o2["poly8"])
        o1 = mm.or_mi(d, IMPS0, "na150_72", with_imp=with_imp_fo)["vp"]
        ROWS.append(dict(analysis="2grp: change in log OR with + onset 0–8 h, %", denominator=lab,
                         est=100 * (np.log(o2["vp"]["OR"]) - np.log(o1["OR"])) / np.log(o1["OR"])))
        for g in ("VP", "NE"):
            for pv in (1, 0):
                s = d[(d.grp == g) & (d.poly8 == pv)]
                ROWS.append(dict(analysis=f"2grp: Na≥150 by poly8={pv}", denominator=lab, group=g, k1=int(s.na150_72.sum()), n1=len(s)))
            s = d[d.grp == g]
            ROWS.append(dict(analysis="2grp: Na tests 0–72 h mean", denominator=lab, group=g, est=s.na_n72.mean(), n1=len(s)))
        say("(1)", lab)
    # Na rise on its original denominator (all primary stays with a pre-T0 Na)
    x = P0[P0.na_pre.notna()]
    o = mm.or_mi(x, IMPS0, "na_rise8_72", with_imp=with_imp_fo)["vp"]
    add("2grp: Na rise ≥8 (all stays with pre-T0 Na)", "unharmonized", x, "na_rise8_72", "grp", "VP", "NE", o)

    # ---- (2) FO% adjustment and ATO (as 06_fluid_balance.py: FO% imputed, COV_FO, FO% in the PS) ---
    F0 = pd.read_parquet(mc.cohort_file(mc.FLUID_TABLE_2GRP))
    fcols = [c for c in F0.columns if c not in ("subject_id", "hadm_id", "grp", "t0", "intime", "weight", "pop_primary")]
    D0f = D0.drop(columns=[c for c in fcols if c in D0.columns and c != "stay_id"]).merge(F0[fcols], on="stay_id", how="left")
    IMPS0f = impute_fo(D0f)
    P0f = D0f[D0f.pop_primary == 1].copy()
    say("FO imputed")
    for lab, fn in (("unharmonized", unharmonized_set), ("harmonized", harmonized_set)):
        d = fn(P0f)
        o = mm.or_mi(d, IMPS0f, "na150_72", cov=COV_FO, with_imp=with_imp_fo)["vp"]
        add("2grp: + cumulative FO% (B: COV_MI + FO%)", lab, d, "na150_72", "grp", "VP", "NE", o)
        res, _, _ = ato_analysis(d, IMPS0f, ["na150_72"], COV_FO, NBOOT, with_imp_fo)
        r = res["na150_72"]
        add("2grp: ATO overlap weighting, PS incl FO%", lab, d, "na150_72", "grp", "VP", "NE",
            dict(OR=np.exp(r["lor"]), lo=np.exp(r["lor_lo"]), hi=np.exp(r["lor_hi"])),
            note="weighted risk VP %.1f%% NE %.1f%%; RD %.1f (%.1f to %.1f); nboot %d" % (100 * r["p1"], 100 * r["p0"], 100 * r["rd"], 100 * r["rd_lo"], 100 * r["rd_hi"], r["nboot"]))
        say("(2)", lab)

    # ---- (3) five groups (T0b) ---------------------------------------------------------
    for lab, fn in (("unharmonized", unharmonized_set), ("harmonized", harmonized_set)):
        d = fn(MAINB)
        for r in group_models(d, IMPSB, "na150_72", "ugrp", G4, with_imp=with_imp_fo):
            ROWS.append(dict(analysis="5grp: " + r["group"] + " vs NE-only", denominator=lab, outcome="na150_72", group=r["group"], ref="NE-only",
                             k1=int(d[d.ugrp == r["group"]].na150_72.sum()), n1=int((d.ugrp == r["group"]).sum()),
                             k0=int(d[d.ugrp == "NE-only"].na150_72.sum()), n0=int((d.ugrp == "NE-only").sum()),
                             aOR=r["adjusted_OR"], est=r.get("adjusted_est"), lo=r.get("adjusted_lo"), hi=r.get("adjusted_hi")))
        say("(3)", lab)

    # ---- (4) NE->VP switch stays removed (definition of switch_flags in 07_vpstop_timing.py) ------
    v = pd.read_csv(os.path.join(mc.DATA, "vaso2.csv"), header=None, names=["subject_id", "stay_id", "starttime", "endtime", "itemid", "amount", "rate", "rateuom", "weight"])
    v = v[v.stay_id.notna() & (v.itemid == VP)].copy()
    v["stay_id"] = v.stay_id.astype("int64")
    v["starttime"] = pd.to_datetime(v.starttime, errors="coerce")
    v["endtime"] = pd.to_datetime(v.endtime, errors="coerce")
    v = v.dropna(subset=["starttime", "endtime"])
    vp_first = v.groupby("stay_id").starttime.min()
    for dd in (P0, MAINB):
        dd["sw_ne_vp"] = ((dd.ugrp == "NE-first") & (dd.stay_id.map(vp_first) > dd.t0a)).astype(int)
    rep("switch count, primary VP group (NE→VP)", "%d/%d" % (P0[P0.grp == "VP"].sw_ne_vp.sum(), (P0.grp == "VP").sum()), "31/743")
    rep("switch count, NE-first at T0b", "%d/%d" % (MAINB[MAINB.ugrp == "NE-first"].sw_ne_vp.sum(), (MAINB.ugrp == "NE-first").sum()), "45/560")
    P0x = P0[~((P0.grp == "VP") & (P0.sw_ne_vp == 1))]
    MBx = MAINB[~((MAINB.ugrp == "NE-first") & (MAINB.sw_ne_vp == 1))]
    for lab, fn in (("unharmonized", unharmonized_set), ("harmonized", harmonized_set)):
        d = fn(P0x)
        o = mm.or_mi(d, IMPS0, "na150_72", with_imp=with_imp_fo)["vp"]
        add("switch: 2grp excl NE→VP switch", lab, d, "na150_72", "grp", "VP", "NE", o)
        d = fn(MBx)
        for r in group_models(d, IMPSB, "na150_72", "ugrp", G4, with_imp=with_imp_fo):
            if r["group"] == "NE-first":
                ROWS.append(dict(analysis="switch: 5grp NE-first excl switch vs NE-only", denominator=lab, group="NE-first",
                                 k1=int(d[d.ugrp == "NE-first"].na150_72.sum()), n1=int((d.ugrp == "NE-first").sum()),
                                 aOR=r["adjusted_OR"], est=r.get("adjusted_est"), lo=r.get("adjusted_lo"), hi=r.get("adjusted_hi")))
        say("(4)", lab)

    R = pd.DataFrame(ROWS)
    R.to_csv(os.path.join(outdir, "sodium_harmonized.csv"), index=False)
    # expected values on the unharmonized denominator
    g = lambda a, den: R[(R.analysis == a) & (R.denominator == den)].aOR.iloc[0]
    for a, k in (("2grp: primary", "2.19 (1.51–3.19)"), ("2grp: + log(1 + number of Na tests 0–72 h)", "1.91 (1.30–2.81)"),
                 ("2grp: restricted to ≥3 Na tests 0–72 h", "2.21 (1.52–3.22)"), ("2grp: chemistry (indirect ISE) Na only", "2.13 (1.46–3.10)"),
                 ("2grp: fixed-time windows (18–30 h, 42–54 h; both measured)", "1.71 (0.97–3.02)"),
                 ("2grp: Na rise ≥8 (all stays with pre-T0 Na)", "2.26 (1.80–2.85)"),
                 ("2grp: + onset 0–8 h (VP term)", "2.17 (1.49–3.15)"), ("2grp: + onset 0–8 h (poly8 term)", "1.36 (0.86–2.17)"),
                 ("2grp: + cumulative FO% (B: COV_MI + FO%)", "2.39 (1.63–3.50)"), ("2grp: ATO overlap weighting, PS incl FO%", "1.95 (1.39–2.64)"),
                 ("5grp: NE-first vs NE-only", "2.44 (1.62–3.69)"), ("5grp: VP-only vs NE-only", "1.60 (0.75–3.42)"),
                 ("5grp: Simultaneous vs NE-only", "2.81 (1.48–5.31)"), ("5grp: VP-first vs NE-only", "2.09 (1.37–3.18)"),
                 ("switch: 2grp excl NE→VP switch", "2.26 (1.55–3.31)"), ("switch: 5grp NE-first excl switch vs NE-only", "2.64 (1.74–4.01)")):
        rep("unharmonized: " + a, g(a, "unharmonized"), k)
    rep("harmonized primary", g("2grp: primary", "harmonized"), "2.27 (1.56–3.29)")
    rep("harmonized + onset 0–8 h", g("2grp: + onset 0–8 h (VP term)", "harmonized"), "2.24 (1.54–3.25)")
    RP = pd.DataFrame(REP)
    RP.to_csv(os.path.join(outdir, "reproduction_checks.csv"), index=False)
    print(RP.to_string())


if __name__ == "__main__":
    main()
