#!/usr/bin/env python3
"""
10_early_late.py - MIMIC-IV v3.1, exploratory analyses requested after v15 (not part of the
manuscript analyses).

  A  early (onset 0-8 h) vs late (>8-24 h) polyuria, VP vs NE, among the stays of the primary
     population with polyuria within 24 h: sodium, haemodynamics, co-interventions, urine and
     background; bootstrap differences for VP-early vs VP-late and VP-early vs NE-early; linear
     model of the sodium rise with group x timing interaction
  B  three-level outcome within 24 h: no polyuria / polyuria without sodium rise / polyuria with
     sodium rise; multinomial logistic regression (reference = no polyuria)

Definitions taken unchanged from v15: cohort, groups, T0, polyuria (onset), covariates
(mimic_models.COV_MI) and imputation (mimic_models.impute, seed 2026) come from the cohort table
of script 02 and the modules mimic_common / mimic_models; the sodium handling is that of
08_sodium_harmonized.py (items 50983/52623/50824; baseline = last value in (-24, 0] h; peak = max
in (0, 72] h; denominator = all pre-T0 24-h values < 150 and >= 1 value in 0-72 h).

New variables defined here (not used in v15)
  map_min24        lowest hourly median MAP in hours 0-23 after T0 (map.csv, values 10-200 mm Hg);
                   map_min24_raw = lowest single value
  restart24/48     vasopressor restart within 24 / 48 h (restart_t of v15)
  ddavp_any48 / ddavp_ad_48   desmopressin order / antidiuretic desmopressin 0-48 h (v15 columns)
  fw_any72         free water in [0, 72) h: inputevents 225797 Free Water (enteral) and 220949
                   Dextrose 5% charted as "Continuous IV" or "Bolus" (drug carriers, "Continuous Med" /
                   "Drug Push", excluded); mL, amount > 0, apportioned with mimic_fluid.win_in
  loop6            IV loop diuretic (diuretic_iv.csv) starting in [onset - 6 h, onset]
  bal_pre24/post24 input - output (mimic_fluid.prep_io conventions), windows [onset - 24 h, onset)
                   and [onset, onset + 24 h); missing without any input or output row; mL/kg
  uo_max_rate24    highest hourly urine bin (mL/kg/h) in bins 0-23 (mimic_common.hourly_bins)

Outputs  OUT_DIR/mimic_10/*.csv (aggregates only; cells < 10 shown as "<10")
         WORK_DIR/mimic/early_late/mimic_early_late_stays.parquet (row level)
Run:     python mimic/10_early_late.py   (after scripts 02 and 07; 07 writes ext/oe_pressorstays.parquet)
Optional environment variables: NBOOT_EL (bootstrap replicates, default 2000), M_IMP (10).
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mimic_common as mc  # noqa: E402
import mimic_models as mm  # noqa: E402
import mimic_fluid as mf  # noqa: E402
from mimic_common import config, NA_ITEMS, hrs  # noqa: E402

warnings.filterwarnings("ignore")
NBOOT = int(os.environ.get("NBOOT_EL", 2000))
SEED = 20261117
FW_D5W, FW_WATER = 220949, 225797
FW_D5W_CATS = {"Continuous IV", "Bolus"}       # "Continuous Med" / "Drug Push" rows are drug carriers
GROUPS = ["VP early", "VP late", "NE early", "NE late"]
T0_START = time.time()
H = pd.Timedelta(hours=1)


def say(*a):
    print("%5.0fs" % (time.time() - T0_START), *a, flush=True)


# ----------------------------------------------------------------------------------------
# shared helpers (identical in eicu/08_early_late.py)
def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def small(k, n=None):
    """True when a count (or its complement in n) is below 10."""
    return k < 10 or (n is not None and n - k < 10)


def fmt_prop(k, n):
    if n < 10:
        return "n<10", np.nan, np.nan, np.nan
    if small(k, n):
        return ("<10" if k < 10 else "masked (n-k <10)"), np.nan, np.nan, np.nan
    lo, hi = wilson(k, n)
    return "%d/%d (%.1f%%; %.1f–%.1f)" % (k, n, 100 * k / n, 100 * lo, 100 * hi), 100 * k / n, 100 * lo, 100 * hi


def fmt_med(x, dec=1):
    x = pd.Series(x).dropna()
    if len(x) < 10:
        return "n<10", np.nan, np.nan, np.nan, len(x)
    q = x.quantile([.25, .5, .75]).values
    f = "%%.%df [%%.%df–%%.%df]" % (dec, dec, dec)
    return f % (q[1], q[0], q[2]), q[1], q[0], q[2], len(x)


def describe(df, gcol, spec, groups):
    rows = []
    for v, lab, kind, dec in spec:
        r = dict(variable=v, label=lab, kind=kind)
        for g in groups:
            s = df.loc[df[gcol] == g, v]
            if kind == "bin":
                s = s.dropna()
                txt, p, lo, hi = fmt_prop(int(s.sum()), len(s))
                r[g] = txt
                r[g + "_pct"], r[g + "_lo"], r[g + "_hi"] = p, lo, hi
                r[g + "_n"] = len(s) if len(s) >= 10 else "<10"
            else:
                txt, m, q1, q3, n = fmt_med(s, dec)
                r[g] = txt
                r[g + "_median"], r[g + "_q1"], r[g + "_q3"] = m, q1, q3
                r[g + "_n"] = n if n >= 10 else "<10"
        rows.append(r)
    return pd.DataFrame(rows)


def boot_contrasts(df, gcol, spec, pairs, nboot=NBOOT, seed=SEED):
    """Difference (group a - group b) of medians (continuous) or proportions (binary, percentage
    points) with percentile bootstrap 95% CIs; stays resampled with replacement within each group."""
    rng = np.random.default_rng(seed)
    vars_ = [s[0] for s in spec]
    kinds = {s[0]: s[2] for s in spec}
    arrs = {g: df.loc[df[gcol] == g, vars_].to_numpy(float) for g in set(sum(map(list, pairs), []))}

    def stat(a):
        out = []
        for j, v in enumerate(vars_):
            x = a[:, j]
            x = x[~np.isnan(x)]
            if len(x) == 0:
                out.append(np.nan)
            elif kinds[v] == "bin":
                out.append(100 * x.mean())
            else:
                out.append(np.median(x))
        return np.array(out)

    rows = []
    for a, b in pairs:
        A, B = arrs[a], arrs[b]
        est = stat(A) - stat(B)
        bs = np.empty((nboot, len(vars_)))
        for i in range(nboot):
            bs[i] = stat(A[rng.integers(0, len(A), len(A))]) - stat(B[rng.integers(0, len(B), len(B))])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        for j, (v, lab, kind, dec) in enumerate(spec):
            na_, nb_ = (~np.isnan(A[:, j])).sum(), (~np.isnan(B[:, j])).sum()
            ka = np.nansum(A[:, j]) if kind == "bin" else np.nan
            kb = np.nansum(B[:, j]) if kind == "bin" else np.nan
            masked = na_ < 10 or nb_ < 10 or (kind == "bin" and (small(ka, na_) or small(kb, nb_)))
            unit = "percentage points" if kind == "bin" else "difference in medians"
            rows.append(dict(contrast=f"{a} - {b}", variable=v, label=lab, measure=unit,
                             diff=np.nan if masked else est[j], lo=np.nan if masked else lo[j],
                             hi=np.nan if masked else hi[j],
                             text="masked (cell <10)" if masked else ("%.*f (%.*f to %.*f)" % (dec, est[j], dec, lo[j], dec, hi[j])),
                             n_a=na_ if na_ >= 10 else "<10", n_b=nb_ if nb_ >= 10 else "<10", nboot=nboot))
    return pd.DataFrame(rows)


def box_stats(df, gcol, vars_, groups):
    rows = []
    for v in vars_:
        for g in groups:
            x = df.loc[df[gcol] == g, v].dropna()
            if len(x) < 10:
                rows.append(dict(variable=v, group=g, n="<10"))
                continue
            q = x.quantile([.1, .25, .5, .75, .9]).values
            rows.append(dict(variable=v, group=g, n=len(x), p10=q[0], q1=q[1], median=q[2], q3=q[3], p90=q[4],
                             mean=x.mean(), sd=x.std()))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# MIMIC-specific estimation (conventions of mimic_models: imputations of mm.impute merged with
# mm.with_imp, SE clustered on subject_id, Rubin's rules of mm.rubin)
def ols_interaction(d, imps, y, label):
    d = d.dropna(subset=[y]).copy()
    names = ["vp", "early", "vp:early"]
    combos = {"interaction (VP x early)": {"vp:early": 1},
              "VP early - VP late (early + interaction)": {"early": 1, "vp:early": 1},
              "VP early - NE early (VP + interaction)": {"vp": 1, "vp:early": 1},
              "NE early - NE late (early)": {"early": 1},
              "VP late - NE late (VP)": {"vp": 1}}
    acc = {k: ([], []) for k in combos}
    for imp in imps:
        x = mm.with_imp(d, imp)
        r = smf.ols(f"{y} ~ vp * early + {mm.COV_MI}", x).fit(cov_type="cluster", cov_kwds={"groups": x.subject_id.values})
        b, V = r.params, r.cov_params()
        for k, w in combos.items():
            L = pd.Series(0.0, index=b.index)
            for n, c in w.items():
                L[n] = c
            acc[k][0].append(float(L @ b))
            acc[k][1].append(float(L @ V @ L))
    rows = []
    for k in combos:
        q, se, lo, hi, p = mm.rubin(*acc[k])
        rows.append(dict(analysis=label, term=k, est=q, se=se, lo=lo, hi=hi, p=p, n=len(d),
                         n_patients=d.subject_id.nunique(),
                         method="OLS, MICE m=%d, SE clustered on subject_id" % len(imps)))
    return rows


def mnlogit_vp(d, imps, ycat, label):
    d = d.dropna(subset=[ycat]).copy()
    acc = {1: ([], []), 2: ([], [])}
    for imp in imps:
        x = mm.with_imp(d, imp)
        r = smf.mnlogit(f"{ycat} ~ vp + {mm.COV_MI}", x).fit(disp=0, maxiter=300, method="newton", cov_type="cluster",
                                                            cov_kwds={"groups": x.subject_id.values})
        b = r.params.loc["vp"].values
        V = r.cov_params()
        k1 = [i for i in V.index if i[1] == "vp"]
        Vv = V.loc[k1, k1].values
        for j, c in enumerate((1, 2)):
            acc[c][0].append(b[j])
            acc[c][1].append(Vv[j, j])
    rows = []
    for c in (1, 2):
        q, se, lo, hi, p = mm.rubin(*acc[c])
        rows.append(dict(analysis=label, category=c, logOR=q, se=se, OR=np.exp(q), lo=np.exp(lo), hi=np.exp(hi), p=p,
                         n=len(d), method="multinomial logit, MICE m=%d, SE clustered on subject_id" % len(imps)))
    return rows


# ----------------------------------------------------------------------------------------
def main():
    out = config.out_dir("mimic_10")
    wk = config.work_dir("mimic", "early_late")
    D0 = pd.read_parquet(mc.cohort_file(mc.PRIMARY_TABLE))

    # ---- 0. reproduction of v15 values ----------------------------------------------------
    P = D0[D0.pop_primary == 1].copy()
    S = P[(P.na_max_pre24 < 150) & (P.na_n72 >= 1)]            # harmonized denominator (script 08)
    rep = []

    def chk(item, val, exp):
        rep.append(dict(check=item, value=val, expected=exp, match=val == exp))
    chk("VP stays, primary", str(int((P.grp == "VP").sum())), "743")
    chk("NE stays, primary", "{:,}".format(int((P.grp == "NE").sum())), "4,667")
    for g, e in (("VP", "30.1"), ("NE", "19.4")):
        chk(f"24-h polyuria % {g}", "%.1f" % (100 * P.loc[P.grp == g, "poly24"].mean()), e)
    for g, e in (("VP", "9.5"), ("NE", "4.8")):
        chk(f"Na>=150 within 72 h % {g} (harmonized)", "%.1f" % (100 * S.loc[S.grp == g, "na150_72"].mean()), e)
    REP = pd.DataFrame(rep)
    REP.to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
    print(REP.to_string(index=False))
    if not REP.match.all():
        say("REPRODUCTION FAILED - analysis stopped")
        sys.exit(1)

    # ---- 1. data -------------------------------------------------------------------------
    data = mc.load_all(say)
    oef = os.path.join(mc.EXT, "oe_pressorstays.parquet")
    if not os.path.exists(oef):
        raise FileNotFoundError("run mimic/07_vpstop_timing.py first (writes %s)" % oef)
    P = P.reset_index()                       # keep the D0 row index (key of the imputations) in column 'index'
    sid = set(P.stay_id)
    T0 = P.set_index("stay_id").t0
    ONH = P.set_index("stay_id").onset
    WT = P.set_index("stay_id").weight

    # ---- 2. new variables ----------------------------------------------------------------
    L = data["lb"].merge(P[["hadm_id", "stay_id", "t0"]], on="hadm_id")
    L["h"] = hrs(L.charttime, L.t0)
    post = L[L.itemid.isin(NA_ITEMS) & (L.h > 0) & (L.h <= 72)]
    mx = post.groupby("stay_id").v.transform("max")
    chkpk = P.stay_id.map(post.groupby("stay_id").v.max())
    assert np.allclose(chkpk.fillna(-1).values, P.na_peak_72.fillna(-1).values), "sodium peak differs from v15"
    P["t_na_peak"] = P.stay_id.map(post[post.v == mx].groupby("stay_id").h.min())
    P["na_set"] = ((P.na_max_pre24 < 150) & (P.na_n72 >= 1)).astype(int)
    P["na_rise"] = (P.na_peak_72 - P.na_pre).where(P.na_set == 1)
    P["na_base"] = P.na_pre.where(P.na_set == 1)
    P["na_peak72"] = P.na_peak_72.where(P.na_set == 1)
    P["na150"] = P.na150_72.astype(float).where(P.na_set == 1)
    P["t_na_peak"] = P.t_na_peak.where(P.na_set == 1)
    mp = data["mp"][data["mp"].stay_id.isin(sid)].copy()
    mp = mp[(mp.v > 10) & (mp.v < 200)]
    mp["h"] = hrs(mp.charttime, mp.stay_id.map(T0))
    m24 = mp[(mp.h >= 0) & (mp.h < 24)].assign(hb=lambda x: np.floor(x.h))
    P["map_min24_raw"] = P.stay_id.map(m24.groupby("stay_id").v.min())
    hmed = m24.groupby(["stay_id", "hb"]).v.median()                     # hourly median, then lowest hour
    P["map_min24"] = P.stay_id.map(hmed.groupby(level=0).min())
    P["map_lt65"] = (P.map_min24 < 65).astype(float).where(P.map_min24.notna())
    P["restart24"] = (P.restart_t <= 24).astype(int)
    P["restart48"] = (P.restart_t <= 48).astype(int)
    P["ddavp_any48"] = P.ddavp_any_48
    # free water 0-72 h
    ie = data["ie"][data["ie"].stay_id.isin(sid)]
    fwm = ((ie.itemid == FW_WATER) | ((ie.itemid == FW_D5W) & ie.ordercategorydescription.isin(FW_D5W_CATS)))
    fw = ie[fwm & (ie.amountuom == "mL") & (ie.amount > 0)][["stay_id", "starttime", "endtime", "amount", "itemid"]]
    a = fw.stay_id.map(T0).values
    amt, hit = mf.win_in(fw, a, a + np.timedelta64(72, "h"))
    g = pd.Series(amt[hit], index=fw.stay_id.values[hit]).groupby(level=0).sum()
    P["fw_any72"] = P.stay_id.isin(set(g[g > 0].index)).astype(int)
    P["fw_mlkg72"] = P.stay_id.map(g).fillna(0) / P.weight
    fwi = fw.assign(vol=amt, hit=hit)
    fwi = fwi[fwi.hit].groupby("itemid").stay_id.nunique().rename("stays")
    fwi[fwi >= 10].to_csv(os.path.join(out, "free_water_items_used.csv"))
    # loop diuretic within 6 h before onset
    dv = data["dv"][data["dv"].stay_id.isin(sid)].copy()
    dv["h"] = hrs(dv.starttime, dv.stay_id.map(T0))
    dv["on"] = dv.stay_id.map(ONH)
    P["loop6"] = P.stay_id.isin(set(dv.loc[(dv.h >= dv.on - 6) & (dv.h <= dv.on), "stay_id"])).astype(float)
    P.loc[P.onset.isna(), "loop6"] = np.nan
    # fluid balance around onset
    oe = pd.read_parquet(oef)
    oe = oe[oe.stay_id.isin(sid)].copy()
    oe["charttime"] = pd.to_datetime(oe.charttime)
    I, O, _ = mf.prep_io(ie, oe)
    onP = P[P.onset.notna()].set_index("stay_id")
    ont = onP.t0 + pd.to_timedelta(onP.onset, unit="h")
    Ii, Oo = I[I.stay_id.isin(ont.index)], O[O.stay_id.isin(ont.index)]
    for nm, (da, db) in {"pre24": (-24, 0), "post24": (0, 24)}.items():
        a = (Ii.stay_id.map(ont) + da * H).values
        b = (Ii.stay_id.map(ont) + db * H).values
        amt, hit = mf.win_in(Ii, a, b)
        inn = pd.DataFrame({"s": Ii.stay_id.values, "v": amt, "h": hit})
        inn = inn[inn.h].groupby("s").v.sum()
        oa, ob = (Oo.stay_id.map(ont) + da * H), (Oo.stay_id.map(ont) + db * H)
        oo = Oo[(Oo.charttime >= oa) & (Oo.charttime < ob)].groupby("stay_id").value.sum()
        bal = (inn - oo).dropna()
        P[f"bal_{nm}_mlkg"] = P.stay_id.map(bal) / P.weight
    # hourly urine bins (mimic_common.hourly_bins) and check of the onset
    hr = mc.hourly_bins(data["u"], P[["stay_id", "t0", "weight"]])
    oc = mc.outcomes_from_bins(hr)
    assert np.allclose(P.stay_id.map(oc.set_index("stay_id").onset).fillna(-1).values, P.onset.fillna(-1).values), "onset differs from v15"
    P["uo_max_rate24"] = P.stay_id.map(hr[(hr.h >= 0) & (hr.h < 24)].groupby("stay_id").rate.max())
    P["uo24_mlkg"] = P.uo24_ml / P.weight
    P["age"] = P.anchor_age
    P["icu_los_d"] = P.los
    P["hosp_death"] = P.hospital_expire_flag
    for k in ("0", "1", "2", "3", "unk"):
        P[f"kdigo_is_{k}"] = (P.kdigo_cat == k).astype(int)
    for u in sorted(P.first_careunit.dropna().unique()):
        P["unit_" + u] = (P.first_careunit == u).astype(int)
    P["timing"] = np.select([P.onset <= 8, (P.onset > 8) & (P.onset <= 24)], ["early", "late"], "none")
    P["early"] = (P.timing == "early").astype(int)
    P["grp4"] = np.where(P.timing == "none", "no polyuria", np.where(P.vp == 1, "VP ", "NE ") + P.timing)
    P = P.set_index("index")                  # restore the D0 index for mm.with_imp
    P.drop(columns=[c for c in P.columns if P[c].dtype == object and c not in ("grp", "grp4", "timing", "first_careunit", "kdigo_cat", "gender")]) \
        .to_parquet(os.path.join(wk, "mimic_early_late_stays.parquet"))
    say("variables built; groups:", P.grp4.value_counts().to_dict())
    IMPS = mm.impute(D0)
    say("imputed")

    # ---- 3. analysis A -------------------------------------------------------------------
    A = P[P.poly24 == 1].copy()
    units = [c for c in P.columns if c.startswith("unit_")]
    SPEC = [("na_base", "Na baseline (last value -24-0 h), mmol/L", "cont", 0),
            ("na_peak72", "Na peak 0-72 h, mmol/L", "cont", 0),
            ("na_rise", "Na rise (peak - baseline), mmol/L", "cont", 1),
            ("na150", "Na >= 150 within 72 h", "bin", 1),
            ("t_na_peak", "time from T0 to Na peak, h", "cont", 1),
            ("map_min24", "lowest hourly median MAP 0-24 h, mm Hg", "cont", 0),
            ("map_lt65", "lowest hourly median MAP < 65 mm Hg 0-24 h", "bin", 1),
            ("map_min24_raw", "lowest single MAP value 0-24 h, mm Hg", "cont", 0),
            ("restart24", "vasopressor restart within 24 h", "bin", 1),
            ("restart48", "vasopressor restart within 48 h", "bin", 1),
            ("ddavp_any48", "any desmopressin order 0-48 h (v15)", "bin", 1),
            ("ddavp_ad_48", "antidiuretic desmopressin 0-48 h (v15)", "bin", 1),
            ("fw_any72", "free water (D5W as IV fluid, enteral free water) 0-72 h", "bin", 1),
            ("fw_mlkg72", "free water volume 0-72 h, mL/kg", "cont", 1),
            ("loop6", "IV loop diuretic within 6 h before onset", "bin", 1),
            ("bal_pre24_mlkg", "fluid balance 24 h before onset, mL/kg", "cont", 1),
            ("bal_post24_mlkg", "fluid balance 24 h after onset, mL/kg", "cont", 1),
            ("uo_max_rate24", "highest hourly urine output 0-24 h, mL/kg/h", "cont", 2),
            ("uo24_mlkg", "urine output 0-24 h, mL/kg", "cont", 1),
            ("age", "age, y", "cont", 0), ("male", "male", "bin", 1), ("cvicu", "cardiac vascular ICU", "bin", 1)] + \
           [(u, "ICU type: " + u[5:], "bin", 1) for u in units] + \
           [(f"kdigo_is_{k}", f"KDIGO pre-T0 = {k}", "bin", 1) for k in ("0", "1", "2", "3", "unk")] + \
           [("hosp_death", "in-hospital death", "bin", 1), ("icu_los_d", "ICU length of stay, d", "cont", 1)]
    nrow = [dict(group=g, n=int((A.grp4 == g).sum())) for g in GROUPS]
    pd.DataFrame(nrow).assign(n=lambda x: x.n.where(x.n >= 10, "<10")).to_csv(os.path.join(out, "A_group_sizes.csv"), index=False)
    describe(A, "grp4", SPEC, GROUPS).to_csv(os.path.join(out, "A_summary.csv"), index=False)
    pairs = [("VP early", "VP late"), ("VP early", "NE early")]
    say("bootstrap A", NBOOT)
    boot_contrasts(A, "grp4", SPEC[:19], pairs).to_csv(os.path.join(out, "A_contrasts_bootstrap.csv"), index=False)
    Ax = A[A.loop6 != 1]
    describe(Ax, "grp4", SPEC[:5], GROUPS).to_csv(os.path.join(out, "A_summary_noloop6.csv"), index=False)
    boot_contrasts(Ax, "grp4", SPEC[:5], pairs).to_csv(os.path.join(out, "A_contrasts_bootstrap_noloop6.csv"), index=False)
    box_stats(A, "grp4", ["na_rise", "map_min24"], GROUPS).to_csv(os.path.join(out, "A_figure_data.csv"), index=False)
    rows = ols_interaction(A[A.na_set == 1], IMPS, "na_rise", "main: polyuria within 24 h, Na set")
    rows += ols_interaction(Ax[Ax.na_set == 1], IMPS, "na_rise", "sensitivity: no loop diuretic within 6 h before onset")
    pd.DataFrame(rows).to_csv(os.path.join(out, "A_regression_na_rise.csv"), index=False)
    say("A done")

    # ---- 4. analysis B -------------------------------------------------------------------
    B = P[P.na_set == 1].copy()
    rise = B.na_peak_72 - B.na_pre
    defs = {"main: rise >= 5 or peak >= 150": (rise >= 5) | (B.na_peak_72 >= 150),
            "sens: rise >= 3 or peak >= 150": (rise >= 3) | (B.na_peak_72 >= 150),
            "sens: peak >= 150 only": B.na_peak_72 >= 150}
    crow, mrow = [], []
    for k, m in defs.items():
        B["cat"] = np.where(B.poly24 == 0, 0, np.where(m, 2, 1))
        for g, gl in (("VP", "VP"), ("NE", "NE")):
            s = B[B.grp == g]
            for c, cl in ((0, "no polyuria"), (1, "polyuria without Na rise"), (2, "polyuria with Na rise")):
                txt, p, lo, hi = fmt_prop(int((s.cat == c).sum()), len(s))
                crow.append(dict(definition=k, group=gl, category=c, label=cl, text=txt, pct=p, lo=lo, hi=hi,
                                 n_group=len(s) if len(s) >= 10 else "<10"))
        mrow += mnlogit_vp(B, IMPS, "cat", k)
        say("B", k)
    pd.DataFrame(crow).to_csv(os.path.join(out, "B_category_counts.csv"), index=False)
    # sodium rise by polyuria status: is the rise specific to stays with polyuria?
    nrow = []
    for k, m in defs.items():
        for g, gl in (("VP", "VP"), ("NE", "NE")):
            for pv, pl in ((0, "no polyuria within 24 h"), (1, "polyuria within 24 h")):
                s = (B.grp == g) & (B.poly24 == pv)
                txt, p, lo, hi = fmt_prop(int((m & s).sum()), int(s.sum()))
                nrow.append(dict(definition=k, group=gl, polyuria=pl, na_rise=txt, pct=p, lo=lo, hi=hi))
    pd.DataFrame(nrow).to_csv(os.path.join(out, "B_na_rise_by_polyuria.csv"), index=False)
    pd.DataFrame(mrow).to_csv(os.path.join(out, "B_multinomial.csv"), index=False)
    say("finished")


if __name__ == "__main__":
    main()
