#!/usr/bin/env python3
"""
08_early_late.py - eICU-CRD v2.0, post hoc exploratory analyses added after v15 (reported in
the manuscript as post hoc exploratory analyses).

  A  early (onset 0-8 h) vs late (>8-24 h) polyuria, VP vs NE-only, among the stays of the
     primary population with polyuria within 24 h: sodium, haemodynamics, co-interventions,
     urine and background; bootstrap differences for VP-early vs VP-late and VP-early vs
     NE-early; linear model of the sodium rise with group x timing interaction
  B  three-level outcome within 24 h: no polyuria / polyuria without sodium rise / polyuria with
     sodium rise; multinomial logistic regression (reference = no polyuria)

Definitions taken unchanged from v15 (module eicu_cohort.py, script 01): cohort, groups, T0,
polyuria (uo_onset_h), covariates (eicu_stats.COV), imputation (eicu_stats.impute) and the
sodium handling of 01 (lab 'sodium', 100 < v < 200; baseline = last value in (-24, 0] h; peak =
max in (0, 72] h; denominator = all pre-T0 24-h values < 150 and >= 1 value in 0-72 h).

New variables defined here (not used in v15)
  map_min24        lowest hourly median MAP in hours 0-23 after T0 (vitalAperiodic + vitalPeriodic,
                   10-200 mm Hg as in eicu_cohort.covariates); map_min24_raw = lowest single value
  restart24/48     first positive vasopressor record after T0 within 24 / 48 h (restart_h of v15)
  ddavp_med_48     desmopressin in the medication table, start in (0, 48] h
  fw_any72         free-water intake in (0, 72] h: intakeOutput intake labels for D5W (plain) or
                   enteral / oral water and tube flushes (FW_RE, D5W_RE below); volume
                   apportioned as in eicu_fluid.build_records
  loop6            loop-diuretic event of v15 (eicu_cohort.covariates: medication + infusionDrug,
                   DIUR_RE) with h in [onset - 6, onset]
  bal_pre24/post24 intake - (urine + other output), apportioned as eicu_fluid, windows
                   (onset - 24 h, onset] and (onset, onset + 24 h]; missing without any intake or
                   any output record in the window; mL/kg
  uo_max_rate24    highest hourly urine bin (mL/kg/h) in bins 0-23 (bins of eicu_cohort.urine_outcomes)

Outputs  OUT_DIR/eicu_08/*.csv (aggregates only; cells < 10 shown as "<10")
         WORK_DIR/eicu/early_late/eicu_early_late_stays.parquet (row level)
Run:     python eicu/08_early_late.py   (after eicu/01_primary_analysis.py)
Optional environment variables: NBOOT_EL (bootstrap replicates, default 2000), MICE_M (10).
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
from eicu_common import COHORT_PRIMARY, cohort_path, env_int, ex_dir, out_dir  # noqa: E402
import config  # noqa: E402  (importable once eicu_common has put the repository root on sys.path)
from eicu_extract import load_source  # noqa: E402
from eicu_cohort import build_primary_cohort, columns_differing_from_stored, urine_and_intake_records, urine_outcomes  # noqa: E402
from eicu_fluid import add_category_and_value, build_records  # noqa: E402
from eicu_stats import COV, MICE_M, impute  # noqa: E402

warnings.filterwarnings("ignore")
NBOOT = env_int("NBOOT_EL", 2000)
SEED = 20261117
D5W_RE = r"^d5w$|^d5w \+ add$|^volume \(ml\)-dextrose 5 % solution$"
FW_RE = (r"free water|free h2o|water bolus|water flush|h2o|h20|po water|enteral water|enteral.*flush|tube.*flush|"
         r"(?:ogt|ngt|og|ng|peg|gi tube) flush")
FW_EXCL = r"bicarbonate|sodium|saline"
GROUPS = ["VP early", "VP late", "NE early", "NE late"]
T0_START = time.time()


def say(*a):
    print("%5.0fs" % (time.time() - T0_START), *a, flush=True)


# ----------------------------------------------------------------------------------------
# shared helpers (identical in mimic/10_early_late.py)
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
# eICU-specific estimation (conventions of eicu_stats.fit_or: chained-equation imputation of the
# covariates with outcome and exposure terms in the imputation model, hospital-clustered SE,
# Rubin's rules with a normal reference distribution)
def pool(P, U):
    P = np.asarray(P)
    m = len(P)
    q = P.mean(axis=0)
    W = sum(U) / m
    B = np.cov(P.T, ddof=1) if m > 1 else 0 * W
    return q, W + (1 + 1 / m) * B


def mi_sets(d, extra):
    cc = list(COV)
    return impute(d, cc, extra, MICE_M), cc


def ols_interaction(d, y, label):
    d = d.dropna(subset=[y, "hospitalid"]).copy()
    d["vp_early"] = d.vp * d.early
    sets, cc = mi_sets(d, [y, "vp", "early", "vp_early"])
    xs = ["vp", "early", "vp_early"] + cc
    P, U = [], []
    for s in sets:
        keep = [c for c in xs if s[c].nunique() > 1]
        X = sm.add_constant(s[keep].astype(float))
        r = sm.OLS(s[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": s.hospitalid})
        P.append(r.params.reindex(["const"] + xs).fillna(0).values)
        U.append(r.cov_params().reindex(index=["const"] + xs, columns=["const"] + xs).fillna(0).values)
    q, T = pool(P, U)
    names = ["const"] + xs
    ix = {n: i for i, n in enumerate(names)}
    combos = {"interaction (VP x early)": {"vp_early": 1},
              "VP early - VP late (early + interaction)": {"early": 1, "vp_early": 1},
              "VP early - NE early (VP + interaction)": {"vp": 1, "vp_early": 1},
              "NE early - NE late (early)": {"early": 1},
              "VP late - NE late (VP)": {"vp": 1}}
    rows = []
    for k, w in combos.items():
        L = np.zeros(len(names))
        for n, c in w.items():
            L[ix[n]] = c
        est, se = L @ q, np.sqrt(L @ T @ L)
        rows.append(dict(analysis=label, term=k, est=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se,
                         p=2 * norm.sf(abs(est / se)), n=len(d), n_hospitals=d.hospitalid.nunique(),
                         method="OLS, MICE m=%d, hospital-clustered SE" % len(sets)))
    return rows


def mnlogit_vp(d, ycat, label):
    d = d.dropna(subset=[ycat, "hospitalid"]).copy()
    d["c1"] = (d[ycat] == 1).astype(int)
    d["c2"] = (d[ycat] == 2).astype(int)
    sets, cc = mi_sets(d, ["c1", "c2", "vp"])
    xs = ["vp"] + cc
    P, U = [], []
    for s in sets:
        keep = [c for c in xs if s[c].nunique() > 1]
        X = sm.add_constant(s[keep].astype(float))
        r = sm.MNLogit(s[ycat].astype(int), X).fit(disp=0, maxiter=300, method="newton",
                                                   cov_type="cluster", cov_kwds={"groups": s.hospitalid})
        b = r.params.loc["vp"].values                      # equations: category 1, category 2
        V = r.cov_params()
        k1 = [i for i in V.index if i[1] == "vp"]           # MultiIndex (equation, term)
        P.append(b)
        U.append(V.loc[k1, k1].values)
    q, T = pool(P, U)
    rows = []
    for j, cat in enumerate((1, 2)):
        se = np.sqrt(T[j, j])
        rows.append(dict(analysis=label, category=cat, logOR=q[j], se=se, OR=np.exp(q[j]),
                         lo=np.exp(q[j] - 1.96 * se), hi=np.exp(q[j] + 1.96 * se), p=2 * norm.sf(abs(q[j] / se)),
                         n=len(d), method="multinomial logit, MICE m=%d, hospital-clustered SE" % len(sets)))
    return rows


# ----------------------------------------------------------------------------------------
def main():
    out = out_dir("08")
    wk = config.work_dir("eicu", "early_late")
    stored = pd.read_parquet(cohort_path(COHORT_PRIMARY))

    # ---- 0. reproduction of v15 values from the stored cohort table -------------------------
    PR = stored[stored.primary == 1].copy()
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
    REP = pd.DataFrame(rep)
    REP.to_csv(os.path.join(out, "reproduction_checks.csv"), index=False)
    print(REP.to_string(index=False))
    if not REP.match.all():
        say("REPRODUCTION FAILED - analysis stopped")
        sys.exit(1)

    # ---- 1. rebuild the cohort with the v15 module (gives the event tables) ----------------
    src = load_source(say)
    C, aux = build_primary_cohort(src, say)
    bad = columns_differing_from_stored(C, stored)
    say("columns differing from the stored cohort table:", bad)
    if bad:
        say("REBUILT COHORT DIFFERS FROM STORED TABLE - analysis stopped")
        sys.exit(1)
    P = C[C.primary == 1].copy().reset_index(drop=True)
    sid = set(P.patientunitstayid)
    T0 = P.set_index("patientunitstayid").t0
    ON = P.set_index("patientunitstayid").uo_onset_h

    # ---- 2. new variables ----------------------------------------------------------------
    # sodium: same filter as eicu_cohort.covariates
    lab = pd.read_csv(os.path.join(ex_dir(), "x_lab.csv"), low_memory=False,
                      usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    lab = lab[lab.patientunitstayid.isin(sid)]
    lab["labname"] = lab.labname.astype(str).str.lower().str.strip()
    lab["v"] = pd.to_numeric(lab.labresult, errors="coerce")
    na = lab[(lab.labname == "sodium") & (lab.v > 100) & (lab.v < 200)].copy()
    na["h"] = (na.labresultoffset - na.patientunitstayid.map(T0)) / 60
    post = na[(na.h > 0) & (na.h <= 72)].sort_values(["patientunitstayid", "h"])
    mx = post.groupby("patientunitstayid").v.transform("max")
    P["na_peak72_chk"] = P.patientunitstayid.map(post.groupby("patientunitstayid").v.max())
    P["t_na_peak"] = P.patientunitstayid.map(post[post.v == mx].groupby("patientunitstayid").h.min())
    assert np.allclose(P.na_peak72_chk.fillna(-1), P.na_post72_max.fillna(-1)), "sodium peak differs from v15"
    P["na_set"] = ((P.na_pre_max < 150) & (P.na_post72_n >= 1)).astype(int)
    P["na_rise"] = (P.na_post72_max - P.na_pre).where(P.na_set == 1)
    P["na_base"] = P.na_pre.where(P.na_set == 1)
    P["na_peak72"] = P.na_post72_max.where(P.na_set == 1)
    P["na150"] = (P.na_post72_max >= 150).astype(float).where(P.na_set == 1)
    P["t_na_peak"] = P.t_na_peak.where(P.na_set == 1)
    # MAP
    mp = pd.concat([pd.read_csv(os.path.join(ex_dir(), "x_vap.csv")), pd.read_csv(os.path.join(ex_dir(), "x_vp.csv"))], ignore_index=True)
    mp = mp[mp.patientunitstayid.isin(sid)]
    mp["map"] = pd.to_numeric(mp["map"], errors="coerce")
    mp = mp[(mp["map"] > 10) & (mp["map"] < 200)]
    mp["h"] = (mp.observationoffset - mp.patientunitstayid.map(T0)) / 60
    m24 = mp[(mp.h >= 0) & (mp.h < 24)].assign(hb=lambda x: np.floor(x.h))
    P["map_min24_raw"] = P.patientunitstayid.map(m24.groupby("patientunitstayid")["map"].min())
    hmed = m24.groupby(["patientunitstayid", "hb"])["map"].median()       # hourly median, then lowest hour
    P["map_min24"] = P.patientunitstayid.map(hmed.groupby(level=0).min())
    P["map_lt65"] = (P.map_min24 < 65).astype(float).where(P.map_min24.notna())
    del mp
    # restart, desmopressin
    P["restart24"] = (P.restart_h <= 24).astype(int)
    P["restart48"] = (P.restart_h <= 48).astype(int)
    med = src.med[src.med.patientunitstayid.isin(sid) & src.med.drugname.str.contains(r"desmopressin|ddavp", case=False)
                  & (src.med.drugordercancelled.astype(str) != "Yes")].copy()
    med["h"] = (med.drugstartoffset - med.patientunitstayid.map(T0)) / 60
    P["ddavp_med_48"] = P.patientunitstayid.isin(set(med.loc[(med.h > 0) & (med.h <= 48), "patientunitstayid"])).astype(int)
    P["ddavp_ad_48"] = P.ddavp_antidiuretic_48h
    P["ddavp_any48"] = ((P.ddavp_med_48 == 1) | (P.ddavp_ad_48 == 1)).astype(int)
    # loop diuretic within 6 h before onset (v15 event table aux.ev, h from T0)
    ev = aux.ev[aux.ev.patientunitstayid.isin(sid)].copy()
    ev["on"] = ev.patientunitstayid.map(ON)
    P["loop6"] = P.patientunitstayid.isin(set(ev.loc[(ev.h >= ev.on - 6) & (ev.h <= ev.on), "patientunitstayid"])).astype(int)
    P.loc[P.uo_onset_h.isna(), "loop6"] = np.nan
    # intake / output records (eicu_fluid conventions), free water and balance around onset
    io = pd.read_csv(os.path.join(ex_dir(), "x_intakeoutput.csv"), low_memory=False)
    io = io[io.patientunitstayid.isin(sid)]
    add_category_and_value(io)
    REC, _ = build_records(io)
    del io
    lab_ = REC.celllabel.astype(str).str.lower().str.strip()
    REC["fw"] = (REC.kind == "in") & (lab_.str.contains(D5W_RE) | (lab_.str.contains(FW_RE) & ~lab_.str.contains(FW_EXCL)))
    fwl = REC[REC.fw].assign(l=lab_[REC.fw]).groupby("l").patientunitstayid.nunique().sort_values(ascending=False)
    fwl[fwl >= 10].rename("stays").to_csv(os.path.join(out, "free_water_labels_used.csv"))

    def window(R, a, b):
        ov = np.clip(np.minimum(R.e.values, b) - np.maximum(R.s.values, a), 0, None)
        return R.v.values * (ov / R.L.values), ov > 0
    R = REC.copy()
    R["t0"] = R.patientunitstayid.map(T0).values.astype(float)
    R["on"] = R.patientunitstayid.map(ON).values.astype(float)
    W = P.set_index("patientunitstayid").weight_kg
    vol, hit = window(R[R.fw], R.t0[R.fw].values, R.t0[R.fw].values + 4320)
    f = R[R.fw].assign(vol=vol, hit=hit)
    g = f[f.hit].groupby("patientunitstayid").vol.sum()
    P["fw_any72"] = P.patientunitstayid.isin(set(g[g > 0].index)).astype(int)
    P["fw_mlkg72"] = (P.patientunitstayid.map(g).fillna(0) / P.weight_kg)
    for nm, (da, db) in {"pre24": (-1440, 0), "post24": (0, 1440)}.items():
        on_min = R.t0.values + 60 * R.on.values
        v, h_ = window(R, on_min + da, on_min + db)
        x = R.assign(vol=v, hit=h_)
        x = x[x.hit & np.isfinite(on_min)]
        inn = x[x.kind == "in"].groupby("patientunitstayid").vol.sum()
        outv = x[x.kind.isin(["urine", "oth"])].groupby("patientunitstayid").vol.sum()
        bal = (inn - outv).dropna()                       # both sides recorded
        P[f"bal_{nm}_mlkg"] = P.patientunitstayid.map(bal) / P.weight_kg
    del R, REC
    # hourly urine bins (eicu_cohort.urine_outcomes on a copy)
    ur, _ = urine_and_intake_records()
    Cc = P[["patientunitstayid", "t0", "admissionweight", "riskset"]].copy()
    hr = urine_outcomes(Cc, src.inf, ur)
    assert np.allclose(Cc.uo_onset_h.fillna(-1).values, P.uo_onset_h.fillna(-1).values), "urine onset differs from v15"
    P["uo_max_rate24"] = P.patientunitstayid.map(hr[(hr.h >= 0) & (hr.h < 24)].groupby("patientunitstayid").rate.max())
    P["uo24_mlkg"] = P.uo_24h_ml / P.weight_kg
    # background
    P["age"] = P.age_n
    P["icu_los_d"] = P.unitdischargeoffset / 1440
    P["hosp_death"] = P.hosp_death.astype(int)
    for k in ("0", "1", "2", "3", "unk"):
        P[f"kdigo_is_{k}"] = (P.kdigo_cat.astype(str).str.replace(".0", "", regex=False) == k).astype(int)
    for u in sorted(P.unittype.dropna().unique()):
        P["unit_" + u] = (P.unittype == u).astype(int)
    # groups
    P["timing"] = np.select([P.uo_onset_h <= 8, (P.uo_onset_h > 8) & (P.uo_onset_h <= 24)], ["early", "late"], "none")
    P["early"] = (P.timing == "early").astype(int)
    P["grp4"] = np.where(P.timing == "none", "no polyuria", np.where(P.vp == 1, "VP ", "NE ") + P.timing)
    P.to_parquet(os.path.join(wk, "eicu_early_late_stays.parquet"), index=False)
    say("variables built; groups:", P.grp4.value_counts().to_dict())

    # ---- 3. analysis A -------------------------------------------------------------------
    A = P[P.polyuria_24h == 1].copy()
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
            ("ddavp_any48", "desmopressin 0-48 h (treatment table or medication)", "bin", 1),
            ("ddavp_ad_48", "antidiuretic desmopressin 0-48 h (v15, treatment table)", "bin", 1),
            ("fw_any72", "free water (D5W, enteral/oral water, tube flushes) 0-72 h", "bin", 1),
            ("fw_mlkg72", "free water volume 0-72 h, mL/kg", "cont", 1),
            ("loop6", "loop diuretic within 6 h before onset", "bin", 1),
            ("bal_pre24_mlkg", "fluid balance 24 h before onset, mL/kg", "cont", 1),
            ("bal_post24_mlkg", "fluid balance 24 h after onset, mL/kg", "cont", 1),
            ("uo_max_rate24", "highest hourly urine output 0-24 h, mL/kg/h", "cont", 2),
            ("uo24_mlkg", "urine output 0-24 h, mL/kg", "cont", 1),
            ("age", "age, y", "cont", 0), ("male", "male", "bin", 1), ("cvicu", "cardiac ICU", "bin", 1)] + \
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
    rows = ols_interaction(A[A.na_set == 1], "na_rise", "main: polyuria within 24 h, Na set")
    rows += ols_interaction(Ax[Ax.na_set == 1], "na_rise", "sensitivity: no loop diuretic within 6 h before onset")
    pd.DataFrame(rows).to_csv(os.path.join(out, "A_regression_na_rise.csv"), index=False)
    say("A done")

    # ---- 4. analysis B -------------------------------------------------------------------
    B = P[P.na_set == 1].copy()
    rise = B.na_post72_max - B.na_pre
    defs = {"main: rise >= 5 or peak >= 150": (rise >= 5) | (B.na_post72_max >= 150),
            "sens: rise >= 3 or peak >= 150": (rise >= 3) | (B.na_post72_max >= 150),
            "sens: peak >= 150 only": B.na_post72_max >= 150}
    crow, mrow = [], []
    for k, m in defs.items():
        B["cat"] = np.where(B.polyuria_24h == 0, 0, np.where(m, 2, 1))
        for g, gl in (("VP", "VP"), ("NE", "NE-only")):
            s = B[B.grp == g]
            for c, cl in ((0, "no polyuria"), (1, "polyuria without Na rise"), (2, "polyuria with Na rise")):
                txt, p, lo, hi = fmt_prop(int((s.cat == c).sum()), len(s))
                crow.append(dict(definition=k, group=gl, category=c, label=cl, text=txt, pct=p, lo=lo, hi=hi,
                                 n_group=len(s) if len(s) >= 10 else "<10"))
        mrow += mnlogit_vp(B, "cat", k)
        say("B", k)
    pd.DataFrame(crow).to_csv(os.path.join(out, "B_category_counts.csv"), index=False)
    # sodium rise by polyuria status: is the rise specific to stays with polyuria?
    nrow = []
    for k, m in defs.items():
        for g, gl in (("VP", "VP"), ("NE", "NE-only")):
            for pv, pl in ((0, "no polyuria within 24 h"), (1, "polyuria within 24 h")):
                s = (B.grp == g) & (B.polyuria_24h == pv)
                txt, p, lo, hi = fmt_prop(int((m & s).sum()), int(s.sum()))
                nrow.append(dict(definition=k, group=gl, polyuria=pl, na_rise=txt, pct=p, lo=lo, hi=hi))
    pd.DataFrame(nrow).to_csv(os.path.join(out, "B_na_rise_by_polyuria.csv"), index=False)
    pd.DataFrame(mrow).to_csv(os.path.join(out, "B_multinomial.csv"), index=False)
    say("finished")


if __name__ == "__main__":
    main()
