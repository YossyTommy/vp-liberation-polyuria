#!/usr/bin/env python3
"""
02_primary_analysis.py - MIMIC-IV v3.1, primary two-group analysis.

Compares new-onset polyuria after liberation from vasopressors between ICU stays in which
vasopressin was the index drug (VP) and stays weaned from norepinephrine without any
vasopressin (NE). Post hoc analysis; unit = ICU stay (patients reported alongside); standard
errors clustered on subject_id. Groups, T0, the polyuria rule and the primary population are
defined in mimic_common.py.

Steps
  1. check that the csv extracts written by 01_extract.sh exist (WORK_DIR/mimic/data)
  2. cached extraction from the raw files (WORK_DIR/mimic/ext; mimic_common.build_caches)
  3. cohort construction (mimic_common.build) for episode-joining gaps of 1 h (main), 0.5 h and 2 h
  4. analyses: logistic regression after multiple imputation (MICE, Rubin's rules), ATT
     inverse probability weighting with a patient-level bootstrap, Aalen-Johansen cumulative
     incidence, cause-specific Cox models, sensitivity and subgroup analyses
  5. output: one results workbook, figure data (csv) and the analysis-ready cohort table

Paths come from config.py (MIMIC_DIR, WORK_DIR, OUT_DIR). Optional environment variables:
NBOOT (bootstrap replicates, default 500), M_IMP (imputations, 10), NJOBS (threads, 6).

Outputs
  OUT_DIR/mimic_02/   results_mimic_primary.xlsx, fig_mimic_*.csv,
                      bootstrap_ipw_replicates.csv, key_results.json, run_log.txt
  WORK_DIR/mimic/cohort/   analysis_ready_mimic_primary.parquet (and _gap0.5 / _gap2.0)

Run:  python mimic/02_primary_analysis.py   (after 01_extract.sh)
"""
import os
import sys
import time
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from joblib import Parallel, delayed
from lifelines import CoxPHFitter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # folder of the shared modules
from mimic_common import (config, RunLog, PRIMARY_TABLE, cohort_file, check_extracts,  # noqa: E402
                          build_caches, load_all, build, cohort, populations)
from mimic_models import (M_IMP, COV_MI, COV_CC, YS, cov_without, impute, with_imp, glm_fit,  # noqa: E402
                          rubin, or_mi, fmt, ps_weights, event_data, aj)

warnings.filterwarnings("ignore")

NBOOT = int(os.environ.get("NBOOT", 500))
NJOBS = int(os.environ.get("NJOBS", 6))
# covariates of the initial model, used in the step-by-step table (sheet 02_reproduction)
COV_INITIAL = "age10+male+cvicu+C(kdigo0)+na_pre+map_base+diur_pre24+fluid_pre24_l+log_press_h+ne_peak_c+other_vaso_pre"
say = RunLog()


# ===========================================================================
# Estimation tools used only by this script
def or_cc(d, y, expo="vp", cov=COV_CC):
    x = d.dropna(subset=[c for c in ["na_pre", "map_base", "lactate_pre24", "log_press_h"] if c in (cov or "")])
    r = glm_fit(f"{y}~{expo}+{cov}" if cov else f"{y}~{expo}", x)
    lo, hi = r.conf_int().loc[expo]
    return dict(OR=np.exp(r.params[expo]), lo=np.exp(lo), hi=np.exp(hi), p=r.pvalues[expo], n=len(x))


def nN(d, y, g):
    s = d[d.grp == g]
    return "%d/%d (%.1f%%)" % (s[y].sum(), len(s), 100 * s[y].mean() if len(s) else np.nan)


def counts(d):
    return dict(n_stays=len(d), n_patients=d.subject_id.nunique(),
                n_VP=int((d.vp == 1).sum()), n_NE=int((d.vp == 0).sum()))


def cs_cox(x, w, T, Cc, lo, hi):
    df = pd.DataFrame({"T": T, "C": Cc, "vp": x.vp.values, "w": w, "sid": x.subject_id.values})
    df = df[df["T"] > lo].copy()
    df["entry"] = float(lo)
    df["dur"] = np.minimum(df["T"], hi)
    df["ev"] = ((df.C == 1) & (df["T"] <= hi)).astype(int)
    df.loc[df.dur <= df.entry, "dur"] = df.entry + 1e-6
    kw = dict(duration_col="dur", event_col="ev", weights_col="w", robust=True, cluster_col="sid")
    cols = ["dur", "ev", "vp", "w", "sid"]
    if lo > 0:
        kw["entry_col"] = "entry"
        cols.append("entry")
    cph = CoxPHFitter().fit(df[cols], **kw)
    s = cph.summary.loc["vp"]
    return dict(HR=s["exp(coef)"], lo=s["exp(coef) lower 95%"], hi=s["exp(coef) upper 95%"], p=s["p"],
                n=len(df), events=int(df.ev.sum()), events_VP=int(df.ev[df.vp == 1].sum()),
                events_NE=int(df.ev[df.vp == 0].sum()))


def evalue(rr, lo=None):
    def e(r):
        if r < 1:
            r = 1 / r
        return r + np.sqrt(r * (r - 1))
    out = {"E_point": e(rr)}
    if lo is not None:
        out["E_CI"] = 1.0 if lo <= 1 else e(lo)
    return out


# --- ATT IPW and bootstrap -----------------------------------------------------------
def ipw_stats(x, w, ys=YS):
    vp = x.vp.values == 1
    out = {}
    for y in ys:
        yy = x[y].values
        r1, r0 = yy[vp].mean(), np.average(yy[~vp], weights=w[~vp])
        out.update({f"{y}_r1": r1, f"{y}_r0": r0, f"{y}_RD": r1 - r0, f"{y}_RR": r1 / r0,
                    f"{y}_OR": (r1 / (1 - r1)) / (r0 / (1 - r0))})
    T, Cc = event_data(x)
    for sel, lab in ((vp, "VP"), (~vp, "NE")):
        o = aj(T[sel], Cc[sel], w[sel], grid=np.array([8, 24, 72]), causes=(1,))
        for k, g in enumerate((8, 24, 72)):
            out[f"cif{g}_{lab}"] = o[1][k]
    for g in (8, 24, 72):
        out[f"cif{g}_RD"] = out[f"cif{g}_VP"] - out[f"cif{g}_NE"]
    return out


def ipw_point(d, imps):
    rs, W = [], []
    for imp in imps:
        x = with_imp(d, imp)
        w, _ = ps_weights(x)
        rs.append(ipw_stats(x, w))
        W.append(w)
    return pd.DataFrame(rs).mean(), np.mean(W, axis=0)


def _boot_one(b, d, imps, grp_idx, us, seed):
    rng = np.random.default_rng(seed + b)
    pick = rng.choice(us, len(us), replace=True)
    idx = np.concatenate([grp_idx[s] for s in pick])
    x = with_imp(d, imps[b % len(imps)]).iloc[idx]
    try:
        w, _ = ps_weights(x)
        return ipw_stats(x, w)
    except Exception:
        return None


def bootstrap(d, imps, B=NBOOT, seed=12345):
    """Patient-level bootstrap; replicate b uses imputed data set b mod m."""
    subj = d.subject_id.values
    grp_idx = pd.Series(np.arange(len(d))).groupby(subj).apply(np.array).to_dict()
    us = np.unique(subj)
    res = Parallel(n_jobs=NJOBS, backend="threading")(
        delayed(_boot_one)(b, d, imps, grp_idx, us, seed) for b in range(B))
    return pd.DataFrame([r for r in res if r is not None])


def smd_table(d, imps, wlist):
    """Standardised mean differences (unweighted and ATT-weighted), computed per imputation and averaged."""
    rows = []
    for imp, w in zip(imps, wlist):
        x = with_imp(d, imp)
        X = pd.DataFrame({"age (10 y)": x.age10, "male": x.male, "CVICU": x.cvicu,
                          "Na pre (mmol/L)": x.na_i, "MAP pre 6 h": x.map_i, "lactate max pre 24 h": x.lact_i,
                          "loop diuretic pre 24 h": x.diur_pre24, "fluid pre 24 h (L)": x.fluid_pre24_l,
                          "log(1+vasopressor h)": x.log_press_h, "NE peak (0.1 mcg/kg/min)": x.ne_peak_c,
                          "other vasopressor pre-T0": x.other_vaso_pre, "invasive ventilation at T0": x.vent_t0,
                          "steroid pre 24 h": x.steroid_pre24, "inotrope at T0": x.inotrope_t0, "sepsis": x.sepsis,
                          "cardiac surgery": x.card_surg, "emergency admission": x.emergency})
        for k in "0123":
            X[f"KDIGO pre {k}"] = (x.kdigo_i == k).astype(int)
        vp = x.vp.values == 1
        r = {}
        for c in X.columns:
            a = X[c].values.astype(float)
            m1, v1 = a[vp].mean(), a[vp].var()
            m0u, v0u = a[~vp].mean(), a[~vp].var()
            m0w = np.average(a[~vp], weights=w[~vp])
            sd = np.sqrt((v1 + v0u) / 2)
            r[c] = (m1, m0u, m0w, (m1 - m0u) / sd if sd > 0 else 0, (m1 - m0w) / sd if sd > 0 else 0)
        rows.append(pd.DataFrame(r, index=["mean_VP", "mean_NE", "mean_NE_weighted", "SMD_unweighted",
                                           "SMD_ATT_weighted"]).T)
    t = sum(rows) / len(rows)
    t["|SMD|<0.1 after weighting"] = t.SMD_ATT_weighted.abs() < 0.1
    return t.round(3)


# --- restricted cubic spline ---------------------------------------------------------
def rcs_basis(x, knots):
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    K = len(k)
    nrm = (k[-1] - k[0]) ** 2
    cols = [x]
    for j in range(K - 2):
        tp = lambda a: np.maximum(a, 0) ** 3
        cols.append((tp(x - k[j]) - tp(x - k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
                     + tp(x - k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2])) / nrm)
    return np.column_stack(cols)


# ===========================================================================
# Analyses
SHEETS = {}
FOREST = []


def add_forest(section, analysis, pop, outcome, est_type, o, d=None, y=None, key="OR"):
    row = dict(section=section, analysis=analysis, population=pop, outcome=outcome, estimate_type=est_type,
               est=o[key], lo=o["lo"], hi=o["hi"], p=o.get("p", np.nan))
    if d is not None and y is not None:
        for g in ("VP", "NE"):
            s = d[d.grp == g]
            row[f"events_{g}"] = int(s[y].sum())
            row[f"N_{g}"] = len(s)
    FOREST.append(row)


def sheet_flow(D):
    rows = []
    for g in ("VP", "NE"):
        s = D[D.grp == g]
        m = pd.Series(True, index=s.index)
        steps = [("ICU stays with index drug (VP: any VP; NE: NE and never VP)", None),
                 ("age >= 18", s.ex_age == 0), ("T0 within the ICU stay", s.ex_notinicu == 0),
                 ("alive at T0", s.ex_deadpre == 0), ("no RRT during the stay", s.rrt_any == 0),
                 ("no pituitary/hypothalamic/DI dx, brain death or pituitary surgery", s.ex_neuroendo == 0),
                 ("evaluable: urine recorded in >=12 of 24 hourly bins", s.obs24 >= 12),
                 ("new-onset: no polyuria in the 12 h before T0  [= secondary population]", s.pre_poly == 0),
                 ("no vasopressor running at T0  [= PRIMARY population]", s.vaso_at_t0 == 0)]
        prev = len(s)
        for lab, c in steps:
            if c is not None:
                m &= c
            rows.append(dict(group=g, step=lab, n_stays=int(m.sum()), n_patients=s[m].subject_id.nunique(),
                             excluded=prev - int(m.sum())))
            prev = int(m.sum())
    f = pd.DataFrame(rows)
    # components of the neuroendocrine exclusion (among stays reaching that step)
    base = (D.ex_age == 0) & (D.ex_notinicu == 0) & (D.ex_deadpre == 0) & (D.rrt_any == 0)
    br = []
    for g in ("VP", "NE"):
        s = D[base & (D.grp == g)]
        for c in ("pit_di_codes", "pit_tumor", "brain_death", "pit_surg", "ex_neuroendo"):
            br.append(dict(group=g, component=c, n_stays=int(s[c].sum()), of=len(s)))
    return f, pd.DataFrame(br)


def sheet_repro(D, IMPS):
    rows = []

    def one(label, d, f, cc=True, mi=False):
        r = dict(step=label, N_VP=int((d.vp == 1).sum()), N_NE=int((d.vp == 0).sum()))
        for y in YS:
            r[f"events_{y}"] = "VP %d / NE %d" % (d[d.vp == 1][y].sum(), d[d.vp == 0][y].sum())
            if mi:
                o = or_mi(d, IMPS, y)["vp"]
                r[y] = fmt(o)
                r["n_model"] = len(d)
            else:
                dd = d.dropna(subset=[c for c in ["na_pre", "map_base", "lactate_pre24", "log_press_h"] if c in f])
                rr = glm_fit(f"{y}~vp+{f}", dd)
                lo, hi = np.exp(rr.conf_int().loc["vp"])
                r[y] = "%.2f (%.2f–%.2f)" % (np.exp(rr.params["vp"]), lo, hi)
                r["n_model"] = len(dd)
        rows.append(r)
    D = D.copy()
    D["kdigo0"] = D.kdigo_pre.fillna(0)
    m0 = cohort(D, neuro=False)
    one("1. initial definitions (dobutamine counted as pressor at T0; KDIGO missing=0)",
        D[m0 & (D.old_other_at_t0 == 0) & (D.pre_poly == 0)], COV_INITIAL)
    one("2. + vasopressor definition at T0 without inotropes (inclusive boundary endtime>=T0)",
        D[m0 & (D.vaso_at_t0_incl == 0) & (D.pre_poly == 0)], COV_INITIAL)
    one("2b. + exclusive boundary (pressor stopped exactly at T0 = not running)",
        D[m0 & (D.vaso_at_t0 == 0) & (D.pre_poly == 0)], COV_INITIAL)
    P = D[populations(D)["primary"]]
    one("3. + neuroendocrine exclusions (= primary population), initial covariates", P, COV_INITIAL)
    one("4. + full covariate set, KDIGO unknown category (complete case)", P, COV_CC)
    one("5. + MICE m=%d (PRIMARY estimate)" % M_IMP, P, COV_MI, mi=True)
    t = pd.DataFrame(rows)
    ev = D[cohort(D, neuro=False) & (D.pre_poly == 0)]
    note = pd.DataFrame([dict(item="evaluable new-onset VP stays reclassified to 'no vasopressor at T0' by excluding inotropes",
                              n=int(((ev.grp == "VP") & (ev.old_other_at_t0 == 1) & (ev.vaso_at_t0_incl == 0)).sum())),
                         dict(item="same, NE-only group",
                              n=int(((ev.grp == "NE") & (ev.old_other_at_t0 == 1) & (ev.vaso_at_t0_incl == 0)).sum())),
                         dict(item="evaluable new-onset VP stays moved into 'no vasopressor at T0' by the exclusive boundary (other pressor stopped exactly at T0)",
                              n=int(((ev.grp == "VP") & (ev.vaso_at_t0_incl == 1) & (ev.vaso_at_t0 == 0)).sum())),
                         dict(item="same (exclusive boundary), NE-only group",
                              n=int(((ev.grp == "NE") & (ev.vaso_at_t0_incl == 1) & (ev.vaso_at_t0 == 0)).sum())),
                         dict(item="VP evaluable stays excluded because T0 lies outside the ICU stay",
                              n=int(((D.grp == "VP") & (D.ex_notinicu == 1) & (D.obs24 >= 12) & (D.rrt_any == 0)).sum()))])
    return t, note


def describe(d, cols_cont, cols_bin):
    rows = []
    for c in cols_cont:
        r = dict(variable=c)
        for g in ("VP", "NE"):
            s = d[d.grp == g][c]
            r[g] = "%.1f [%.1f–%.1f]" % (s.median(), s.quantile(.25), s.quantile(.75))
            r[f"missing_{g}"] = "%d (%.1f%%)" % (s.isna().sum(), 100 * s.isna().mean())
        rows.append(r)
    for c in cols_bin:
        r = dict(variable=c)
        for g in ("VP", "NE"):
            s = d[d.grp == g][c]
            r[g] = "%d (%.1f%%)" % (s.sum(), 100 * s.mean())
            r[f"missing_{g}"] = "0"
        rows.append(r)
    for k in ("0", "1", "2", "3", "unk"):
        r = dict(variable=f"KDIGO pre-T0 = {k}")
        for g in ("VP", "NE"):
            s = d[d.grp == g].kdigo_cat == k
            r[g] = "%d (%.1f%%)" % (s.sum(), 100 * s.mean())
        rows.append(r)
    return pd.DataFrame(rows)


CONT = ["anchor_age", "weight", "na_pre", "map_base", "lactate_pre24", "fluid_pre24_l", "press_h", "ne_peak",
        "idx_ep1_h", "obs24"]
BIN = ["male", "cvicu", "diur_pre24", "other_vaso_pre", "dopamine_pre", "vent_t0", "steroid_pre24", "inotrope_t0",
       "milrinone_t0", "dob_at_t0", "sepsis", "card_surg", "emergency", "brain_any", "restart24", "hospital_expire_flag"]


def outcome_rows(d, imps, ys, pop, section, cov=COV_MI, label="", forest=True):
    rows = []
    for y in ys:
        o = or_mi(d, imps, y, cov=cov)["vp"]
        r = dict(analysis=label or section, population=pop, outcome=y, VP=nN(d, y, "VP"), NE=nN(d, y, "NE"),
                 adj_OR_MI=fmt(o), p=round(o["p"], 4), n_model=len(d), OR=o["OR"], lo=o["lo"], hi=o["hi"])
        rows.append(r)
        if forest:
            add_forest(section, label or section, pop, y, "adjusted OR (MICE)", o, d, y)
    return rows


def sheet_primary(D, IMPS, PP):
    rows = []
    for pop in ("primary", "secondary"):
        d = D[PP[pop]]
        cov = COV_MI if pop == "primary" else COV_MI + "+vaso_at_t0"
        covcc = COV_CC if pop == "primary" else COV_CC + "+vaso_at_t0"
        for y in YS:
            o = or_mi(d, IMPS, y, cov=cov)["vp"]
            c0 = or_mi(d, IMPS, y, cov=None)["vp"]
            cc = or_cc(d, y, cov=covcc)
            rows.append(dict(population=pop, outcome=y, VP=nN(d, y, "VP"), NE=nN(d, y, "NE"),
                             crude_OR=fmt(c0), adj_OR_MI_PRIMARY=fmt(o), p_MI=round(o["p"], 4),
                             n_MI=len(d), adj_OR_completecase=fmt(cc), n_CC=cc["n"],
                             patients=d.subject_id.nunique()))
            add_forest("primary", "adjusted logistic (MICE)", pop, y, "adjusted OR (MICE)", o, d, y)
            add_forest("primary", "crude logistic", pop, y, "crude OR", c0, d, y)
            add_forest("primary", "adjusted logistic (complete case)", pop, y, "adjusted OR (CC)", cc, d, y)
    return pd.DataFrame(rows)


def sheet_ipw(P, IMPS):
    t = time.time()
    pt, Wbar = ipw_point(P, IMPS)
    BS = bootstrap(P, IMPS, NBOOT)
    say("bootstrap %d replicates %.0fs" % (len(BS), time.time() - t))
    q = lambda k: np.nanpercentile(BS[k], [2.5, 97.5])
    rows = []
    for y in YS:
        rd, rr, orr = q(f"{y}_RD"), q(f"{y}_RR"), q(f"{y}_OR")
        ev = evalue(pt[f"{y}_RR"], rr[0])
        rows.append(dict(outcome=y, risk_VP=round(pt[f"{y}_r1"], 4), risk_NE_ATTweighted=round(pt[f"{y}_r0"], 4),
                         RD="%.3f (%.3f–%.3f)" % (pt[f"{y}_RD"], rd[0], rd[1]),
                         RR="%.2f (%.2f–%.2f)" % (pt[f"{y}_RR"], rr[0], rr[1]),
                         OR="%.2f (%.2f–%.2f)" % (pt[f"{y}_OR"], orr[0], orr[1]),
                         E_value_RR=round(ev["E_point"], 2), E_value_CI=round(ev["E_CI"], 2),
                         n=len(P), N_VP=int(P.vp.sum()), N_NE=int((P.vp == 0).sum()), bootstrap_reps=len(BS)))
        add_forest("IPW", "ATT IPW", "primary", y, "IPW OR", dict(OR=pt[f"{y}_OR"], lo=orr[0], hi=orr[1]), P, y)
        add_forest("IPW", "ATT IPW", "primary", y, "IPW RD", dict(OR=pt[f"{y}_RD"], lo=rd[0], hi=rd[1]), P, y)
    cifrows = []
    for g in (8, 24, 72):
        c = q(f"cif{g}_RD")
        cifrows.append(dict(hour=g, CIF_VP=round(pt[f"cif{g}_VP"], 4), CIF_NE_IPW=round(pt[f"cif{g}_NE"], 4),
                            RD_IPW="%.3f (%.3f–%.3f)" % (pt[f"cif{g}_RD"], c[0], c[1])))
    w_ne = Wbar[P.vp.values == 0]
    wsum = pd.DataFrame([dict(item="NE weights (imputation-averaged)", min=w_ne.min(), median=np.median(w_ne),
                              max=w_ne.max(), ESS_NE=w_ne.sum() ** 2 / (w_ne ** 2).sum(), N_NE=len(w_ne))])
    return pd.DataFrame(rows), pd.DataFrame(cifrows), wsum, Wbar, BS, pt


def cif_curves(d, w, label):
    out = []
    T, Cc = event_data(d)
    names = {1: "polyuria", 2: "death", 3: "ICU discharge", 4: "vasopressor restart"}
    for g in ("VP", "NE"):
        s = (d.grp == g).values
        for wt, wl in ((None, "crude"), (w, "IPW_ATT")):
            if wl == "IPW_ATT" and w is None:
                continue
            o = aj(T[s], Cc[s], None if wt is None else wt[s])
            for c, nm in names.items():
                out.append(pd.DataFrame(dict(analysis=label, group=g, weighting=wl, cause=nm, hour=np.arange(73),
                                             cif=o[c], n_at_risk=o["nrisk"])))
    return pd.concat(out)


def sheet_cox(P, Wbar):
    T, Cc = event_data(P)
    rows = []
    for (lo, hi) in ((0, 8), (8, 24), (0, 24)):
        for wl, w in (("crude", np.ones(len(P))), ("IPW_ATT", Wbar)):
            o = cs_cox(P, w, T, Cc, lo, hi)
            rows.append(dict(window=f"{lo}–{hi} h", weighting=wl, csHR="%.2f (%.2f–%.2f)" % (o["HR"], o["lo"], o["hi"]),
                             p=round(o["p"], 4), n_at_risk=o["n"], events_VP=o["events_VP"], events_NE=o["events_NE"]))
            add_forest("cs-Cox", f"cause-specific Cox {wl}", "primary", f"{lo}-{hi} h", "csHR",
                       dict(OR=o["HR"], lo=o["lo"], hi=o["hi"], p=o["p"]))
    return pd.DataFrame(rows)


def sheet_internal(D, IMPS, PP):
    d = D[PP["secondary"] & (D.grp == "VP")].copy()
    d["ne_running_t0"] = d.ne_at_t0.fillna(0).astype(int)
    d["ne_stopped_t0"] = 1 - d.ne_running_t0
    d["no_vaso_t0"] = 1 - d.vaso_at_t0
    rows = []
    for expo, lab in (("ne_stopped_t0", "NE already stopped at T0 vs NE running"),
                      ("no_vaso_t0", "no vasopressor at T0 vs another vasopressor continuing")):
        for y in YS:
            o = or_mi(d, IMPS, y, expo=expo)[expo]
            e1, e0 = d[d[expo] == 1], d[d[expo] == 0]
            rows.append(dict(comparison=lab, outcome=y,
                             exposed="%d/%d (%.1f%%)" % (e1[y].sum(), len(e1), 100 * e1[y].mean()),
                             reference="%d/%d (%.1f%%)" % (e0[y].sum(), len(e0), 100 * e0[y].mean()),
                             adj_OR_MI=fmt(o), p=round(o["p"], 4), n=len(d)))
            add_forest("internal VP", lab, "VP secondary", y, "adjusted OR (MICE)", o)
    return pd.DataFrame(rows)


def sheet_subgroups(P, IMPS):
    rows = []
    for sub, lab in (("sepsis", "sepsis (ICD)"), ("card_surg", "cardiac surgery"), ("cvicu", "CVICU"),
                     ("press_ge6", "vasopressor >= 6 h before T0")):
        cov = cov_without(sub) if sub != "press_ge6" else COV_MI
        for y in YS:
            pint = or_mi(P, IMPS, y, expo=f"vp*{sub}", cov=cov, terms=[f"vp:{sub}"])[f"vp:{sub}"]["p"]
            for lev in (1, 0):
                d = P[P[sub] == lev]
                o = or_mi(d, IMPS, y, cov=cov)["vp"]
                rows.append(dict(subgroup=lab, level=lev, outcome=y, VP=nN(d, y, "VP"), NE=nN(d, y, "NE"),
                                 adj_OR_MI=fmt(o), p_interaction=round(pint, 4)))
                add_forest("subgroup", f"{lab}={lev}", "primary", y, "adjusted OR (MICE)", o, d, y)
    return pd.DataFrame(rows)


def noagent_outcome(d, coint, hours=6):
    """Polyuria with no loop diuretic, mannitol or hypertonic saline in the 6 h before onset
    (from 6 h before the start of the 2-h run to its end)."""
    on = d.set_index("stay_id").onset
    c = coint[coint.stay_id.isin(set(on.dropna().index))].copy()
    c["on"] = c.stay_id.map(on)
    hit = c[(c.he >= c.on - 2 - hours) & (c.h <= c.on)]
    bad = set(hit.stay_id)
    out = d.copy()
    for y, lo, hi in (("poly24", 0, 24), ("poly8", 0, 8), ("poly_8_24", 8, 24)):
        out[y + "_noagent"] = ((out.onset > lo) & (out.onset <= hi) & ~out.stay_id.isin(bad)).astype(int)
    out["agent_before_onset"] = out.stay_id.isin(bad).astype(int)
    return out


def ipew_or(R, imps, y):
    acc = ([], [])
    for imp in imps:
        x = with_imp(R, imp)
        pm = smf.glm(f"eval12~vp+{COV_MI}", x, family=sm.families.Binomial()).fit().predict(x)
        xe = x[x.eval12 == 1]
        w = 1 / pm[xe.index].clip(0.01, 1)
        r = glm_fit(f"{y}~vp+{COV_MI}", xe, w.values)
        acc[0].append(r.params["vp"])
        acc[1].append(r.cov_params().loc["vp", "vp"])
    q, se, lo, hi, p = rubin(*acc)
    return dict(OR=np.exp(q), lo=np.exp(lo), hi=np.exp(hi), p=p)


def sheet_sensitivity(D, IMPS, PP, coint, alt_builds):
    rows = []
    P = D[PP["primary"]]

    def add(label, d, ys, imps=IMPS, cov=COV_MI, rename=None):
        for y in ys:
            o = or_mi(d, imps, y, cov=cov)["vp"]
            yl = rename.get(y, y) if rename else y
            rows.append(dict(analysis=label, outcome=yl, VP=nN(d, y, "VP"), NE=nN(d, y, "NE"),
                             adj_OR_MI=fmt(o), p=round(o["p"], 4), n=len(d), patients=d.subject_id.nunique()))
            add_forest("sensitivity", label, "primary-variant", yl, "adjusted OR (MICE)", o, d, y)
    v0 = D.vaso_at_t0 == 0
    add("main (reference)", P, YS)
    add("(a) include stays polyuric before T0", D[cohort(D) & v0], YS)
    add("(b) evaluability >= 6 bins", D[cohort(D, obs_thr=6) & v0 & (D.pre_poly == 0)], YS)
    add("(b) evaluability >= 18 bins", D[cohort(D, obs_thr=18) & v0 & (D.pre_poly == 0)], YS)
    R = D[cohort(D, obs_thr=0) & v0 & (D.pre_poly == 0)]
    for y in YS:
        o = ipew_or(R, IMPS, y)
        Re = R[R.eval12 == 1]
        rows.append(dict(analysis="(c) risk set, inverse-probability-of-evaluability weighted", outcome=y,
                         VP=nN(Re, y, "VP"), NE=nN(Re, y, "NE"), adj_OR_MI=fmt(o), p=round(o["p"], 4),
                         n=len(R), patients=R.subject_id.nunique()))
        add_forest("sensitivity", "(c) risk set IPEW", "risk set", y, "adjusted OR (MICE, IPEW)", o)
    add("(d) 24-h volume definition (>3 L or >50 mL/kg)", P, ["vol24"])
    add("(e) interval-allocated urine output", P, ["poly24_alloc", "poly8_alloc", "poly_8_24_alloc"])
    add("(f) polyuria counted only before vasopressor restart (restart = competing)", P,
        ["poly24_prerestart", "poly8_prerestart", "poly_8_24_prerestart"])
    add("(g) stays without vasopressor restart within 24 h", P[P.restart24 == 0], YS)
    Pn = noagent_outcome(P, coint)
    add("(h) polyuria without loop diuretic/mannitol/hypertonic saline in preceding 6 h", Pn,
        ["poly24_noagent", "poly8_noagent", "poly_8_24_noagent"])
    add("(i) exclude any coded neurological diagnosis (brain_any)", D[cohort(D, brain_any=True) & v0 & (D.pre_poly == 0)], YS)
    for gap, (Dg, IMPg) in alt_builds.items():
        PPg = populations(Dg)
        add(f"(j) episode-joining threshold {gap} h", Dg[PPg["primary"]], YS, imps=IMPg)
    first = P.sort_values("intime").groupby("subject_id").head(1)
    add("(k) one stay per patient (first eligible stay)", first, YS)
    Pg = D[cohort(D, obscol="obs24_nogu") & v0 & (D.pre_poly_nogu == 0)]
    add("(l) GU irrigant (227489) excluded from urine", Pg, ["poly24_nogu", "poly8_nogu", "poly_8_24_nogu"])
    Pr = D[cohort(D, rrt_rule="pre") & v0 & (D.pre_poly == 0)]
    add("(m) RRT: exclude pre-T0 RRT only; polyuria after RRT start not counted", Pr,
        ["poly24_prerrt", "poly8_prerrt", "poly_8_24_prerrt"])
    for y in YS:
        cc = or_cc(P, y)
        rows.append(dict(analysis="(n) complete case (KDIGO unknown as category)", outcome=y, VP=nN(P, y, "VP"),
                         NE=nN(P, y, "NE"), adj_OR_MI=fmt(cc), p=round(cc["p"], 4), n=cc["n"]))
        add_forest("sensitivity", "(n) complete case", "primary", y, "adjusted OR (CC)", cc)
    m_narrow = (D.ex_age == 0) & (D.ex_notinicu == 0) & (D.ex_deadpre == 0) & (D.rrt_any == 0) & (D.obs24 >= 12) & \
            (D.pit_di_codes == 0) & (D.brain_death == 0) & (D.pit_surg == 0)
    add("(o) neuroendocrine exclusion without the pituitary tumour codes",
        D[m_narrow & v0 & (D.pre_poly == 0)], YS)
    add("(p) secondary population (other vasopressor may continue at T0; +vaso_at_t0 covariate)",
        D[PP["secondary"]], YS, cov=COV_MI + "+vaso_at_t0")
    add("(q) initial covariate set (MICE)", P, YS,
        cov="age10+male+cvicu+C(kdigo_i)+na_i+map_i+diur_pre24+fluid_pre24_l+lact_i+log_press_h+ne_peak_c+other_vaso_pre")
    return pd.DataFrame(rows), R, Pn, Pr


def sheet_added_sensitivity(D, IMPS, PP):
    """Additional sensitivity analyses: one condition changed at a time from the primary population
    (adjusted OR, MICE)."""
    rows = []
    P = D[PP["primary"]]
    base_nn = (D.ex_age == 0) & (D.ex_notinicu == 0) & (D.ex_deadpre == 0) & (D.rrt_any == 0) & (D.obs24 >= 12) & \
              (D.pre_poly == 0)

    def add(label, d, ys=YS, cov=COV_MI):
        for y in ys:
            o = or_mi(d, IMPS, y, cov=cov)["vp"]
            rows.append(dict(analysis=label, outcome=y, VP=nN(d, y, "VP"), NE=nN(d, y, "NE"), adj_OR_MI=fmt(o),
                             p=round(o["p"], 4), n=len(d), patients=d.subject_id.nunique()))
            add_forest("robustness", label, "primary-variant", y, "adjusted OR (MICE)", o, d, y)
    add("main (reference)", P)
    add("(w) inclusive boundary: pressor stopped exactly at T0 counted as running", D[base_nn & (D.ex_neuroendo == 0) & (D.vaso_at_t0_incl == 0)])
    add("(r) SIADH (253.6/E22.2) not excluded", D[base_nn & (D.ex_neuroendo_nosiadh == 0) & (D.vaso_at_t0 == 0)])
    add("(s1) + number of recorded hourly bins 0-24 h as covariate", P, cov=COV_MI + "+obs24")
    add("(s2) restricted to >=21 recorded hourly bins", P[P.obs24 >= 21])
    add("(s3) restricted to 24/24 recorded hourly bins", P[P.obs24 == 24])
    add("(t1) NE stays with VP in another ICU stay of the same admission excluded", P[P.ne_vp_same_hadm == 0])
    add("(t2) NE stays with VP in same admission (other ICU stay or prescription) excluded",
        P[(P.ne_vp_same_hadm == 0) & (P.ne_vp_rx_hadm == 0)])
    add("(u) first index episode < 1 h excluded", P[P.idx_ep1_h >= 1])
    add("(v) another vasopressor started within 1 h after T0 (switch) excluded", P[P.switch_1h == 0])
    both = set(P[P.grp == "VP"].subject_id) & set(P[P.grp == "NE"].subject_id)
    add("(x) patients contributing to both groups excluded (%d patients)" % len(both), P[~P.subject_id.isin(both)])
    return pd.DataFrame(rows)


def sheet_na_detection(P, IMPS):
    """Detection bias of Na >= 150 (number of tests, fixed time windows, assay)."""
    d = P[P.na_max_pre24 < 150].copy()
    d["log_na_n72"] = np.log1p(d.na_n72)
    d["na150_fixed"] = ((d.na_fix24 >= 150) | (d.na_fix48 >= 150)).astype(int)
    d["na150_chem"] = (d.na_peak_72_chem >= 150).astype(int)
    rows = []
    for g in ("VP", "NE"):
        s = d[d.grp == g]
        rows.append(dict(analysis="Na tests 0-72 h, median [IQR] / mean", group=g,
                         value="%.0f [%.0f-%.0f] / %.2f" % (s.na_n72.median(), s.na_n72.quantile(.25), s.na_n72.quantile(.75), s.na_n72.mean())))
    specs = [("Na>=150 72 h (main definition)", d, "na150_72", COV_MI),
             ("+ log(1 + number of Na tests 0-72 h) as covariate", d, "na150_72", COV_MI + "+log_na_n72"),
             ("restricted to >=3 Na tests 0-72 h", d[d.na_n72 >= 3], "na150_72", COV_MI),
             ("fixed-time windows (max Na 18-30 h or 42-54 h >=150; both windows measured)",
              d[d.na_fix24.notna() & d.na_fix48.notna()], "na150_fixed", COV_MI),
             ("chemistry (indirect ISE) Na only", d, "na150_chem", COV_MI)]
    for lab, x, y, cov in specs:
        o = or_mi(x, IMPS, y, cov=cov)["vp"]
        rows.append(dict(analysis=lab, group="VP vs NE", value="VP %s; NE %s; aOR %s; n=%d" % (nN(x, y, "VP"), nN(x, y, "NE"), fmt(o), len(x))))
        add_forest("clinical-detection", lab, "primary", y, "adjusted OR (MICE)", o, x, y)
    return pd.DataFrame(rows)


def sheet_riskset_cif(R, IMPS):
    w = np.mean([ps_weights(with_imp(R, imp))[0] for imp in IMPS], axis=0)
    rows = []
    T, Cc = event_data(R, comp=("death", "icuout"))
    curves = []
    for g in ("VP", "NE"):
        s = (R.grp == g).values
        for wl, wt in (("crude", None), ("IPW_ATT", w)):
            o = aj(T[s], Cc[s], None if wt is None else wt[s], causes=(1, 2, 3))
            rows.append(dict(group=g, weighting=wl, n=int(s.sum()), CIF8=round(o[1][8], 4), CIF24=round(o[1][24], 4),
                             CIF72=round(o[1][72], 4)))
            for c, nm in {1: "polyuria", 2: "death", 3: "ICU discharge"}.items():
                curves.append(pd.DataFrame(dict(analysis="risk set (sens c): death/discharge competing", group=g,
                                                weighting=wl, cause=nm, hour=np.arange(73), cif=o[c],
                                                n_at_risk=o["nrisk"])))
    return pd.DataFrame(rows), pd.concat(curves)


def sheet_restart_censored_cif(P, Wbar):
    T, Cc = event_data(P, comp=("death", "icuout"), extra_censor=P.restart_t.values)
    rows = []
    for g in ("VP", "NE"):
        s = (P.grp == g).values
        for wl, wt in (("crude", None), ("IPW_ATT", Wbar)):
            o = aj(T[s], Cc[s], None if wt is None else wt[s], causes=(1,))
            rows.append(dict(analysis="(f) restart treated as censoring; death/discharge competing", group=g,
                             weighting=wl, CIF8=round(o[1][8], 4), CIF24=round(o[1][24], 4), CIF72=round(o[1][72], 4)))
    return pd.DataFrame(rows)


def sheet_secondary_outcomes(P, IMPS):
    rows = outcome_rows(P, IMPS, ["marked24", "vol24", "poly24_alloc", "poly72", "severe_wd"], "primary",
                        "secondary outcomes")
    return pd.DataFrame(rows).drop(columns=["OR", "lo", "hi"])


def sheet_clinical(P, IMPS, Wbar):
    rows = []
    spec = [("na150_72", "Na >= 150 within 72 h (pre-T0 24-h Na all < 150)", P.na_max_pre24 < 150),
            ("na_rise8_72", "Na rise >= 8 mmol/L within 72 h (pre-T0 Na measured)", P.na_pre.notna()),
            ("ddavp_ad_48", "antidiuretic desmopressin within 48 h (IV/SC/nasal; hemostatic doses excluded)", None),
            ("restart24", "vasopressor restart within 24 h", None),
            ("severe_wd", "severe water diuresis (polyuria 72 h + Na>=145 7 d + [Uosm<300 or antidiuretic DDAVP])", None)]
    for y, lab, sub in spec:
        d = P if sub is None else P[sub]
        w = Wbar if sub is None else Wbar[sub.values]
        vp = d.vp.values == 1
        e1, e0 = int(d[vp][y].sum()), int(d[~vp][y].sum())
        r1, r0 = d[y].values[vp].mean(), np.average(d[y].values[~vp], weights=w[~vp])
        fisher = stats.fisher_exact([[e1, vp.sum() - e1], [e0, (~vp).sum() - e0]])[1]
        if min(e1, e0) < 5:
            adj, pv, note = "not estimable (<5 events in a group)", np.nan, "sparse: counts and Fisher exact test only"
        else:
            o = or_mi(d, IMPS, y)["vp"]
            adj, pv = fmt(o), round(o["p"], 4)
            note = "sparse (<20 events in a group): interpret with caution" if min(e1, e0) < 20 else ""
            add_forest("clinical", lab, "primary", y, "adjusted OR (MICE)", o, d, y)
        rows.append(dict(outcome=lab, variable=y, VP=nN(d, y, "VP"), NE=nN(d, y, "NE"), adj_OR_MI=adj,
                         p=pv, fisher_p=round(fisher, 4), IPW_risk_VP=round(r1, 4), IPW_risk_NE=round(r0, 4),
                         IPW_RD_point=round(r1 - r0, 4), n=len(d), note=note))
    rows.append(dict(outcome="in-hospital death (descriptive only)", variable="hospital_expire_flag",
                     VP=nN(P, "hospital_expire_flag", "VP"), NE=nN(P, "hospital_expire_flag", "NE")))
    return pd.DataFrame(rows)


def sheet_coint(P, Pn, Pr):
    vars_ = [("loop_post24", "loop diuretic IV 0–24 h"), ("mannitol_post24", "mannitol 0–24 h"),
             ("hts_post24", "hypertonic saline (3%/23.4%) 0–24 h"), ("glu250_24", "glucose >= 250 mg/dL 0–24 h"),
             ("inotrope_post24", "dobutamine/milrinone 0–24 h"), ("ddavp_any_48", "any desmopressin order 0–48 h"),
             ("ddavp_ad_48", "antidiuretic desmopressin 0–48 h")]
    rows = []
    for v_, lab in vars_:
        r = dict(co_intervention=lab)
        for g in ("VP", "NE"):
            s = P[P.grp == g]
            r[f"{g} all"] = "%d/%d (%.1f%%)" % (s[v_].sum(), len(s), 100 * s[v_].mean())
            for yv, yl in ((1, "polyuria24"), (0, "no polyuria24")):
                ss = s[s.poly24 == yv]
                r[f"{g} {yl}"] = "%d/%d (%.1f%%)" % (ss[v_].sum(), len(ss), 100 * ss[v_].mean())
        rows.append(r)
    r = dict(co_intervention="agent within 6 h before polyuria onset (among polyuria24)")
    for g in ("VP", "NE"):
        s = Pn[(Pn.grp == g) & (Pn.poly24 == 1)]
        r[f"{g} polyuria24"] = "%d/%d (%.1f%%)" % (s.agent_before_onset.sum(), len(s), 100 * s.agent_before_onset.mean())
    rows.append(r)
    r = dict(co_intervention="RRT start 0–24 h (population of sensitivity m: pre-T0 RRT excluded only)")
    for g in ("VP", "NE"):
        s = Pr[Pr.grp == g]
        r[f"{g} all"] = "%d/%d (%.1f%%)" % (s.rrt_start24.sum(), len(s), 100 * s.rrt_start24.mean())
    rows.append(r)
    return pd.DataFrame(rows)


def sheet_ddavp(Dd, P):
    x = Dd[Dd.stay_id.isin(set(P.stay_id)) & (Dd.h >= -24) & (Dd.h <= 168)].copy()
    x["window"] = np.where(x.h < 0, "-24–0 h", np.where(x.h <= 48, "0–48 h", "48–168 h"))
    x["dose_mcg_per_kg"] = x.dose_mcg / x.weight
    rows = []
    for (g, wdw, rt, hm), s in x.groupby(["grp", "window", "route_c", "hemostatic"]):
        rows.append(dict(group=g, window=wdw, route=rt, classified_hemostatic=hm, n_orders=len(s),
                         n_stays=s.stay_id.nunique(),
                         dose_mcg_median_IQR="%.1f [%.1f–%.1f]" % (s.dose_mcg.median(), s.dose_mcg.quantile(.25),
                                                                     s.dose_mcg.quantile(.75)),
                         dose_mcg_per_kg_median="%.3f" % s.dose_mcg_per_kg.median()))
    return pd.DataFrame(rows)


def sheet_spline(D, IMPS, PP):
    out_rows, curve_rows = [], []
    COV_SPL = {"adjusted (covariates minus log vasopressor h)": cov_without("log_press_h"),
               "minimal (age, sex, CVICU)": "age10+male+cvicu"}
    specs = [(pop, extra, cl, cv) for pop, extra in (("VP, secondary population", "+vaso_at_t0"),
                                                     ("VP, primary population", ""))
             for cl, cv in COV_SPL.items()]
    for pop, extra, cov_label, cov_spl in specs:
        d = D[PP["secondary" if "secondary" in pop else "primary"] & (D.grp == "VP")].copy()
        # exposure = length of the first VP episode (h); vasopressor time accumulated during the episode is not adjusted for
        x = d.idx_ep1_h.values
        knots = np.percentile(x, [5, 35, 65, 95])
        B = rcs_basis(x, knots)
        for j in range(B.shape[1]):
            d[f"s{j}"] = B[:, j]
        grid = np.arange(1, np.floor(np.percentile(x, 99)) + 1)
        Bg, Bref = rcs_basis(grid, knots), rcs_basis([12.0], knots)
        cons = {"24 vs 6 h": (24, 6), "48 vs 12 h": (48, 12), "72 vs 24 h": (72, 24)}
        for y in YS:
            pvals, acc_c, acc_g = [], {k: ([], []) for k in cons}, ([], [])
            for imp in IMPS:
                xi = with_imp(d, imp)
                f_lin = f"{y}~s0+{cov_spl}{extra}"
                f_spl = f"{y}~s0+s1+s2+{cov_spl}{extra}"
                l0 = smf.glm(f_lin, xi, family=sm.families.Binomial()).fit().llf
                r_ = smf.glm(f_spl, xi, family=sm.families.Binomial()).fit()
                pvals.append(stats.chi2.sf(2 * (r_.llf - l0), 2))
                rr = glm_fit(f_spl, xi)
                b = rr.params[["s0", "s1", "s2"]].values
                V = rr.cov_params().loc[["s0", "s1", "s2"], ["s0", "s1", "s2"]].values
                for k, (a1, a0) in cons.items():
                    dl = (rcs_basis([a1], knots) - rcs_basis([a0], knots))[0]
                    acc_c[k][0].append(dl @ b)
                    acc_c[k][1].append(dl @ V @ dl)
                dg = Bg - Bref
                acc_g[0].append(dg @ b)
                acc_g[1].append(np.einsum("ij,jk,ik->i", dg, V, dg))
            r = dict(population=pop, covariates=cov_label, outcome=y, n=len(d), events=int(d[y].sum()),
                     knots_h=", ".join("%.1f" % k for k in knots),
                     LR_nonlinearity_p_median=round(float(np.median(pvals)), 4),
                     LR_p_range="%.4f–%.4f" % (min(pvals), max(pvals)))
            for k in cons:
                q, se, lo, hi, p = rubin(*acc_c[k])
                r[k] = "%.2f (%.2f–%.2f)" % (np.exp(q), np.exp(lo), np.exp(hi))
            out_rows.append(r)
            E = np.array(acc_g[0])
            Vv = np.array(acc_g[1])
            m = len(IMPS)
            qb, ub, bb = E.mean(0), Vv.mean(0), E.var(0, ddof=1)
            tt = np.sqrt(ub + (1 + 1 / m) * bb)
            curve_rows.append(pd.DataFrame(dict(population=pop, covariates=cov_label, outcome=y, duration_h=grid, OR_vs_12h=np.exp(qb),
                                                lo=np.exp(qb - 1.96 * tt), hi=np.exp(qb + 1.96 * tt))))
    return pd.DataFrame(out_rows), pd.concat(curve_rows)


def sheet_restart_timing(P):
    rows = []
    for g in ("VP", "NE"):
        s = P[(P.grp == g) & (P.poly24 == 1)]
        rs = s.restart_t
        rows.append(dict(group=g, polyuria24=len(s), restart_before_onset=int((rs < s.onset).sum()),
                         restart_after_onset_within24=int(((rs >= s.onset) & (rs <= 24)).sum()),
                         no_restart_within24=int((rs.isna() | (rs > 24)).sum()),
                         restart24_all_stays=nN(P, "restart24", g),
                         restart_median_h_if_24=round(P[(P.grp == g) & (P.restart24 == 1)].restart_t.median(), 1)))
    return pd.DataFrame(rows)


def sheet_missing(P):
    cols = ["kdigo_pre", "na_pre", "map_base", "lactate_pre24", "press_h", "weight"]
    rows = []
    for c in cols:
        r = dict(variable=c)
        for g in ("VP", "NE"):
            s = P[P.grp == g][c]
            r[g] = "%d/%d (%.1f%%)" % (s.isna().sum(), len(s), 100 * s.isna().mean())
        rows.append(r)
    return pd.DataFrame(rows)


def onset_hist(P):
    rows = []
    for g in ("VP", "NE"):
        s = P[P.grp == g]
        for h in range(2, 73):
            rows.append(dict(group=g, onset_hour=h, n=int((s.onset == h).sum()), N=len(s)))
    return pd.DataFrame(rows)


# ===========================================================================
# Run and output
README = [
    ("script", "02_primary_analysis.py (MIMIC-IV v3.1). Post hoc comparator analysis."),
    ("vasopressor at T0", "'Vasopressor running at T0' = a non-index vasopressor row with starttime <= T0 < endtime; a pressor stopped at exactly T0 is not running, so stays liberated from all vasopressors simultaneously are retained. Same rule for ne_at_t0, dobutamine/milrinone at T0. The inclusive boundary (endtime >= T0) is examined in 18_added_sensitivity (w)."),
    ("unit", "ICU stay; patients reported alongside. Cluster-robust SE by subject_id. ORs 2 decimals."),
    ("groups", "VP: any vasopressin (222315) in the ICU stay, T0 = end of first VP episode. NE-only: norepinephrine (221906) and never VP in the stay, T0 = end of first NE episode. Episodes joined if gap <= 1 h."),
    ("vasopressors", "VP, NE, phenylephrine 221749, epinephrine 221289, dopamine 221662, angiotensin II 229709/229764. Inotropes (dobutamine 221653, milrinone 221986) are covariates, not 'vasopressor at T0'."),
    ("primary population", "adult; T0 within ICU stay; alive at T0; no RRT in the stay; no pituitary/hypothalamic/DI codes (ICD-9 253.x, 588.1, pituitary tumour 227.3/237.0/194.3; ICD-10 E22-E23, N25.1, D35.2/D44.3/C75.1), brain death (G93.82/348.82) or pituitary surgery; urine in >=12 of 24 hourly bins after T0; no polyuria in the 12 h before T0; no vasopressor running at T0."),
    ("outcome", "polyuria = urine >3 mL/kg/h for >=2 consecutive recorded hourly bins (run reset at an unrecorded hour); onset = end of 2nd hour; 0–24 h primary, 0–8 h and 8–24 h time-resolved. Weight = median patientweight of the pressor inputevents rows."),
    ("covariates", COV_MI),
    ("missing data", "MICE (statsmodels MICEData, PMM k=5), m=%d, imputed once on all extracted stays (VP+NE, %s), analysis populations are subsets; Rubin's rules. Imputed: KDIGO pre-T0, Na pre, MAP pre, log lactate. Complete case = sensitivity (n)." % (M_IMP, "auxiliary: outcome indicators, group, other covariates")),
    ("IPW", "ATT weights from PS logistic on the same covariates (per imputation); NE weights truncated at the 99th percentile. CI = patient-level bootstrap (%d replicates; replicate b uses imputation b mod m)." % NBOOT),
    ("competing risks", "hourly Aalen–Johansen 0–72 h; competing: death, ICU discharge, vasopressor restart (any vasopressor started after T0); ties -> polyuria. Validated against KM (single cause) and lifelines AalenJohansenFitter."),
    ("n/N", "every table prints events/N by group; 'n' columns = stays entering the model."),
]


def main():
    t_all = time.time()
    outdir = config.out_dir("mimic_02")
    check_extracts()
    build_caches(say)
    data = load_all(say)
    say("build gap 1.0")
    D, COINT, DD = build(data, 1.0)
    IMPS = impute(D)
    alt = {}
    for gap in (0.5, 2.0):
        say("build gap", gap)
        Dg, _, _ = build(data, gap)
        alt[gap] = (Dg, impute(Dg))
        Dg.to_parquet(cohort_file(PRIMARY_TABLE.replace(".parquet", f"_gap{gap}.parquet")))
    PP = populations(D)
    P = D[PP["primary"]]
    say("primary population:", counts(P), " secondary:", counts(D[PP["secondary"]]))
    D.assign(pop_primary=PP["primary"].astype(int), pop_secondary=PP["secondary"].astype(int)) \
        .to_parquet(cohort_file(PRIMARY_TABLE))

    SHEETS["00_README"] = pd.DataFrame(README, columns=["item", "description"])
    f, br = sheet_flow(D)
    SHEETS["01_flow"] = f
    SHEETS["01b_neuroendo_excl"] = br
    rp, note = sheet_repro(D, IMPS)
    SHEETS["02_reproduction"] = rp
    SHEETS["02b_repro_notes"] = note
    SHEETS["03_table1_primary"] = describe(P, CONT, BIN)
    SHEETS["03b_missing"] = sheet_missing(P)
    SHEETS["04_primary_logistic"] = sheet_primary(D, IMPS, PP)
    say("primary logistic done")
    ipw, cifsum, wsum, Wbar, BS, pt = sheet_ipw(P, IMPS)
    SHEETS["05_IPW_ATT"] = ipw
    SHEETS["05b_IPW_weights"] = wsum
    wl = [ps_weights(with_imp(P, imp))[0] for imp in IMPS]
    SHEETS["06_SMD"] = smd_table(P, IMPS, wl).reset_index().rename(columns={"index": "covariate"})
    curves = cif_curves(P, Wbar, "primary: competing death/ICU discharge/vasopressor restart")
    c24 = curves.pivot_table(index=["group", "weighting", "cause"], columns="hour", values="cif")[[8, 24, 72]].reset_index()
    SHEETS["07_CIF_summary"] = c24.round(4)
    SHEETS["07b_CIF_IPW_RD_boot"] = cifsum
    SHEETS["08_csCox"] = sheet_cox(P, Wbar)
    say("IPW/CIF/Cox done")
    SHEETS["09_internal_VP"] = sheet_internal(D, IMPS, PP)
    SHEETS["10_subgroups"] = sheet_subgroups(P, IMPS)
    say("internal/subgroups done")
    sens, R, Pn, Pr = sheet_sensitivity(D, IMPS, PP, COINT, alt)
    SHEETS["11_sensitivity"] = sens
    rs_cif, rs_curves = sheet_riskset_cif(R, IMPS)
    SHEETS["11b_riskset_CIF"] = rs_cif
    SHEETS["11c_restart_censored_CIF"] = sheet_restart_censored_cif(P, Wbar)
    say("sensitivity done")
    SHEETS["12_clinical"] = sheet_clinical(P, IMPS, Wbar)
    SHEETS["12b_na150_detection"] = sheet_na_detection(P, IMPS)
    SHEETS["18_added_sensitivity"] = sheet_added_sensitivity(D, IMPS, PP)
    SHEETS["13_cointerventions"] = sheet_coint(P, Pn, Pr)
    SHEETS["14_ddavp_dose_route"] = sheet_ddavp(DD, D[PP["secondary"]]).assign(population="secondary (all evaluable new-onset stays)")
    SHEETS["15_secondary_outcomes"] = sheet_secondary_outcomes(P, IMPS)
    sp, spc = sheet_spline(D, IMPS, PP)
    SHEETS["16_VP_duration_spline"] = sp
    SHEETS["17_restart_timing"] = sheet_restart_timing(P)
    oh = onset_hist(P)
    say("all analyses done")

    xl = os.path.join(outdir, "results_mimic_primary.xlsx")
    with pd.ExcelWriter(xl) as xw:
        for k, t in SHEETS.items():
            t.to_excel(xw, sheet_name=k[:31], index=False)
    pd.concat([curves, rs_curves]).to_csv(os.path.join(outdir, "fig_mimic_cif_hourly.csv"), index=False)
    pd.DataFrame(FOREST).to_csv(os.path.join(outdir, "fig_mimic_forest_estimates.csv"), index=False)
    spc.to_csv(os.path.join(outdir, "fig_mimic_vp_duration_spline.csv"), index=False)
    oh.to_csv(os.path.join(outdir, "fig_mimic_onset_hist.csv"), index=False)
    BS.to_csv(os.path.join(outdir, "bootstrap_ipw_replicates.csv"), index=False)
    key = {"primary_counts": counts(P), "secondary_counts": counts(D[PP["secondary"]]),
           "patients_by_group_primary": P.groupby("grp").subject_id.nunique().to_dict(),
           "ipw_point": pt.round(4).to_dict()}
    with open(os.path.join(outdir, "key_results.json"), "w") as f:
        json.dump(key, f, indent=1, default=str)
    say("total %.0f s" % (time.time() - t_all))
    say.write(os.path.join(outdir, "run_log.txt"))


if __name__ == "__main__":
    main()