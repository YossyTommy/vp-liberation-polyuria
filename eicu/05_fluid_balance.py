#!/usr/bin/env python3
"""
05_fluid_balance.py - eICU-CRD v2.0, cumulative fluid balance.

Cumulative fluid balance from ICU admission to T0b (liberation from all vasopressors) and a
re-analysis, with multiple imputation, of whether polyuria after liberation is explained by
excretion of retained resuscitation fluid. Post hoc, exploratory.

Populations and definitions are those of the earlier scripts
  * five-group analysis population = cohort_usage_groups.parquet, primary == 1 (04_usage_groups.py, T0b)
  * two-group primary              = cohort_primary.parquet, primary == 1 (01_primary_analysis.py)
    T0 of every two-group stay equals its T0b in the usage-group cohort (asserted below).
  * Statistical functions and the covariate list COV come from eicu_stats.py. The cumulative
    fluid overload "fo_pct" (and the dense-charting variant "fo_pct_dense") is added to the list
    of imputed covariates so that FO% is imputed when it is a covariate.
  * Firth regression (model-based SE) only when a group in the model has < 10 events.

Fluid variables (intakeOutput; offsets in minutes from ICU admission = 0): see eicu_fluid.py.
FO% = (input - output, L) / weight (kg) x 100 over (ICU admission, T0b], with the weight of the
usage-group cohort. Fluids given before ICU admission are not captured; records with negative
offsets are not used.

Optional environment variables: MICE_M (default 10), NBOOT (200).
Outputs
  OUT_DIR/eicu_05/   eICU_fluid_balance_results.xlsx, run_log.txt
  WORK_DIR/eicu/cohort/cohort_fluid_balance.parquet
  WORK_DIR/eicu/ex_fluid/   intakeOutput filtered to the classified stays (re-used when present)

Run:  python eicu/05_fluid_balance.py   (after 01_primary_analysis.py and 04_usage_groups.py)
"""
import os
import sys
import time
import warnings
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # eicu_* modules; needed when Python runs in safe-path mode
import eicu_stats  # noqa: E402
from eicu_common import COHORT_FLUID, COHORT_PRIMARY, COHORT_USAGE, RUN_LOG, RunLog, cohort_path, env_int, out_dir  # noqa: E402
from eicu_fluid import LDEF, LMAX, ato_fit, build_records, fluid_vars, load_intake_output  # noqa: E402
from eicu_stats import COV, MICE_M, MISS, fmt, hosp_boot, impute, logit_cluster, smd, smd2  # noqa: E402

NBOOT = env_int("NBOOT", 200)
MISS_FLUID = MISS + ["fo_pct", "fo_pct_dense"]              # FO% is imputed when it is a covariate
fit_or = partial(eicu_stats.fit_or, miss=MISS_FLUID)
fit_groups = partial(eicu_stats.fit_groups, miss=MISS_FLUID)

MAIN = ["NE-only", "VP-only", "NE-first", "Simultaneous", "VP-first"]
EXTRA = ["VP-first+other", "VP restarted"]
OUTS = [("polyuria_24h", "24 h"), ("poly8", "0–8 h"), ("poly_8_24", ">8–24 h"), ("na150_72h", "Na≥150 72 h (new-onset)")]
TWO = "2-group primary VP vs NE"
NEF = "NE-first vs NE-only"
H2H = "NE-first vs VP-first (head-to-head)"
POPS = {"all analysis stays": lambda d: d, "recorded (FO% computable)": lambda d: d[d.fo_rec == 1],
        "dense charting (intake in ≥50% of hours)": lambda d: d[d.fo_dense == 1]}
MODELS = {"crude": ([], False), "COV": (COV, True), "COV+FO%": (COV + ["fo_pct"], True),
          "COV+FO%(dense only observed)": (COV + ["fo_pct_dense"], True)}
DESCV = [("cum_win_h", "ICU admission–T0b, h", "{:.1f}"), ("cum_in_l", "cumulative input (admission–T0b), L", "{:.2f}"),
         ("cum_out_l", "cumulative output (admission–T0b), L", "{:.2f}"), ("cum_urine_l", "of which urine, L", "{:.2f}"),
         ("cum_bal_l", "cumulative balance (admission–T0b), L", "{:.2f}"), ("fo_pct", "FO% (admission–T0b)", "{:.1f}"),
         ("fo_pct_dense", "FO% (dense charting only)", "{:.1f}"),
         ("pre24_in_l", "pre-T0b 24 h input, L", "{:.2f}"), ("pre24_out_l", "pre-T0b 24 h output, L", "{:.2f}"),
         ("pre24_bal_l", "pre-T0b 24 h balance, L", "{:.2f}"), ("post8_in_l", "input 0–8 h after T0b, L", "{:.2f}"),
         ("post24_in_l", "input 0–24 h after T0b, L", "{:.2f}"), ("fluid_pre24_l", "(reference) usage-group cohort pre-T0b 24 h intake (not apportioned), L", "{:.2f}"),
         ("press_h_pre", "vasopressor start–T0b, h", "{:.1f}"), ("apache4", "APACHE IV", "{:.0f}")]


def nn(k, n):
    """'k/n (%)'."""
    return f"{int(k)}/{int(n)} ({100*k/n:.1f}%)" if n else "0/0"


def medi(s, f="{:.2f}"):
    """'median [Q1–Q3]' of the non-missing values."""
    s = pd.Series(s).dropna()
    return (f + " [" + f + "–" + f + "]").format(s.median(), s.quantile(.25), s.quantile(.75)) if len(s) else "—"


def pool(P, U):
    """Rubin's rules: pooled coefficients and total covariance matrix."""
    P = pd.concat(P, axis=1).T
    Q = P.mean()
    W = sum(U) / len(U)
    B = P.cov() if len(P) > 1 else 0 * W
    return Q, W + (1 + 1 / len(P)) * B


class Comparisons:
    """The analysis populations (P2 two-group primary, P5 five-group) and the group comparisons
    fitted on them; collects the result blocks."""

    def __init__(self, P2, P5, say):
        self.P2, self.P5, self.say = P2, P5, say
        self.spec = {TWO: ("grp", ["VP"], "NE"), NEF: ("grp", ["NE-first"], "NE-only"), H2H: ("grp", ["NE-first"], "VP-first")}
        self.res = {}

    def data(self, comp):
        if comp == TWO:
            return self.P2
        if comp == NEF:
            return self.P5[self.P5.grp.isin(["NE-only", "NE-first"])]
        return self.P5[self.P5.grp.isin(["VP-first", "NE-first"])]

    def fit(self, df, y, comp, model, analysis, pop, **extra):
        """One group comparison under one covariate model; returns the result row."""
        gcol, gl, ref = self.spec[comp]
        covs, mice = MODELS[model]
        d = df.dropna(subset=[y])
        r = fit_groups(d, y, gcol, gl, ref=ref, covs=covs, mice=mice, label=analysis)[0]
        r.update(comparison=comp, model=model, population=pop, outcome_label=dict(OUTS)[y],
                 n_N=f"{r['events_group']} vs {r['events_ref']}", **extra)
        return r

    def block(self, key, rows):
        df = pd.DataFrame(rows)
        self.res[key] = df
        cols = [c for c in ["analysis", "comparison", "population", "outcome_label", "model", "n_N", "OR_95CI", "method"] if c in df.columns]
        self.say(f"---- {key}\n" + df[cols].to_string(index=False))
        return df

    def grp_frames(self):
        fr = [(g, self.P5[self.P5.grp == g]) for g in MAIN + EXTRA]
        fr.append(("2-group primary VP (n=370)", self.P2[self.P2.grp == "VP"]))
        return fr


def reproduction(P2, P5, say):
    """Expected sizes and estimates of the two populations; raises when one is not reproduced."""
    REP = []

    def rep(item, val, target):
        REP.append(dict(item=item, value=val, target=target, ok=str(val) == str(target)))
        say("REPRO", item, val, "target", target)

    for g, n in (("NE-only", 3422), ("NE-first", 263), ("VP-first", 147), ("VP-only", 134), ("Simultaneous", 275)):
        rep(f"5-group analysis population {g}", int((P5.grp == g).sum()), n)
    rep("2-group VP n", int((P2.grp == "VP").sum()), 370)
    rep("2-group NE n", int((P2.grp == "NE").sum()), 3422)
    rep("2-group VP 24-h events", int(P2.loc[P2.grp == "VP", "polyuria_24h"].sum()), 81)
    rep("2-group NE 24-h events", int(P2.loc[P2.grp == "NE", "polyuria_24h"].sum()), 571)
    r = fit_or(P2, "polyuria_24h", "vp", COV, label="repro")
    rep(f"2-group 24-h aOR (MICE m={MICE_M}, fit_or)", f"{r['OR']:.2f}", "1.34" if MICE_M == 10 else f"{r['OR']:.2f}")
    m = P2[["patientunitstayid", "t0"]].merge(P5[["patientunitstayid", "t0"]], on="patientunitstayid", how="left", suffixes=("_two", "_usage"))
    assert m.t0_usage.notna().all() and (m.t0_two == m.t0_usage).all(), "two-group T0 differs from T0b of the usage-group cohort"
    rep("2-group stays in 5-group population with identical T0 (= T0b)", int(len(m)), 3792)
    assert all(r["ok"] for r in REP), "reproduction failed"
    return REP


def consistency_totals(io):
    """Agreement of the summed cell values with intaketotal / outputtotal / nettotal per stay and offset."""
    g_ = io[io.cat.isin(["Intake (ml)", "Output (ml)", "Dialysis (ml)"])]
    S_ = g_.pivot_table(index=["patientunitstayid", "intakeoutputoffset"], columns="cat", values="v", aggfunc="sum").fillna(0)
    T_ = io.groupby(["patientunitstayid", "intakeoutputoffset"])[["intaketotal", "outputtotal", "dialysistotal", "nettotal"]].max()
    J_ = S_.join(T_, how="inner")
    CONS = []
    for a, b in (("Intake (ml)", "intaketotal"), ("Output (ml)", "outputtotal"), ("Dialysis (ml)", "dialysistotal")):
        d = (J_[a] - J_[b]).abs()
        pos = J_[a] > 0
        CONS.append(dict(check=f"sum of '{a}' cells vs {b}", keys=len(J_), agree_le1ml_all=f"{int((d <= 1).sum())}/{len(J_)} ({100*(d <= 1).mean():.1f}%)",
                         keys_with_cells=int(pos.sum()), agree_le1ml_among_cells=f"{int((d[pos] <= 1).sum())}/{int(pos.sum())} ({100*(d[pos] <= 1).mean():.1f}%)",
                         median_abs_diff_when_disagree=float(d[d > 1].median()) if (d > 1).any() else 0.0))
    for lab_, expr in (("nettotal = intaketotal - outputtotal + dialysistotal", J_.intaketotal - J_.outputtotal + J_.dialysistotal),
                       ("nettotal = intaketotal - outputtotal - dialysistotal", J_.intaketotal - J_.outputtotal - J_.dialysistotal)):
        d = (expr - J_.nettotal).abs()
        CONS.append(dict(check=lab_, keys=len(J_), agree_le1ml_all=f"{int((d <= 1).sum())}/{len(J_)} ({100*(d <= 1).mean():.1f}%)"))
    return CONS


def coverage(cmp):
    """(a) Coverage of the fluid records per group and per hospital, and coverage predictors."""
    P5 = cmp.P5
    cov_rows = []
    for g, x in cmp.grp_frames():
        N = len(x)
        r = {"group": g, "N": N}
        for lev in ["intake+output", "intake only", "output only", "neither", "T0b<=ICU admission"]:
            r[f"ICU admission–T0b: {lev}"] = nn((x.cov_cum == lev).sum(), N)
        r["FO% computable (recorded)"] = nn(x.fo_rec.sum(), N)
        r["FO% dense charting (intake in ≥50% of hours)"] = nn(x.fo_dense.sum(), N)
        r["pre-T0b 24 h balance computable"] = nn(x.pre24_rec.sum(), N)
        r["intake recorded 0–8 h after T0b"] = nn(x.has_in_post8.sum(), N)
        r["intake recorded 0–24 h after T0b"] = nn(x.has_in_post24.sum(), N)
        r["fraction of hours with intake recorded (median [IQR])"] = medi(x.in_cum_hourfrac)
        r["hospitals"] = int(x.hospitalid.nunique())
        cov_rows.append(r)
    COVG = pd.DataFrame(cov_rows)
    _h = P5[P5.cov_cum != "T0b<=ICU admission"].groupby("hospitalid").agg(n=("has_in_cum", "size"), rec=("has_in_cum", "mean"))
    HOSP = pd.DataFrame([dict(item="hospitals (5-group analysis population)", value=len(_h)),
                         dict(item="hospitals with no intake record in any stay (stays)", value=f"{int((_h.rec == 0).sum())} ({int(_h.loc[_h.rec == 0, 'n'].sum())})"),
                         dict(item="hospitals with intake recorded in >=90% of stays (stays)", value=f"{int((_h.rec >= .9).sum())} ({int(_h.loc[_h.rec >= .9, 'n'].sum())})"),
                         dict(item="hospitals with intake recorded in 1-89% of stays (stays)", value=f"{int(((_h.rec > 0) & (_h.rec < .9)).sum())} ({int(_h.loc[(_h.rec > 0) & (_h.rec < .9), 'n'].sum())})")])
    # coverage predictors (does recording depend on group / severity?)
    _cp = []
    for comp in [TWO, NEF]:
        d = cmp.data(comp)
        gcol, gl, ref = cmp.spec[comp]
        for y in ("fo_rec", "fo_dense"):
            r = fit_groups(d, y, "grp", gl, ref=ref, covs=COV, mice=True, label=f"coverage model: {y}")[0]
            r.update(comparison=comp, n_N=f"{r['events_group']} vs {r['events_ref']}")
            _cp.append(r)
    return COVG, HOSP, pd.DataFrame(_cp)


def description(cmp):
    """(a) Fluid variables per group with the standardised mean difference versus NE-only."""
    ref = cmp.P5[cmp.P5.grp == "NE-only"]
    drow = []
    for c, lab_, f in DESCV:
        r = {"variable": lab_}
        for g, x in cmp.grp_frames():
            s = x[c]
            r[f"{g}"] = medi(s, f) + f"; n={int(s.notna().sum())}/{len(s)}"
            if g != "NE-only":
                r[f"SMD {g}"] = smd2(s, ref[c])
        drow.append(r)
    for c, lab_ in (("fo_rec", "FO% computable"), ("fo_dense", "dense charting"), ("press24", "vasopressor ≥24 h"), ("diur_post24", "loop diuretic within 24 h after T0b")):
        r = {"variable": lab_}
        for g, x in cmp.grp_frames():
            r[g] = nn(x[c].sum(), len(x))
            if g != "NE-only":
                r[f"SMD {g}"] = smd2(x[c], ref[c])
        drow.append(r)
    return pd.DataFrame(drow)


def fo_adjustment(cmp):
    """(b) Adjusted odds ratios before and after adding FO%."""
    rows = []
    for comp in [TWO, NEF]:
        D0 = cmp.data(comp)
        for y, _ in OUTS:
            for pop, f_ in POPS.items():
                mods = ["crude", "COV", "COV+FO%"] + (["COV+FO%(dense only observed)"] if pop == "all analysis stays" else [])
                for mdl in mods:
                    rows.append(cmp.fit(f_(D0), y, comp, mdl, "(b) before/after adding FO%", pop))
    cmp.block("b_adjOR", rows)


def interaction_test(d, y, expo, covs, strat):
    """Wald test of the exposure x stratum interaction (Rubin's rules when covariates are imputed)."""
    d = d.dropna(subset=[y, strat]).copy()
    lv = sorted(d[strat].astype(str).unique())[1:]
    xs = [expo]
    for lev in lv:
        d[f"S_{lev}"] = (d[strat].astype(str) == lev).astype(int)
        d[f"X_{lev}"] = d[f"S_{lev}"] * d[expo]
        xs += [f"S_{lev}", f"X_{lev}"]
    xs += covs
    sets = impute(d, covs, [y] + xs[:1 + 2 * len(lv)], MICE_M) if d[[c for c in MISS_FLUID if c in covs]].isna().any().any() else [d]
    P, U = [], []
    for s in sets:
        p, v = logit_cluster(s, y, xs)
        P.append(p)
        U.append(v)
    Q, T = pool(P, U)
    ix = [f"X_{lev}" for lev in lv]
    b = Q[ix].values
    V = T.loc[ix, ix].values
    st = float(b @ np.linalg.solve(V, b))
    return dict(chi2=round(st, 3), df=len(ix), p_interaction=chi2.sf(st, len(ix)), n=len(d), m=len(sets))


def tertiles(cmp, TC, tert):
    """(c) Odds ratios by FO% tertile in stays with records, and with FO% imputed (tertile
    assigned within each imputed data set)."""
    say = cmp.say
    crow, irow = [], []
    for comp in [TWO, NEF]:
        D0 = cmp.data(comp)
        D0 = D0[D0.fo_rec == 1].copy()
        D0["expo"] = (D0.grp == cmp.spec[comp][1][0]).astype(int)
        for y in ("poly8", "polyuria_24h"):
            for t_ in ("T1", "T2", "T3"):
                crow.append(cmp.fit(D0[D0.fo_t == t_], y, comp, "COV", "(c) by FO% tertile", f"recorded, {t_} (cut-points {TC[0]:.2f}/{TC[1]:.2f} %)", tertile=t_))
            it = interaction_test(D0, y, "expo", COV, "fo_t")
            it.update(comparison=comp, outcome=y, population="recorded", model="COV + tertile + exposure×tertile")
            irow.append(it)
    cmp.block("c_tertile", crow)
    cmp.res["c_interaction"] = pd.DataFrame(irow)
    say(cmp.res["c_interaction"].to_string(index=False))
    mrow = []
    for comp in [TWO, NEF]:
        D0 = cmp.data(comp).copy()
        D0["expo"] = (D0.grp == cmp.spec[comp][1][0]).astype(int)
        for y in ("poly8", "polyuria_24h"):
            sets = impute(D0, COV + ["fo_pct"], [y, "expo"], MICE_M)
            for t_ in ("T1", "T2", "T3"):
                P, U, ns, es, e1 = [], [], [], [], []
                for s in sets:
                    ss = s[tert(s.fo_pct) == t_]
                    p, v = logit_cluster(ss, y, ["expo"] + COV)
                    P.append(p)
                    U.append(v)
                    ns.append(len(ss))
                    es.append(int(ss[y].sum()))
                    e1.append(f"{int(ss.loc[ss.expo == 1, y].sum())}/{int(ss.expo.sum())}")
                Q, T = pool(P, U)
                se = np.sqrt(T.loc["expo", "expo"])
                mrow.append(dict(comparison=comp, outcome=y, tertile=t_, model="COV, logistic clustered SE, MICE m=10 (FO% imputed; tertile per imputation)",
                                 n_range=f"{min(ns)}–{max(ns)}", events_range=f"{min(es)}–{max(es)}", exposed_events_imp1=e1[0],
                                 OR=np.exp(Q["expo"]), lo=np.exp(Q["expo"] - 1.96 * se), hi=np.exp(Q["expo"] + 1.96 * se),
                                 OR_95CI=fmt(np.exp(Q["expo"]), np.exp(Q["expo"] - 1.96 * se), np.exp(Q["expo"] + 1.96 * se))))
            # interaction with imputed tertiles
            P, U = [], []
            for s in sets:
                s = s.copy()
                s["fo_t"] = tert(s.fo_pct).astype(str)
                xs = ["expo"]
                for lev in ("T2", "T3"):
                    s[f"S_{lev}"] = (s.fo_t == lev).astype(int)
                    s[f"X_{lev}"] = s[f"S_{lev}"] * s.expo
                    xs += [f"S_{lev}", f"X_{lev}"]
                p, v = logit_cluster(s, y, xs + COV)
                P.append(p)
                U.append(v)
            Q, T = pool(P, U)
            ix = ["X_T2", "X_T3"]
            b = Q[ix].values
            st = float(b @ np.linalg.solve(T.loc[ix, ix].values, b))
            mrow.append(dict(comparison=comp, outcome=y, tertile="interaction", model="exposure×tertile, 2 df Wald (Rubin)", n_range=str(len(D0)),
                             OR_95CI=f"chi2={st:.2f}, p={chi2.sf(st, 2):.3f}"))
    cmp.res["c_tertile_MI"] = pd.DataFrame(mrow)
    say(cmp.res["c_tertile_MI"].to_string(index=False))


def ne_only_fo(cmp, NE):
    """(d) NE-only stays: FO% quartiles, odds ratio per 5 points of FO% and per litre of the
    balance in the 24 h before T0b."""
    drow = []
    for pop, sub in (("recorded", NE[NE.fo_rec == 1]), ("dense charting", NE[NE.fo_dense == 1])):
        sub = sub.copy()
        sub["foq"] = pd.qcut(sub.fo_pct, 4, labels=["QI", "QII", "QIII", "QIV"])
        qb = sub.groupby("foq", observed=True).fo_pct.agg(["min", "max"])
        for y in ("poly8", "polyuria_24h", "poly_8_24"):
            r = dict(analysis="(d) NE-only: proportion by quartile", population=pop, outcome=y,
                     quartile_bounds=" / ".join(f"{q}: {qb.loc[q, 'min']:.1f} to {qb.loc[q, 'max']:.1f} %" for q in ["QI", "QII", "QIII", "QIV"]))
            for q in ["QI", "QII", "QIII", "QIV"]:
                x = sub[sub.foq == q]
                r[q] = nn(x[y].sum(), len(x))
            drow.append(r)
            o = fit_or(sub, y, "fo5", COV, label="(d) aOR per 5% FO% (COV; FO% not among the covariates)")
            drow.append(dict(analysis=o["analysis"], population=pop, outcome=y, n_N=f"{o['events']}/{o['n']}", aOR_95CI=o["aOR_95CI"], p=o["p"], method=o["method"],
                             OR=o["OR"], lo=o["lo"], hi=o["hi"]))
            for r_ in fit_groups(sub.assign(foq=sub.foq.astype(str)), y, "foq", ["QII", "QIII", "QIV"], ref="QI", covs=COV, label="(d) quartile aOR (ref QI)"):
                drow.append(dict(analysis=r_["analysis"], population=pop, outcome=y, group=r_["group"], n_N=f"{r_['events_group']} vs {r_['events_ref']}",
                                 aOR_95CI=r_["OR_95CI"], p=r_["p"], method=r_["method"], OR=r_["OR"], lo=r_["lo"], hi=r_["hi"]))
    # pre-24 h balance per 1 L (NE-only)
    s2 = NE[NE.pre24_rec == 1].copy()
    s2["bal1"] = s2.pre24_bal_l
    for y in ("poly8", "polyuria_24h"):
        o = fit_or(s2, y, "bal1", COV, label="(d) aOR per 1 L of pre-T0b 24 h balance")
        drow.append(dict(analysis=o["analysis"], population="pre-T0b 24 h balance computable", outcome=y, n_N=f"{o['events']}/{o['n']}", aOR_95CI=o["aOR_95CI"], p=o["p"], method=o["method"],
                         OR=o["OR"], lo=o["lo"], hi=o["hi"]))
    d = pd.DataFrame(drow)
    cmp.res["d_NEonly"] = d[[c for c in d.columns if c not in ("OR", "lo", "hi")] + ["OR", "lo", "hi"]]
    cmp.say(cmp.res["d_NEonly"].to_string(index=False))


def severe_fluid_loaded(cmp, FOMED):
    """(e) Stays with vasopressors for >= 24 h and FO% at or above the NE-only median."""
    erow = []
    for comp in [NEF, TWO]:
        D0 = cmp.data(comp)
        S = D0[(D0.press24 == 1) & (D0.fo_pct >= FOMED)]
        for y, _ in OUTS:
            for mdl in ("crude", "COV", "COV+FO%"):
                erow.append(cmp.fit(S, y, comp, mdl, "(e) vasopressor ≥24 h and FO% ≥ NE-only median", f"FO%≥{FOMED:.2f}% & vasopressor ≥24 h"))
    cmp.block("e_severe", erow)


def overlap_weighting(cmp):
    """(f) Overlap weighting in the two-group primary population: multiple imputation, a
    hospital-cluster bootstrap within each imputed data set, Rubin's rules."""
    P2 = cmp.P2
    frow, srow = [], []
    for popname, D0, ys in (("2-group primary", P2.copy(), ["polyuria_24h", "poly8", "poly_8_24"]),
                            ("2-group primary, Na evaluable", P2[P2.na150_72h.notna()].copy(), ["na150_72h"])):
        for spec, xcols in (("PS: COV", COV), ("PS: COV+FO%", COV + ["fo_pct"])):
            sets = impute(D0, xcols, ys + ["vp"], MICE_M)
            Q, Wb, boots, smds = [], [], [], []
            for k, s in enumerate(sets):
                s = s.reset_index(drop=True)
                est, w, ps = ato_fit(s, xcols, ys)
                Q.append(est)
                bt = hosp_boot(s, lambda b: ato_fit(b, xcols, ys)[0], B=NBOOT, seed=2026 + k)
                boots.append(bt)
                Wb.append(bt.var(axis=0, ddof=1))
                for c in COV + ["fo_pct", "pre24_bal_l", "lactate_pre24"]:
                    x = s[c].astype(float)
                    ok = x.notna().values
                    smds.append(dict(spec=spec, population=popname, variable=c, imp=k, before=smd(x[ok], s.vp.values[ok]), after=smd(x[ok], s.vp.values[ok], w[ok])))
                ess = [(w[s.vp == g].sum() ** 2) / (w[s.vp == g] ** 2).sum() for g in (1, 0)]
            Q = np.array(Q)
            Wb = np.array(Wb)
            m = len(Q)
            qb = Q.mean(0)
            Bv = Q.var(0, ddof=1) if m > 1 else 0 * qb
            Tv = Wb.mean(0) + (1 + 1 / m) * Bv
            allb = np.vstack(boots)
            for j, y in enumerate(ys):
                d_ = D0.dropna(subset=[y])
                k1 = int(d_.loc[d_.vp == 1, y].sum())
                n1 = int((d_.vp == 1).sum())
                k0 = int(d_.loc[d_.vp == 0, y].sum())
                n0 = int((d_.vp == 0).sum())
                se = np.sqrt(Tv[j])
                lo, hi = np.exp(qb[j] - 1.96 * se), np.exp(qb[j] + 1.96 * se)
                frow.append(dict(analysis="(f) overlap weighting (ATO)", population=popname, ps_model=spec, outcome=y, n_N=f"{nn(k1, n1)} vs {nn(k0, n0)}",
                                 OR=np.exp(qb[j]), lo=lo, hi=hi, OR_95CI=fmt(np.exp(qb[j]), lo, hi),
                                 pct_CI=f"{np.exp(np.percentile(allb[:, j], 2.5)):.2f}–{np.exp(np.percentile(allb[:, j], 97.5)):.2f}",
                                 method=f"MICE m={m}; hospital-cluster bootstrap B={NBOOT} per imputation (successful {min(len(b) for b in boots)}–{max(len(b) for b in boots)}); Rubin (bootstrap within-variance)",
                                 ESS_VP_NE_lastimp=f"{ess[0]:.0f} / {ess[1]:.0f}"))
            srow += smds
    cmp.res["f_ATO"] = pd.DataFrame(frow)
    cmp.say(cmp.res["f_ATO"].to_string(index=False))
    _s = pd.DataFrame(srow)
    cmp.res["f_ATO_SMD"] = _s.groupby(["population", "spec", "variable"], sort=False)[["before", "after"]].mean().round(3).reset_index()


def restricted_models(cmp):
    """(g) No loop diuretic after T0b; NE-first versus VP-first; long vasopressor course only."""
    grow = []
    for comp in [TWO, NEF]:
        D0 = cmp.data(comp)
        S = D0[D0.diur_post24 == 0]
        for y, _ in OUTS:
            for mdl in ("crude", "COV", "COV+FO%"):
                grow.append(cmp.fit(S, y, comp, mdl, "(g1) no loop diuretic within 24 h after T0b", "no loop diuretic (24 h after T0b)"))
    cmp.block("g_nodiuretic", grow)
    hrow = []
    D0 = cmp.data(H2H)
    for y, _ in OUTS:
        for pop, f_ in POPS.items():
            for mdl in (("crude", "COV", "COV+FO%") if pop == "all analysis stays" else ("COV", "COV+FO%")):
                hrow.append(cmp.fit(f_(D0), y, H2H, mdl, "(g2) head-to-head", pop))
    cmp.block("g_h2h", hrow)
    # the same models restricted to vasopressor support >= 24 h
    qrow = []
    for comp in [NEF, TWO]:
        D0 = cmp.data(comp)
        S = D0[D0.press24 == 1]
        for y in ("poly8", "polyuria_24h"):
            for mdl in ("COV", "COV+FO%"):
                qrow.append(cmp.fit(S, y, comp, mdl, "(g3) long vasopressor course only (vasopressor ≥24 h)", "vasopressor ≥24 h"))
    cmp.block("g_severe24", qrow)


def readme_sheet(TC):
    return pd.DataFrame({"item": [
        "script", "data", "populations", "functions", "new variables", "apportioning", "coverage", "FO%", "fluids before ICU admission", "models", "Firth", "tertiles", "ATO", "multiplicity", "reproduction"],
        "description": [
        "05_fluid_balance.py. Statistical functions are imported from eicu_stats.py, fluid-balance functions from eicu_fluid.py.",
        "eICU-CRD v2.0 intakeOutput.csv.gz (intakeoutputoffset, cellpath, celllabel, cellvaluenumeric).",
        "5 groups = cohort_usage_groups primary==1 (NE-only 3,422 / NE-first 263 / VP-first 147 / VP-only 134 / Simultaneous 275). 2-group primary = cohort_primary primary==1 (VP 370 / NE 3,422). T0 equals T0b in every stay.",
        "impute, logit_cluster, fit_or, fit_groups, firth_fit, hosp_boot, smd, smd2, COV and MISS are those of the two-group primary and usage-group analyses (eicu_stats.py). Only change: fo_pct / fo_pct_dense appended to MISS (so that FO% is imputed by MICE).",
        "cum_in_l/cum_out_l/cum_bal_l (admission–T0b, L), fo_pct, pre24_* (24 h before T0b), post8_in_l/post24_in_l (descriptive), cov_cum, fo_rec, fo_dense, in_cum_hourfrac.",
        f"A record covers (time of the previous record of the same label, charting time] when the gap is 0–{LMAX:.0f} min, otherwise the preceding {LDEF:.0f} min. Apportioned by overlap with the window.",
        "No record = missing (not 0). Balance and FO% need both intake and output records. T0b <= ICU admission = missing.",
        "FO% = (input − output, L) / weight kg × 100. Weight = weight_kg of the usage-group cohort. Dense charting = intake recorded in ≥50% of the hours from admission to T0b.",
        "Fluids given in the ED, operating room, ward or referring hospital are not in intakeOutput and cannot be counted (records with negative offsets are not used). FO% may be underestimated.",
        "Logistic regression, hospital-clustered robust SE, MICE m=10 (Rubin). Covariates COV (including APACHE IV).",
        "Firth (model-based SE, no clustering) only when a group has <10 events.",
        f"Cut on the FO% distribution of NE-only stays with records: {TC[0]:.2f} / {TC[1]:.2f} %. Main analysis: stays with records (observed FO%). Sensitivity: FO% imputed by MICE, tertiles assigned within each imputed data set.",
        f"PS = logistic on COV (+ FO%); weights VP: 1−ps, NE: ps. Marginal OR. Hospital-cluster bootstrap ({NBOOT} replicates) within each imputed data set, pooled with Rubin's rules.",
        "Post hoc, exploratory. No correction for multiplicity.",
        "See the reproduction sheet (all values asserted)."]})


def main():
    warnings.filterwarnings("ignore")
    t_start = time.time()
    log = RunLog()
    say = log.say
    out = out_dir("05")

    # ---- cohorts and reproduction ------------------------------------------------------------
    cohort5 = pd.read_parquet(cohort_path(COHORT_USAGE))
    cohort2 = pd.read_parquet(cohort_path(COHORT_PRIMARY))
    P5 = cohort5[cohort5.primary == 1].copy()
    P2 = cohort2[cohort2.primary == 1].copy()
    REP = reproduction(P2, P5, say)
    P2["grp2"] = P2.grp.map({"VP": "VP", "NE": "NE"})

    # ---- intakeOutput and fluid variables ------------------------------------------------------
    io = load_intake_output(cohort5.patientunitstayid.unique(), say)
    say("intakeOutput rows for cohort stays", len(io), io.cat.value_counts().to_dict())
    PROC = [dict(item="intakeOutput rows (cohort stays)", value=len(io)),
            dict(item="rows with negative offset (before ICU admission; not used)", value=int((io.intakeoutputoffset < 0).sum()))]
    for k, v_ in io.cat.value_counts().items():
        PROC.append(dict(item=f"rows, cellpath category '{k}'", value=int(v_)))
    CONS = consistency_totals(io)
    REC, RINFO = build_records(io)
    for k, v_ in RINFO.items():
        PROC.append(dict(item=k, value=v_))
    FV = fluid_vars(REC, cohort5[["patientunitstayid", "t0", "weight_kg"]])
    # apportioning sensitivity: point attribution at charting time
    RECp, _ = build_records(io, point=True)
    FVp = fluid_vars(RECp, cohort5[["patientunitstayid", "t0", "weight_kg"]])[["patientunitstayid", "fo_pct", "pre24_bal_l", "post8_in_l"]]
    del io, RECp
    P5 = P5.merge(FV.drop(columns="t0"), on="patientunitstayid", how="left")
    P2 = P2.merge(FV.drop(columns="t0"), on="patientunitstayid", how="left")
    for D in (P5, P2):
        D["fo5"] = D.fo_pct / 5
        D["na150_72h"] = np.where((D.na_pre_max < 150) & (D.na_post72_n >= 1), (D.na_post72_max >= 150).astype(float), np.nan)
        D["press24"] = (D.press_h_pre >= 24).astype(int)
    _ap = P5[["patientunitstayid", "fo_pct", "pre24_bal_l", "post8_in_l"]].merge(FVp, on="patientunitstayid", suffixes=("", "_point")).dropna()
    for c in ("fo_pct", "pre24_bal_l", "post8_in_l"):
        PROC.append(dict(item=f"apportioned vs point attribution, {c}: Spearman rho / median |diff| (n)",
                         value=f"{spearmanr(_ap[c], _ap[c + '_point']).correlation:.4f} / {(_ap[c] - _ap[c + '_point']).abs().median():.4f} (n={len(_ap)})"))
    say("fluid variables built in %.0f s" % (time.time() - t_start))

    # ---- analyses --------------------------------------------------------------------------------
    cmp = Comparisons(P2, P5, say)
    COVG, HOSP, COVMOD = coverage(cmp)
    DESC = description(cmp)
    say(COVG.to_string(index=False))
    say(DESC.iloc[:, :8].to_string(index=False))
    fo_adjustment(cmp)
    # FO% tertiles: cut points from NE-only stays with records
    TC = P5.loc[(P5.grp == "NE-only") & (P5.fo_rec == 1), "fo_pct"].quantile([1/3, 2/3]).values

    def tert(x):
        return pd.cut(x, [-np.inf, TC[0], TC[1], np.inf], labels=["T1", "T2", "T3"])

    for D in (P5, P2):
        D["fo_t"] = tert(D.fo_pct)
    say("FO% tertile cut points (NE-only, recorded):", TC)
    tertiles(cmp, TC, tert)
    NE = P5[P5.grp == "NE-only"].copy()
    ne_only_fo(cmp, NE)
    severe_fluid_loaded(cmp, float(NE.loc[NE.fo_rec == 1, "fo_pct"].median()))
    overlap_weighting(cmp)
    restricted_models(cmp)

    # ---- write -----------------------------------------------------------------------------------
    FVout = cohort5.merge(FV, on=["patientunitstayid", "t0"], how="left")
    FVout["in_2grp_primary"] = FVout.patientunitstayid.isin(set(P2.patientunitstayid)).astype(int)
    FVout["grp2_primary"] = FVout.patientunitstayid.map(P2.set_index("patientunitstayid").grp)
    FVout["fo_tertile_NEonly_cut"] = tert(FVout.fo_pct).astype(str).replace("nan", "")
    FVout.to_parquet(cohort_path(COHORT_FLUID), index=False)
    RES = cmp.res
    SHEETS = [("README", readme_sheet(TC)), ("reproduction", pd.DataFrame(REP)), ("coverage", COVG), ("coverage_hospital", HOSP), ("coverage_predictors", COVMOD),
              ("consistency_totals", pd.DataFrame(CONS)), ("processing_log", pd.DataFrame(PROC)), ("a_description", DESC),
              ("b_FO_before_after", RES["b_adjOR"]), ("c_tertiles", RES["c_tertile"]), ("c_interaction", RES["c_interaction"]), ("c_tertiles_MI_sens", RES["c_tertile_MI"]),
              ("d_NEonly_FO", RES["d_NEonly"]), ("e_long_course_high_FO", RES["e_severe"]), ("f_ATO", RES["f_ATO"]), ("f_ATO_SMD", RES["f_ATO_SMD"]),
              ("g1_no_diuretic", RES["g_nodiuretic"]), ("g2_head_to_head", RES["g_h2h"]), ("g3_pressor_ge24h", RES["g_severe24"])]
    with pd.ExcelWriter(os.path.join(out, "eICU_fluid_balance_results.xlsx")) as xw:
        for nm, df in SHEETS:
            df.to_excel(xw, sheet_name=nm[:31], index=False)
    log.write(os.path.join(out, RUN_LOG))
    say("done in %.0f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
