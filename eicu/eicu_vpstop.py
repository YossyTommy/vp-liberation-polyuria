"""Analyses anchored at the vasopressin stop, shared by 06_vpstop_timing.py and
07_switch_closed.py (eICU-CRD v2.0).

Anchors
  * VP groups: VP stop = end of the first VP episode (t0a_vp). NE-only: NE stop (= T0 = T0b).
  * Anchor population: adult, no renal replacement therapy, no pituitary / diabetes insipidus /
    brain-death code, alive and in the ICU at the anchor. NE-only additionally requires that
    no other vasopressor is running at the NE stop (the NE stop is liberation). No urine or
    new-onset requirement.
  * New-onset population: + >= 12 recorded bins in 0-24 h after the anchor + no polyuria in
    -12..-1 h.
Polyuria = > 3 mL/kg/h in >= 2 consecutive recorded hourly bins (an unrecorded bin resets the
run; onset = end of the second bin). Competing events of the cumulative incidence: ICU death,
ICU discharge alive, vasopressor restart after liberation from all vasopressors (time from the
anchor = (T0b - anchor) + restart_h); ties go to the competing event. Hospital = cluster.

Stop-order comparison (NE-first vs VP-first): covariates are re-measured at an arbitrary anchor
with the rules of the cohort construction, and one overlap-weighted estimate is computed from
a propensity score of ten pre-specified variables (multiple imputation, hospital-cluster
bootstrap within each imputed data set, Rubin's rules).
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from eicu_cohort import AIRWAYS, STER_RE, onset_series
from eicu_common import DB, ex_dir
from eicu_extract import INO, PRESS
from eicu_fluid import add_category_and_value, ato_fit, build_records, fluid_vars
from eicu_stats import MICE_M, aj_cif, cr_times, hosp_boot, impute, logit_cluster

ANCHOR_GROUPS = ["NE-first", "VP-first", "VP-first+other", "VP-only", "NE-only", "Simultaneous"]
ORDER_GROUPS = ["NE-first", "VP-first"]
HRS = np.arange(-24, 24)
GRID24 = np.arange(0, 25)
# propensity-score variables of the stop-order comparison (measured at T0a)
PSV = ["age10", "cvicu", "sepsis", "apache10", "log_press_h", "map6", "na_last", "kdigo_ge2", "uo_pre6", "steroid"]
FIX = ["age_n", "male", "cvicu", "card_surg", "sepsis", "apache4", "hospitalid", "grp", "uniquepid"]


def n_pct(k, n):
    """'k/n (%)'."""
    return f"{int(k)}/{int(n)} ({100 * k / n:.1f}%)" if n else f"{int(k)}/0"


# =====================================================================================
# Anchor population and hourly urine relative to the anchor
# =====================================================================================
def in_run_bins(d):
    """Bins of {bin: rate} above 3 mL/kg/h that are adjacent to another recorded bin above 3."""
    hs = sorted(d)
    out = set()
    for h in hs:
        if d[h] > 3 and ((h - 1 in d and d[h - 1] > 3) or (h + 1 in d and d[h + 1] > 3)):
            out.add(h)
    return out


def anchor_population(C, ur):
    """Anchor (anc, anc_name) and anchor-population flag (anc_pop) are added to C.

    Returns a namespace with
      A    anchor-population stays with onset_anc (polyuria onset, h from the anchor), pre12_anc,
           obs24_anc, obs_after_anc_h, restart_anc_h, t0b_from_anc_h, new_onset_pop, urine24_pop
      HRA  hourly urine (stay, bin h relative to the anchor, mL, weight, rate), bins -48..71
      SER  {stay: {bin: rate}}
      RUN  {stay: set of bins that are part of a run above 3 mL/kg/h}"""
    C["anc"] = np.where(C.grp == "NE-only", C.t0, C.t0a_vp)
    C["anc_name"] = np.where(C.grp == "NE-only", "NE_stop", "VP_stop")
    # NE-only: the NE stop must be liberation from all vasopressors (press_t0 == 0); otherwise the
    # records of the continuing vasopressor would count as an immediate restart
    C["anc_pop"] = np.where(C.grp == "NE-only", (C.riskset == 1) & (C.excl_pit == 0) & (C.press_t0 == 0), (C.a_riskset == 1) & (C.excl_pit == 0)).astype(int)
    A = C[C.grp.isin(ANCHOR_GROUPS) & (C.anc_pop == 1)].copy()
    # hourly urine relative to the anchor: floor((offset - anchor) / 60), sum mL / weight_kg
    u = ur.merge(A[["patientunitstayid", "anc", "weight_kg"]], on="patientunitstayid")
    u = u[u.weight_kg > 0]
    u["h"] = np.floor((u.intakeoutputoffset - u.anc) / 60)
    u = u[(u.h >= -48) & (u.h < 72)]
    HRA = u.groupby(["patientunitstayid", "h"]).agg(ml=("v", "sum"), w=("weight_kg", "first")).reset_index()
    HRA["rate"] = HRA.ml / HRA.w
    HRA["h"] = HRA.h.astype(int)
    SER = {sid: dict(zip(g.h, g.rate)) for sid, g in HRA.groupby("patientunitstayid")}
    A["onset_anc"] = A.patientunitstayid.map({s: onset_series(v, 3.0, 2) for s, v in SER.items()})
    A["pre12_anc"] = A.patientunitstayid.map({s: int(onset_series(v, 3.0, 2, -12, 0) is not None) for s, v in SER.items()}).fillna(0).astype(int)
    A["obs24_anc"] = A.patientunitstayid.map(HRA[(HRA.h >= 0) & (HRA.h < 24)].groupby("patientunitstayid").h.nunique()).fillna(0)
    A["obs_after_anc_h"] = (A.unitdischargeoffset - A.anc) / 60
    A["restart_anc_h"] = (A.t0 - A.anc) / 60 + A.restart_h          # restart after liberation from all vasopressors
    A["t0b_from_anc_h"] = (A.t0 - A.anc) / 60
    A["new_onset_pop"] = ((A.obs24_anc >= 12) & (A.pre12_anc == 0)).astype(int)
    A["urine24_pop"] = (A.obs24_anc >= 12).astype(int)
    RUN = {s: in_run_bins(v) for s, v in SER.items()}
    return SimpleNamespace(A=A, HRA=HRA, SER=SER, RUN=RUN)


def hourly_table(X, g, an, anc, hrs=HRS, extra=None):
    """Hourly urine description of the stays X relative to their anchor: stays alive in the ICU
    at the bin start, stays with urine recorded, bins above 3 mL/kg/h, bins that are part of a
    run, and the distribution of the hourly rate. `anc` is the result of anchor_population."""
    ids = set(X.patientunitstayid)
    H = anc.HRA[anc.HRA.patientunitstayid.isin(ids)]
    st = X.anc.values
    dis = X.unitdischargeoffset.values
    rows = []
    for h in hrs:
        b0 = st + 60 * h
        alive = int(((b0 >= 0) & (b0 < dis)).sum())     # in ICU (admission offset 0) at bin start; ICU death = unit discharge
        x = H[H.h == h]
        n_run = sum(1 for s in x.patientunitstayid if h in anc.RUN.get(s, ()))
        r = x.rate
        d = dict(db=DB, group=g, anchor=an, hour=int(h), n_alive_in_icu=alive, n_recorded=int(len(x)), n_uo_gt3=int((r > 3).sum()), n_in_run=int(n_run),
                 uo_median=r.median() if len(r) else np.nan, uo_q1=r.quantile(.25) if len(r) else np.nan,
                 uo_q3=r.quantile(.75) if len(r) else np.nan, uo_mean=r.mean() if len(r) else np.nan)
        if extra:
            d.update(extra)
        rows.append(d)
    return pd.DataFrame(rows)


# =====================================================================================
# Cumulative incidence after the anchor
# =====================================================================================
def competing_risk_frame(X):
    """Columns expected by eicu_stats.cr_times, with times measured from the anchor."""
    return pd.DataFrame({"uo_onset_h": X.onset_anc.values, "obs_after_t0_h": X.obs_after_anc_h.values,
                         "died_icu": X.died_icu.values, "restart_h": X.restart_anc_h.values, "hospitalid": X.hospitalid.values})


def _cif24(D):
    t, s = cr_times(D, horizon=24.0)
    return aj_cif(t, s, None, GRID24)


def cif_block(X, g, an, label, nboot):
    """Aalen-Johansen cumulative incidence 0-24 h after the anchor with a hospital-cluster
    bootstrap percentile interval. Returns the hourly rows and a row of counts."""
    D = competing_risk_frame(X)
    t, s = cr_times(D, horizon=24.0)
    est = aj_cif(t, s, None, GRID24)
    bs = hosp_boot(D, _cif24, B=nboot, seed=2026)
    lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
    rr = []
    for j, h in enumerate(GRID24):
        rr.append(dict(db=DB, group=g, anchor=an, time_h=int(h), n_at_risk=int((t >= h).sum()) if h > 0 else len(t),
                       n_events=int(((s == 1) & (t <= h)).sum()), cif=est[j], lo=lo[j], hi=hi[j]))
    cnt = dict(population=label, group=g, anchor=an, n=len(X), n_hospitals=int(X.hospitalid.nunique()),
               crude_onset_0_8=n_pct(int((X.onset_anc <= 8).sum()), len(X)),
               crude_onset_gt8_24=n_pct(int(((X.onset_anc > 8) & (X.onset_anc <= 24)).sum()), len(X)),
               crude_onset_0_24=n_pct(int((X.onset_anc <= 24).sum()), len(X)),
               AJ_events_0_8=n_pct(int(((s == 1) & (t <= 8)).sum()), len(X)),
               AJ_events_gt8_24=n_pct(int(((s == 1) & (t > 8) & (t <= 24)).sum()), len(X)),
               competing_by24=f"death {int(((s == 2)).sum())}, ICU discharge {int((s == 3).sum())}, restart {int((s == 4).sum())}",
               polyuria_after_competing_event_0_24=int(((X.onset_anc.values <= 24) & (s != 1)).sum()),
               CIF8_pct=f"{100*est[8]:.1f} ({100*lo[8]:.1f}–{100*hi[8]:.1f})", CIF24_pct=f"{100*est[24]:.1f} ({100*lo[24]:.1f}–{100*hi[24]:.1f})",
               n_boot_ok=len(bs))
    return rr, cnt


# =====================================================================================
# Sodium after T0b
# =====================================================================================
def sodium_frame(P, NA):
    """Stays of P with all sodium values < 150 mmol/L in the 24 h before T0b and >= 1 value in
    (0, 72] h, with na150_72h, the time of the first value >= 150 (t_na150) and dNa = maximum
    after T0b - last value before. NA = sodium values with h = hours from T0b. Returns the
    stays and their sodium values in (0, 72] h."""
    S = P[(P.na_pre_max < 150) & (P.na_post72_n >= 1)].copy()
    S["na150_72h"] = (S.na_post72_max >= 150).astype(int)
    x = NA[NA.patientunitstayid.isin(set(S.patientunitstayid)) & (NA.h > 0) & (NA.h <= 72)]
    S["t_na150"] = S.patientunitstayid.map(x[x.v >= 150].groupby("patientunitstayid").h.min())
    S["dNa"] = S.na_post72_max - S.na_pre
    return S, x


# =====================================================================================
# Stop-order comparison: covariates at an arbitrary anchor, overlap-weighted estimate
# =====================================================================================
def order_sources(C, tabs, src):
    """Source tables restricted to the NE-first and VP-first stays, for cov_at.
    tabs = intermediate tables of eicu_cohort.build_usage_cohort, src = eicu_extract.load_source."""
    ids = set(C.loc[C.grp.isin(ORDER_GROUPS), "patientunitstayid"])
    exd = ex_dir()
    pos = tabs.pos
    posr = pos[pos.patientunitstayid.isin(ids)][["patientunitstayid", "infusionoffset", "rate", "unit", "patientweight"] + [f"c_{c}" for c in PRESS + INO]].copy()
    allep = pd.concat([tabs.EP[c].assign(drug=c) for c in PRESS])
    allep = allep[allep.patientunitstayid.isin(ids)]
    lab = tabs.lab[tabs.lab.patientunitstayid.isin(ids)][["patientunitstayid", "labresultoffset", "labname", "v"]]
    parts = []
    for f in ("x_vap.csv", "x_vp.csv"):
        for ch in pd.read_csv(os.path.join(exd, f), chunksize=5_000_000):
            parts.append(ch[ch.patientunitstayid.isin(ids)])
    mp = pd.concat(parts, ignore_index=True)
    del parts
    mp["map"] = pd.to_numeric(mp["map"], errors="coerce")
    mp = mp[(mp["map"] > 10) & (mp["map"] < 200)]
    ur = tabs.ur[tabs.ur.patientunitstayid.isin(ids)]
    trt = src.trt[src.trt.patientunitstayid.isin(ids)]
    rc = src.rc[src.rc.patientunitstayid.isin(ids)]
    med = src.med[src.med.patientunitstayid.isin(ids)]
    io = []
    for ch in pd.read_csv(os.path.join(exd, "x_intakeoutput.csv"), low_memory=False, chunksize=3_000_000):
        io.append(ch[ch.patientunitstayid.isin(ids)])
    io = pd.concat(io, ignore_index=True)
    add_category_and_value(io)
    rec, _ = build_records(io)
    del io
    return SimpleNamespace(posr=posr, allep=allep, EP=tabs.EP, lab=lab, mp=mp, ur=ur, trt=trt, rc=rc, med=med, rec=rec,
                           weight=src.pt.set_index("patientunitstayid").admissionweight,
                           weight_kg=C[["patientunitstayid", "weight_kg"]])


def cov_at(U, anc, S):
    """Covariates of the cohort construction re-measured at an arbitrary anchor (minutes from
    ICU admission) for the stays U. S = order_sources(...)."""
    T = pd.DataFrame({"patientunitstayid": U.patientunitstayid.values, "t0": np.asarray(anc, float)})
    out = T.set_index("patientunitstayid")
    p = S.posr.merge(T, on="patientunitstayid")
    p["dt"] = p.infusionoffset - p.t0
    e = S.allep.merge(T, on="patientunitstayid")
    fps = e[e.start <= e.t0].groupby("patientunitstayid").start.min()
    out["press_h"] = (out.t0 - fps.reindex(out.index)) / 60
    out["log_press_h"] = np.log1p(out.press_h)
    pre = p[p.dt <= 0]
    out["other_press"] = out.index.isin(set(pre.loc[pre[["c_pe", "c_epi", "c_dopa", "c_ang"]].any(axis=1), "patientunitstayid"])).astype(int)
    # inotrope at the anchor: record within [anchor - 60, anchor] or episode straddling the anchor
    ino = set()
    for c in INO:
        ino |= set(p.loc[p[f"c_{c}"] & (p.dt >= -60) & (p.dt <= 0), "patientunitstayid"])
        ee = S.EP[c].merge(T, on="patientunitstayid")
        ino |= set(ee.loc[(ee.start <= ee.t0) & (ee.end > ee.t0), "patientunitstayid"])
    out["inotrope"] = out.index.isin(ino).astype(int)
    # NE / VP rate immediately before the anchor: last positive record within the episode of that
    # drug that covers the anchor (start <= anchor <= end, end inclusive so the drug being stopped
    # at the anchor counts); dose conversion of the cohort construction (NE: mcg/kg/min, 0-3;
    # VP: units/min records with 0 < rate <= 0.2). Drug not running -> rate 0.
    for drug in ("ne", "vp"):
        ee = S.EP[drug].merge(T, on="patientunitstayid")
        ee = ee[(ee.start <= ee.t0) & (ee.end >= ee.t0)]
        ee = ee.sort_values("start").groupby("patientunitstayid").last()[["start"]]
        q = p[p[f"c_{drug}"]].merge(ee.reset_index(), on="patientunitstayid")
        q = q[(q.infusionoffset >= q.start) & (q.infusionoffset <= q.t0)].sort_values("infusionoffset").groupby("patientunitstayid").last()
        u = q.unit.fillna("")
        if drug == "ne":
            w = q.index.map(S.weight).values.astype(float)
            w = np.where(w > 20, w, np.where(q.patientweight > 20, q.patientweight, np.nan))
            d = np.select([u == "mcg/kg/min", u == "mcg/kg/hr", u == "mcg/min", u == "mcg/hr", u == "mg/hr", u == "mg/min"],
                          [q.rate, q.rate / 60, q.rate / w, q.rate / 60 / w, q.rate * 1000 / 60 / w, q.rate * 1000 / w], np.nan)
            d = pd.Series(d, index=q.index)
            d[(d <= 0) | (d > 3)] = np.nan
        else:
            d = pd.Series(np.where((u == "units/min") & (q.rate > 0) & (q.rate <= 0.2), q.rate, np.nan), index=q.index)
        out[f"{drug}_running"] = out.index.isin(set(ee.index)).astype(int)
        out[f"{drug}_rate"] = np.where(out[f"{drug}_running"] == 1, d.reindex(out.index), 0.0)
        out[f"{drug}_rate_unknown"] = ((out[f"{drug}_running"] == 1) & pd.isna(out[f"{drug}_rate"])).astype(int)
    # laboratory values
    L = S.lab.merge(T, on="patientunitstayid")
    L["h"] = (L.labresultoffset - L.t0) / 60
    cre = L[(L.labname == "creatinine") & (L.v > 0) & (L.v < 30)]
    cb = cre[(cre.h > -720) & (cre.h <= 0)].groupby("patientunitstayid").v.min().reindex(out.index)
    cp = cre[(cre.h > -168) & (cre.h <= 0)].groupby("patientunitstayid").v.max().reindex(out.index)
    r = cp / cb
    k = np.select([r >= 3, r >= 2, (r >= 1.5) | (cp - cb >= 0.3)], [3, 2, 1], default=0)
    out["kdigo_cat"] = pd.Series(k.astype(str), index=out.index).where(cp.notna(), "unk")
    na_ = L[(L.labname == "sodium") & (L.v > 100) & (L.v < 200) & (L.h > -24) & (L.h <= 0)]
    out["na_last"] = na_.sort_values("h").groupby("patientunitstayid").v.last().reindex(out.index)
    lac = L[(L.labname == "lactate") & (L.v >= 0) & (L.v < 40) & (L.h > -24) & (L.h <= 0)]
    out["lactate_max24"] = lac.groupby("patientunitstayid").v.max().reindex(out.index)
    M_ = S.mp.merge(T, on="patientunitstayid")
    M_["h"] = (M_.observationoffset - M_.t0) / 60
    out["map6"] = M_[(M_.h >= -6) & (M_.h < 0)].groupby("patientunitstayid")["map"].median().reindex(out.index)
    # urine 6 h before the anchor (mean of recorded hourly rates, bins -6..-1)
    uu = S.ur.merge(T.merge(S.weight_kg, on="patientunitstayid"), on="patientunitstayid")
    uu = uu[uu.weight_kg > 0]
    uu["h"] = np.floor((uu.intakeoutputoffset - uu.t0) / 60)
    hh = uu[(uu.h >= -6) & (uu.h < 0)].groupby(["patientunitstayid", "h"]).agg(ml=("v", "sum"), w=("weight_kg", "first")).reset_index()
    hh["rate"] = hh.ml / hh.w
    out["uo_pre6"] = hh.groupby("patientunitstayid").rate.mean().reindex(out.index)
    # mechanical ventilation / corticosteroid 24 h before the anchor
    tr = S.trt.merge(T, on="patientunitstayid")
    tr["h"] = (tr.treatmentoffset - tr.t0) / 60
    ts = tr.treatmentstring.astype(str)
    mvt = tr[ts.str.contains("mechanical ventilation", case=False) & ~ts.str.contains("non-invasive", case=False) & (tr.h > -24) & (tr.h <= 0)]
    rc2 = S.rc.merge(T, on="patientunitstayid")
    rc2["h"] = (rc2.respcarestatusoffset - rc2.t0) / 60
    mvr = rc2[rc2.airwaytype.isin(AIRWAYS) & (rc2.h > -24) & (rc2.h <= 0)]
    out["mv"] = out.index.isin(set(mvt.patientunitstayid) | set(mvr.patientunitstayid)).astype(int)
    ms = S.med[S.med.drugname.str.contains(STER_RE, case=False) & (S.med.drugordercancelled.astype(str) != "Yes")].merge(T, on="patientunitstayid")
    ms = ms[~ms.routeadmin.astype(str).str.contains(r"topical|ophth|inhal|eye|ear|otic|rectal|intra-artic", case=False)]
    active = (ms.drugstartoffset <= ms.t0) & (ms.drugstartoffset > ms.t0 - 4320) & ((ms.drugstopoffset >= ms.t0 - 1440) | (ms.drugstopoffset <= 0) | ms.drugstopoffset.isna())
    started = (ms.drugstartoffset > ms.t0 - 1440) & (ms.drugstartoffset <= ms.t0)
    tst = tr[ts.str.contains(r"glucocorticoid|steroid", case=False) & (tr.h > -24) & (tr.h <= 0)]
    out["steroid"] = out.index.isin(set(ms[active | started].patientunitstayid) | set(tst.patientunitstayid)).astype(int)
    # cumulative fluid overload from ICU admission to the anchor
    fv = fluid_vars(S.rec, T.merge(S.weight_kg, on="patientunitstayid"))
    out["fo_pct"] = fv.set_index("patientunitstayid").fo_pct.reindex(out.index)
    out = out.reset_index()
    for kk in ["0", "1", "2", "3", "unk"]:
        out[f"kdigo_{kk}"] = (out.kdigo_cat == kk).astype(int)
    out["kdigo_ge2"] = out.kdigo_cat.isin(["2", "3"]).astype(int)
    return out.drop(columns=["t0"])


def order_frame(P, X):
    """Analysis frame of the stop-order comparison: fixed characteristics and the outcome _y of
    P joined with the anchor covariates X; exposure vp = 1 for NE-first, 0 for VP-first."""
    D = P[["patientunitstayid"] + FIX + [c for c in P.columns if c.startswith("_y")]].merge(X, on="patientunitstayid")
    D["age10"] = D.age_n / 10
    D["apache10"] = D.apache4 / 10
    D["vp"] = (D.grp == "NE-first").astype(int)
    return D.reset_index(drop=True)


def ato_est(s, xcols):
    """Overlap-weighted log odds ratio and risk difference of _y, with the weights and the
    propensity scores."""
    lo, w, ps = ato_fit(s, xcols, ["_y"])
    t = s.vp.values == 1
    yy = s._y.values.astype(float)
    return np.array([lo[0], np.average(yy[t], weights=w[t]) - np.average(yy[~t], weights=w[~t])]), w, ps


def order_estimate(DA, nboot):
    """Crude and overlap-weighted estimate of NE-first vs VP-first in the frame DA.

    Returns a namespace with the group sizes and events (n1, k1 NE-first; n0, k0 VP-first), the
    crude odds ratio (cor) and the SE of its logarithm (cse), the pooled log odds ratio and
    risk difference (qb) with their SE (se), the number of imputations (m), the successful
    bootstrap replicates per imputation (nboot) and, per imputation, the weights (w),
    propensity scores (ps), exposure (vp) and non-finite replicate counts (nonfinite)."""
    n1, n0 = int(DA.vp.sum()), int((DA.vp == 0).sum())
    k1, k0 = int(DA.loc[DA.vp == 1, "_y"].sum()), int(DA.loc[DA.vp == 0, "_y"].sum())
    pc, vc = logit_cluster(DA, "_y", ["vp"])
    cor, cse = np.exp(pc["vp"]), np.sqrt(vc.loc["vp", "vp"])
    sets = impute(DA, PSV, ["_y", "vp"], MICE_M)
    Q, Wv, nb, imps = [], [], [], []
    for k, s in enumerate(sets):
        s = s.reset_index(drop=True)
        e_, w, ps = ato_est(s, PSV)
        Q.append(e_)
        bt = hosp_boot(s, lambda b: ato_est(b, PSV)[0], B=nboot, seed=2026 + k)
        fin = np.isfinite(bt)
        nb.append(len(bt))
        Wv.append(np.array([bt[fin[:, j], j].var(ddof=1) for j in range(bt.shape[1])]))
        imps.append(SimpleNamespace(w=w, ps=ps, vp=s.vp, nonfinite=(~fin).sum(axis=0)))
    Q = np.array(Q)
    Wv = np.array(Wv)
    m = len(Q)
    qb = Q.mean(0)
    Bv = Q.var(0, ddof=1) if m > 1 else 0 * qb
    se = np.sqrt(Wv.mean(0) + (1 + 1 / m) * Bv)
    return SimpleNamespace(n1=n1, n0=n0, k1=k1, k0=k0, cor=cor, cse=cse, qb=qb, se=se, m=m, nboot=nb, imps=imps)
