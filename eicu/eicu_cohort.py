"""Cohort construction for eICU-CRD v2.0.

Two stay-level tables are built from the same steps:

  build_primary_cohort()   two groups (VP, NE-only); T0 = end of the first vasopressin episode
                           (VP) or of the first norepinephrine episode (NE-only)
  build_usage_cohort()     usage / stop-order groups; T0 = T0b, liberation from all vasopressors

Shared rules
  - infusionDrug point records with rate > 0 = running; records <= `gap` min apart form one
    episode; episode end = first rate-0 record <= `gap` min after the last positive record,
    else the last positive record; duration = end - start + `dur` (single record = `dur`)
  - "running at T0" = a positive record in [T0 - `dur`, T0] or an episode straddling T0
    (start <= T0 < end); records and episodes of the classes that define T0 are ignored
  - urine = intakeOutput cells with 'Output' in cellpath and a urine label, 0 <= value < 3000 mL,
    weight > 0; hourly bins floor((offset - T0) / 60); unrecorded hours are missing (the run
    counter resets)
  - polyuria = > 3 mL/kg/h in >= 2 consecutive bins; onset = end of the qualifying bin
  - risk set = age >= 18, no renal replacement therapy (treatment table, whole stay), in the
    ICU at T0 and no recorded death before T0; evaluable = risk set + >= 12 urine bins in 0-24 h
  - new onset: stays meeting the polyuria criterion in bins -12..-1 are excluded
  - pituitary disease / diabetes insipidus / brain death excluded (diagnosis ICD codes and
    strings, admissionDx, pastHistory, apacheadmissiondx)
  - NE dose: mcg/kg/min as is; mcg/min / weight; mcg/kg/hr / 60; mcg/hr, mg/hr, mg/min converted
    with weight; ml/hr and blank units = unknown (indicator); > 3 mcg/kg/min = unknown

Every step function adds its columns to the cohort table C in place, in a fixed order.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from eicu_common import ex_dir
from eicu_extract import CLS, INO, PRESS, VP_RE

GAP_MIN = 180.0            # join threshold of infusion records, minutes
POINT_RECORD_MIN = 60.0    # duration credited to a point record, minutes

URINE_RE = r"urine|foley|urethral|indwelling catheter|void|urinary catheter output: indwelling/continuous ure"
# the last alternative matches the truncated label 'Urinary Catheter Output: Indwelling/Continuous Ure'
URINE_EXCL_RE = r"urine count|urine occurrence|number of incontinent voids"   # void counts, not volumes
RRT_RE = r"dialysis|crrt|cvvh|hemofiltration|ultrafiltration"
DIUR_RE = r"furosemide|lasix|bumetanide|bumex|torsemide|demadex|ethacrynic|metolazone"
STER_RE = r"hydrocortisone|solu-cortef|methylprednisolone|solu-medrol|dexamethasone|decadron|prednisone|prednisolone|fludrocortisone"
PIT_RE = (r"insipidus|hypopituitar|pituitary hypothyroid|pituitary adrenal|hyperprolactin|acromegal|"
          r"pituitary (?:tumor|adenoma|apoplexy|mass|neoplasm)|craniopharyng|brain death|brain dead")
PIT_ICD = r"(?:^|, )(?:253\.|588\.1|E22|E23|N25\.1|G93\.82|348\.82)"
CARD_UNITS = ("CCU-CTICU", "CSICU", "Cardiac ICU", "CTICU")
CS_RE = r"CABG|valve replacement|valve repair|Aortic and Mitral valve|valve surgery"
OSM_RE = r"mannitol|osmitrol|sodium chloride 3|sodium chloride 23|hypertonic|3% saline|23\.4"
HICL_VP = 2839.0   # 'vasopressin 60 units in 100 ml ns' (only VP HICL in eICU medication)
AIRWAYS = ["Oral ETT", "Tracheostomy", "Nasal ETT", "Double-Lumen Tube", "Cricothyrotomy"]
PATIENT_COLS = ["patientunitstayid", "patienthealthsystemstayid", "uniquepid", "gender", "age_n", "hospitalid",
                "unittype", "admissionweight", "unitdischargeoffset", "unitdischargestatus",
                "hospitaldischargeoffset", "hospitaldischargestatus", "unitvisitnumber", "apacheadmissiondx"]


# =====================================================================================
# Episodes and index stays
# =====================================================================================
def episodes(df, gap=GAP_MIN, dur=POINT_RECORD_MIN):
    """Infusion episodes of one drug class (one row per stay and episode)."""
    df = df[df.rate.notna()].sort_values(["patientunitstayid", "infusionoffset"])
    on = df[df.rate > 0].copy()
    g = on.groupby("patientunitstayid").infusionoffset.diff()
    on["ep"] = (g.isna() | (g > gap)).astype(int).groupby(on.patientunitstayid).cumsum()
    ep = (on.groupby(["patientunitstayid", "ep"])
            .agg(start=("infusionoffset", "min"), last_on=("infusionoffset", "max"),
                 n_rec=("infusionoffset", "size"), unit=("unit", "first")).reset_index())
    off = df[df.rate <= 0][["patientunitstayid", "infusionoffset"]].rename(columns={"infusionoffset": "off_t"})
    m = ep[["patientunitstayid", "ep", "last_on"]].merge(off, on="patientunitstayid")
    m = m[(m.off_t > m.last_on) & (m.off_t - m.last_on <= gap)]
    ep = ep.merge(m.groupby(["patientunitstayid", "ep"]).off_t.min().rename("stop_charted").reset_index(),
                  on=["patientunitstayid", "ep"], how="left")
    ep["end"] = ep.stop_charted.fillna(ep.last_on)
    ep["dur_h"] = (ep.end - ep.start) / 60 + dur / 60
    return ep


def drug_episodes(inf, gap=GAP_MIN, dur=POINT_RECORD_MIN):
    """Episodes of every drug class: {class: episode table}."""
    return {c: episodes(inf[inf["c_" + c]], gap, dur) for c in CLS}


def first_ep(ep):
    """First episode of each stay, indexed by stay."""
    return ep.sort_values(["patientunitstayid", "start"]).groupby("patientunitstayid").first()


def ne_only_stays(src, EP, say):
    """Stays with a norepinephrine episode and no vasopressin anywhere (infusionDrug name,
    medication name or HICL code, treatment string). Returns the stay set and the first rows
    of the flow table."""
    inf, med, trt = src.inf, src.med, src.trt
    vp_inf_any = set(inf.loc[inf.c_vp, "patientunitstayid"])
    vp_med_name = set(med.loc[med.drugname.str.contains(VP_RE, case=False), "patientunitstayid"])
    vp_med_hicl = set(med.loc[med.drughiclseqno == HICL_VP, "patientunitstayid"])
    vp_trt = set(trt.loc[trt.treatmentstring.astype(str).str.contains("vasopressin", case=False), "patientunitstayid"])
    ne_stays = set(EP["ne"].patientunitstayid)
    ne_cand = ne_stays - vp_inf_any
    ne_excl_other = ne_cand & (vp_med_name | vp_med_hicl | vp_trt)
    neonly = ne_cand - ne_excl_other
    flow = []
    flow.append(dict(step="Stays with a reconstructable VP episode (infusionDrug)", group="VP", n_stays=EP["vp"].patientunitstayid.nunique()))
    flow.append(dict(step="Stays with a reconstructable NE episode and no VP record in infusionDrug", group="NE-only", n_stays=len(ne_cand)))
    flow.append(dict(step="  excluded: VP in medication (name or HICL 2839) or treatment string", group="NE-only", n_stays=len(ne_excl_other)))
    say("NE episode stays", len(ne_stays), "; no VP in infusionDrug", len(ne_cand),
        "; excluded for VP in medication/treatment", len(ne_excl_other),
        "(med name", len(ne_cand & vp_med_name), ", HICL", len(ne_cand & vp_med_hicl), ", treatment", len(ne_cand & vp_trt), ")")
    return neonly, flow


def two_group_index(EP, neonly):
    """One row per stay: VP stays (first vasopressin episode) and NE-only stays (first
    norepinephrine episode); T0 = end of that episode."""
    fv = first_ep(EP["vp"])
    fn = first_ep(EP["ne"]).loc[sorted(neonly)]
    C = pd.concat([
        pd.DataFrame({"patientunitstayid": fv.index, "grp": "VP", "t0": fv.end.values, "idx_start": fv.start.values,
                      "idx_dur_h": fv.dur_h.values, "idx_nrec": fv.n_rec.values, "idx_stop_charted": fv.stop_charted.notna().values}),
        pd.DataFrame({"patientunitstayid": fn.index, "grp": "NE", "t0": fn.end.values, "idx_start": fn.start.values,
                      "idx_dur_h": fn.dur_h.values, "idx_nrec": fn.n_rec.values, "idx_stop_charted": fn.stop_charted.notna().values})], ignore_index=True)
    C["vp"] = (C.grp == "VP").astype(int)
    return C


def liberation_groups(EP):
    """Liberation from all vasopressors (T0b) and usage / stop-order group of every stay with a
    vasopressin episode.

    T0a(VP) = end of the first VP episode. T0b = first time >= T0a(VP) at which no vasopressor
    episode is running ("running at t" = start <= t < end; first VP episode excluded); t is
    advanced to the latest end of the episodes running at t until nothing is running."""
    fv = first_ep(EP["vp"])
    A = pd.concat([EP[c].assign(drug=c) for c in PRESS if len(EP[c])], ignore_index=True)
    A = A[A.patientunitstayid.isin(set(fv.index))]
    rows = []
    for s, g in A.groupby("patientunitstayid"):
        f1 = fv.loc[s]
        t0a = f1.end
        o = g[~((g.drug == "vp") & (g.start == f1.start))]
        st, en, dr = o.start.values, o.end.values, o.drug.values
        t = t0a
        chain = set()
        nit = 0
        while True:
            m = (st <= t) & (en > t)
            if not m.any():
                break
            chain |= set(dr[m])
            t = en[m].max()
            nit += 1
        # "used before T0b" = episode start <= T0b (single-record episodes have start == end; an
        # episode starting exactly at T0b is counted as used/stopped at T0b)
        vpr = bool(((dr == "vp") & (st > t0a) & (st <= t)).any())
        nem = (dr == "ne") & (st <= t)
        lne = en[nem].max() if nem.any() else np.nan
        oth = bool((np.isin(dr, ["pe", "epi", "dopa", "ang"]) & (st <= t) & (en > t0a)).any())
        rows.append((s, t0a, t, nit, vpr, lne, oth, ",".join(sorted(chain))))
    VG = pd.DataFrame(rows, columns=["patientunitstayid", "t0a_vp", "t0b", "chain_steps", "vp_restart",
                                     "last_ne_end", "other_after_vp", "chain_drugs"])
    d = VG.last_ne_end - VG.t0a_vp
    VG["ne_minus_vp_stop_min"] = d
    VG["grp"] = np.select([VG.vp_restart, VG.last_ne_end.isna(), d < -60, d.abs() <= 60, (d > 60) & ~VG.other_after_vp, d > 60],
                          ["VP restarted", "VP-only", "NE-first", "Simultaneous", "VP-first", "VP-first+other"], "?")
    assert (VG.grp != "?").all()
    VG["t0a"] = np.where(VG.last_ne_end.notna(), np.fmin(VG.last_ne_end, VG.t0a_vp), VG.t0a_vp)
    return VG


def usage_group_index(EP, neonly, VG):
    """One row per stay: VP-containing stays with their usage group and T0 = T0b, and NE-only
    stays (T0 = end of the first norepinephrine episode). ign_<class> marks the vasopressor
    classes whose own records and episodes are ignored in the "running at T0b" rule."""
    fv = first_ep(EP["vp"])
    fn = first_ep(EP["ne"]).loc[sorted(neonly)]
    fvv = fv.loc[VG.patientunitstayid.values]
    C = pd.concat([
        pd.DataFrame({"patientunitstayid": fvv.index, "grp": VG.grp.values, "t0": VG.t0b.values, "idx_start": fvv.start.values,
                      "idx_dur_h": fvv.dur_h.values, "idx_nrec": fvv.n_rec.values, "idx_stop_charted": fvv.stop_charted.notna().values,
                      "t0a_vp": VG.t0a_vp.values, "t0a": VG.t0a.values, "last_ne_end": VG.last_ne_end.values,
                      "ne_minus_vp_stop_min": VG.ne_minus_vp_stop_min.values, "chain_steps": VG.chain_steps.values,
                      "chain_drugs": VG.chain_drugs.values, "other_after_vp": VG.other_after_vp.values}),
        pd.DataFrame({"patientunitstayid": fn.index, "grp": "NE-only", "t0": fn.end.values, "idx_start": fn.start.values,
                      "idx_dur_h": fn.dur_h.values, "idx_nrec": fn.n_rec.values, "idx_stop_charted": fn.stop_charted.notna().values,
                      "t0a_vp": np.nan, "t0a": fn.end.values, "last_ne_end": fn.end.values, "ne_minus_vp_stop_min": np.nan,
                      "chain_steps": 0, "chain_drugs": "", "other_after_vp": False})], ignore_index=True)
    C["vp"] = (C.grp != "NE-only").astype(int)
    C["same_stop_minute"] = (C.ne_minus_vp_stop_min == 0).astype(int)
    C["gap_t0a_t0b_h"] = (C.t0 - C.t0a) / 60
    C["gap_vpstop_t0b_h"] = (C.t0 - C.t0a_vp) / 60
    for c in PRESS:
        C[f"ign_{c}"] = np.where(C.grp == "NE-only", int(c == "ne"),
                                 (c in ("vp", "ne")) | C.chain_drugs.str.contains(c)).astype(int)
    return C


def add_patient_columns(C, pt):
    """Merge the patient table (age as a number; '> 89' = 90)."""
    pt["age_n"] = pd.to_numeric(pt.age.astype(str).str.replace("> 89", "90", regex=False), errors="coerce")
    return C.merge(pt[PATIENT_COLS], on="patientunitstayid", how="left")


# =====================================================================================
# Vasopressors / inotropes at T0, restart, burden, NE dose
# =====================================================================================
def positive_records(inf, C):
    """Positive vasopressor / inotrope records with the stay's group, T0 and dt = offset - T0."""
    pos = inf[(inf.rate > 0) & inf[[f"c_{c}" for c in PRESS + INO]].any(axis=1)][
        ["patientunitstayid", "infusionoffset", "rate", "unit", "patientweight"] + [f"c_{c}" for c in PRESS + INO]]
    pos = pos.merge(C[["patientunitstayid", "grp", "t0"]], on="patientunitstayid")
    pos["dt"] = pos.infusionoffset - pos.t0
    return pos


def running_at_t0(C, pos, EP, ignore, dur=POINT_RECORD_MIN):
    """<class>_t0_rec (positive record in [T0 - dur, T0]) and <class>_t0 (record or episode
    straddling T0) for every vasopressor and inotrope class.

    ignore: table indexed by stay with one boolean column per vasopressor class; True = the
    records and episodes of that class belong to the index sequence and are not counted."""
    T0 = C[["patientunitstayid", "grp", "t0"]]
    for c in PRESS + INO:
        if c in PRESS:
            own = pos.patientunitstayid.map(ignore[c]).astype(bool)
        else:
            own = pd.Series(False, index=pos.index)
        near = pos[pos[f"c_{c}"] & (pos.dt >= -dur) & (pos.dt <= 0) & ~own]
        e = EP[c].merge(T0, on="patientunitstayid")
        if c in PRESS:
            e = e[~e.patientunitstayid.map(ignore[c]).astype(bool)]
        strad = e[(e.start <= e.t0) & (e.end > e.t0)]
        C[f"{c}_t0_rec"] = C.patientunitstayid.isin(set(near.patientunitstayid)).astype(int)
        C[f"{c}_t0"] = (C[f"{c}_t0_rec"] | C.patientunitstayid.isin(set(strad.patientunitstayid))).astype(int)


def pressor_summary(C, pos, EP):
    """Any vasopressor / inotrope at T0, restart after T0, vasopressor time before T0 and the
    norepinephrine dose before T0. Returns the records at or before T0 and the table of
    norepinephrine dose units."""
    T0b = C[["patientunitstayid", "t0"]]
    C["press_t0"] = C[[f"{c}_t0" for c in PRESS]].max(axis=1)
    C["press_t0_rec"] = C[[f"{c}_t0_rec" for c in PRESS]].max(axis=1)
    C["inotrope_t0"] = C[["dob_t0", "mil_t0"]].max(axis=1)
    pp = pos[pos[[f"c_{c}" for c in PRESS]].any(axis=1)]
    C["restart_h"] = C.patientunitstayid.map(pp[pp.dt > 0].groupby("patientunitstayid").dt.min() / 60)
    allep = pd.concat([EP[c] for c in PRESS]).merge(T0b, on="patientunitstayid")
    C["first_press_start"] = C.patientunitstayid.map(allep[allep.start <= allep.t0].groupby("patientunitstayid").start.min())
    C["press_h_pre"] = (C.t0 - C.first_press_start) / 60
    C["log_press_h"] = np.log1p(C.press_h_pre)
    pre = pos[pos.dt <= 0]
    C["other_press_pre"] = C.patientunitstayid.isin(set(pre.loc[pre[["c_pe", "c_epi", "c_dopa", "c_ang"]].any(axis=1), "patientunitstayid"])).astype(int)
    nr = pre[pre.c_ne].copy()
    w = nr.patientunitstayid.map(C.set_index("patientunitstayid").admissionweight)
    w = w.where(w > 20, nr.patientweight.where(nr.patientweight > 20))
    u = nr.unit.fillna("")
    nr["dose"] = np.select([u == "mcg/kg/min", u == "mcg/kg/hr", u == "mcg/min", u == "mcg/hr", u == "mg/hr", u == "mg/min"],
                           [nr.rate, nr.rate / 60, nr.rate / w, nr.rate / 60 / w, nr.rate * 1000 / 60 / w, nr.rate * 1000 / w], np.nan)
    nr.loc[(nr.dose <= 0) | (nr.dose > 3), "dose"] = np.nan
    ne_units = (nr.assign(grp=nr.grp, unit=u.replace("", "(blank)"), convertible=nr.dose.notna())
                  .groupby(["grp", "unit"]).agg(records=("rate", "size"), stays=("patientunitstayid", "nunique"),
                                                 convertible_records=("convertible", "sum")).reset_index()
                  .sort_values(["grp", "records"], ascending=[True, False]))
    C["ne_any_pre"] = C.patientunitstayid.isin(set(nr.patientunitstayid)).astype(int)
    C["ne_peak_pre"] = C.patientunitstayid.map(nr.dropna(subset=["dose"]).groupby("patientunitstayid").dose.max())
    C["ne_dose_unknown"] = ((C.ne_any_pre == 1) & C.ne_peak_pre.isna()).astype(int)
    C["ne_peak01"] = C.ne_peak_pre.fillna(0) / 0.1
    return pre, ne_units


# =====================================================================================
# Risk set and urine
# =====================================================================================
def risk_set(C, trt):
    """Adult, no renal replacement therapy, in the ICU and alive at T0."""
    C["rrt"] = C.patientunitstayid.isin(set(trt.loc[trt.treatmentstring.astype(str).str.contains(RRT_RE, case=False), "patientunitstayid"])).astype(int)
    C["adult"] = (C.age_n >= 18).astype(int)
    C["obs_after_t0_h"] = (C.unitdischargeoffset - C.t0) / 60
    C["died_icu"] = (C.unitdischargestatus.astype(str) == "Expired").astype(int)
    C["hosp_death"] = (C.hospitaldischargestatus.astype(str) == "Expired").astype(int)
    C["excl_death_pre"] = (((C.hosp_death == 1) & (C.hospitaldischargeoffset < C.t0)) |
                           ((C.died_icu == 1) & (C.unitdischargeoffset < C.t0))).astype(int)
    C["excl_disch_pre"] = ((C.excl_death_pre == 0) & (C.unitdischargeoffset <= C.t0)).astype(int)
    C["riskset"] = ((C.adult == 1) & (C.rrt == 0) & (C.excl_death_pre == 0) & (C.excl_disch_pre == 0)).astype(int)


def urine_and_intake_records():
    """Urine output records (0 <= v < 3000 mL; a positive volume charted at the same time under
    two urine labels is kept once) and intake records (0 < v < 5000 mL) of the filtered
    intakeOutput table."""
    io = pd.read_csv(os.path.join(ex_dir(), "x_intakeoutput.csv"), low_memory=False,
                     usecols=["patientunitstayid", "intakeoutputoffset", "cellpath", "celllabel", "cellvaluenumeric"])
    cp = io.cellpath.astype(str)
    ur = io[cp.str.contains("Output", case=False, na=False) & io.celllabel.astype(str).str.contains(URINE_RE, case=False)
            & ~io.celllabel.astype(str).str.contains(URINE_EXCL_RE, case=False)].copy()
    ur["v"] = pd.to_numeric(ur.cellvaluenumeric, errors="coerce")
    ur = ur.dropna(subset=["v"])
    ur = ur[(ur.v >= 0) & (ur.v < 3000)]
    z = ur.v == 0
    ur = pd.concat([ur[z], ur[~z].drop_duplicates(["patientunitstayid", "intakeoutputoffset", "v"])])
    intake = io[cp.str.contains(r"\|Intake", case=False, na=False)].copy()
    intake["v"] = pd.to_numeric(intake.cellvaluenumeric, errors="coerce")
    intake = intake.dropna(subset=["v"])
    intake = intake[(intake.v > 0) & (intake.v < 5000)]
    return ur, intake


def onset_series(hrs, th, mh, lo=0, hi=72):
    """End (hour) of the first run of `mh` consecutive recorded bins above `th` mL/kg/h within
    bins lo..hi-1 of {bin: rate}; None when there is no such run."""
    run = 0
    for hh in range(lo, hi):
        if hh not in hrs:
            run = 0
            continue
        if hrs[hh] > th:
            run += 1
            if run >= mh:
                return hh + 1
        else:
            run = 0
    return None


def urine_outcomes(C, inf, ur):
    """Weight, hourly urine bins relative to T0 and the polyuria outcomes. Returns the hourly
    table (stay, bin h, mL, weight, rate in mL/kg/h; bins -24..71)."""
    # weight: admissionweight if 30-300 kg, else median infusionDrug patientweight (30-300 kg), else missing
    pw = inf[inf.patientweight.between(30, 300)].groupby("patientunitstayid").patientweight.median()
    C["weight_kg"] = C.admissionweight.where(C.admissionweight.between(30, 300), C.patientunitstayid.map(pw))
    u2 = ur.merge(C[["patientunitstayid", "t0", "weight_kg"]], on="patientunitstayid").rename(columns={"weight_kg": "admissionweight"})
    u2 = u2[u2.admissionweight > 0]
    u2["h"] = np.floor((u2.intakeoutputoffset - u2.t0) / 60)
    # charting interval: minutes between consecutive urine records, -24..72 h
    u3 = u2[(u2.h >= -24) & (u2.h < 72)].sort_values(["patientunitstayid", "intakeoutputoffset"])
    u3["gap"] = u3.groupby("patientunitstayid").intakeoutputoffset.diff()
    chart = u3.dropna(subset=["gap"]).groupby("patientunitstayid").gap.median().rename("urine_gap_med_min")
    hr = u3.groupby(["patientunitstayid", "h"]).agg(ml=("v", "sum"), w=("admissionweight", "first")).reset_index()
    hr["rate"] = hr.ml / hr.w
    series = {sid: dict(zip(g.h.astype(int), g.rate)) for sid, g in hr.groupby("patientunitstayid")}

    def pairs(lo, hi):
        x = hr[(hr.h >= lo) & (hr.h < hi)].sort_values(["patientunitstayid", "h"])
        return (x.groupby("patientunitstayid").h.diff() == 1).groupby(x.patientunitstayid).sum()

    C["pre_pairs"] = C.patientunitstayid.map(pairs(-24, 0)).fillna(0)   # charting density before T0
    C["post_pairs"] = C.patientunitstayid.map(pairs(0, 24)).fillna(0)
    C["uo_onset_h"] = C.patientunitstayid.map({s: onset_series(v, 3.0, 2) for s, v in series.items()})
    C["uo_onset5_h"] = C.patientunitstayid.map({s: onset_series(v, 5.0, 2) for s, v in series.items()})
    C["poly_pre12"] = C.patientunitstayid.map({s: int(onset_series(v, 3.0, 2, -12, 0) is not None) for s, v in series.items()}).fillna(0).astype(int)
    post = hr[(hr.h >= 0) & (hr.h < 24)]
    C["obs_h_24"] = C.patientunitstayid.map(post.groupby("patientunitstayid").h.nunique()).fillna(0)
    C["uo_24h_ml"] = C.patientunitstayid.map(post.groupby("patientunitstayid").ml.sum())
    C["has_urine_any"] = C.patientunitstayid.isin(set(ur.patientunitstayid)).astype(int)
    C["urine_gap_med_min"] = C.patientunitstayid.map(chart)
    C["polyuria_24h"] = (C.uo_onset_h <= 24).astype(int)
    C["polyuria_72h"] = C.uo_onset_h.notna().astype(int)
    C["poly8"] = (C.uo_onset_h <= 8).astype(int)
    C["poly_8_24"] = ((C.uo_onset_h > 8) & (C.uo_onset_h <= 24)).astype(int)
    C["marked_24h"] = (C.uo_onset5_h <= 24).astype(int)
    C["uo24_def"] = ((C.uo_24h_ml > 3000) | (C.uo_24h_ml / C.weight_kg > 50)).astype(int)
    C["urine_evaluable"] = (C.obs_h_24 >= 12).astype(int)
    C["evaluable"] = ((C.riskset == 1) & (C.urine_evaluable == 1)).astype(int)
    return hr


# =====================================================================================
# Covariates
# =====================================================================================
def covariates(C, src, intake):
    """Laboratory values, fluid intake, diuretics, blood pressure, severity scores, ventilation,
    corticosteroids, diagnoses and post-T0 co-interventions relative to T0.

    Returns a namespace with lab (numeric laboratory results with h = hours from T0),
    na (plausible sodium values), ev (loop-diuretic events) and osm (osmotic agents)."""
    inf, dn, med, trt, rc, dx, adx, ph, apr, apv = (src.inf, src.dn, src.med, src.trt, src.rc, src.dx, src.adx,
                                                    src.ph, src.apr, src.apv)
    T0b = C[["patientunitstayid", "t0"]]
    lab = pd.read_csv(os.path.join(ex_dir(), "x_lab.csv"), low_memory=False,
                      usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    lab["labname"] = lab.labname.astype(str).str.lower().str.strip()
    lab["v"] = pd.to_numeric(lab.labresult, errors="coerce")
    lab = lab.dropna(subset=["v"]).merge(T0b, on="patientunitstayid")
    lab["h"] = (lab.labresultoffset - lab.t0) / 60
    cre = lab[(lab.labname == "creatinine") & (lab.v > 0) & (lab.v < 30)]
    C["cre_base_pre"] = C.patientunitstayid.map(cre[(cre.h > -720) & (cre.h <= 0)].groupby("patientunitstayid").v.min())
    C["cre_peak_pre"] = C.patientunitstayid.map(cre[(cre.h > -168) & (cre.h <= 0)].groupby("patientunitstayid").v.max())
    r = C.cre_peak_pre / C.cre_base_pre
    C["kdigo"] = np.select([r >= 3, r >= 2, (r >= 1.5) | (C.cre_peak_pre - C.cre_base_pre >= 0.3)], [3, 2, 1], default=0)
    C["kdigo_missing_as_0"] = C.kdigo.where(C.cre_peak_pre.notna(), 0)
    C["kdigo_cat"] = C.kdigo.astype(str).where(C.cre_peak_pre.notna(), "unk")
    for k in ["1", "2", "3", "unk"]:
        C[f"kdigo_{k}"] = (C.kdigo_cat == k).astype(int)
    na = lab[(lab.labname == "sodium") & (lab.v > 100) & (lab.v < 200)]
    na_pre = na[(na.h > -24) & (na.h <= 0)]
    C["na_pre"] = C.patientunitstayid.map(na_pre.sort_values("h").groupby("patientunitstayid").v.last())
    C["na_pre_max"] = C.patientunitstayid.map(na_pre.groupby("patientunitstayid").v.max())
    na_post = na[(na.h > 0) & (na.h <= 72)]
    C["na_post72_max"] = C.patientunitstayid.map(na_post.groupby("patientunitstayid").v.max())
    C["na_post72_n"] = C.patientunitstayid.map(na_post.groupby("patientunitstayid").size()).fillna(0)
    lac = lab[(lab.labname == "lactate") & (lab.v >= 0) & (lab.v < 40)]
    C["lactate_pre24"] = C.patientunitstayid.map(lac[(lac.h > -24) & (lac.h <= 0)].groupby("patientunitstayid").v.max())
    glu = lab[(lab.labname == "glucose") & (lab.v > 0) & (lab.v < 2000)]
    C["glucose_max24"] = C.patientunitstayid.map(glu[(glu.h > 0) & (glu.h <= 24)].groupby("patientunitstayid").v.max())
    it = intake.merge(T0b, on="patientunitstayid")
    it["h"] = (it.intakeoutputoffset - it.t0) / 60
    C["fluid_pre24_l"] = C.patientunitstayid.map(it[(it.h > -24) & (it.h <= 0)].groupby("patientunitstayid").v.sum()) / 1000
    md = med[med.drugname.str.contains(DIUR_RE, case=False)][["patientunitstayid", "drugstartoffset"]].rename(columns={"drugstartoffset": "off"})
    idr = inf[dn.str.contains(DIUR_RE, case=False)][["patientunitstayid", "infusionoffset"]].rename(columns={"infusionoffset": "off"})
    ev = pd.concat([md, idr]).merge(T0b, on="patientunitstayid")
    ev["h"] = (ev.off - ev.t0) / 60
    C["diur_pre24"] = C.patientunitstayid.isin(set(ev.loc[(ev.h > -24) & (ev.h <= 0), "patientunitstayid"])).astype(int)
    C["diur_post24"] = C.patientunitstayid.isin(set(ev.loc[(ev.h > 0) & (ev.h <= 24), "patientunitstayid"])).astype(int)
    mp = pd.concat([pd.read_csv(os.path.join(ex_dir(), "x_vap.csv")), pd.read_csv(os.path.join(ex_dir(), "x_vp.csv"))], ignore_index=True)
    mp["map"] = pd.to_numeric(mp["map"], errors="coerce")
    mp = mp[(mp["map"] > 10) & (mp["map"] < 200)]
    mp = mp.merge(T0b, on="patientunitstayid")
    mp["h"] = (mp.observationoffset - mp.t0) / 60
    C["map_base"] = C.patientunitstayid.map(mp[(mp.h >= -6) & (mp.h < 0)].groupby("patientunitstayid")["map"].median())
    del mp
    ap = apr[apr.apacheversion == "IV"].drop_duplicates("patientunitstayid").set_index("patientunitstayid").apachescore
    C["apache4"] = C.patientunitstayid.map(ap)
    C.loc[C.apache4 < 0, "apache4"] = np.nan
    C["apache10"] = C.apache4 / 10
    aps = apr[apr.apacheversion == "IV"].drop_duplicates("patientunitstayid").set_index("patientunitstayid").acutephysiologyscore
    C["aps10"] = C.patientunitstayid.map(aps).where(lambda s: s >= 0) / 10
    pv = apv.drop_duplicates("patientunitstayid").set_index("patientunitstayid")
    C["elective"] = C.patientunitstayid.map(pv.electivesurgery).fillna(0).clip(0, 1).astype(int)
    tr = trt.merge(T0b, on="patientunitstayid")
    tr["h"] = (tr.treatmentoffset - tr.t0) / 60
    ts = tr.treatmentstring.astype(str)
    mvt = tr[ts.str.contains("mechanical ventilation", case=False) & ~ts.str.contains("non-invasive", case=False) & (tr.h > -24) & (tr.h <= 0)]
    rc2 = rc.merge(T0b, on="patientunitstayid")
    rc2["h"] = (rc2.respcarestatusoffset - rc2.t0) / 60
    mvr = rc2[rc2.airwaytype.isin(AIRWAYS) & (rc2.h > -24) & (rc2.h <= 0)]
    C["mv_t0"] = C.patientunitstayid.isin(set(mvt.patientunitstayid) | set(mvr.patientunitstayid)).astype(int)
    ms = med[med.drugname.str.contains(STER_RE, case=False) & (med.drugordercancelled.astype(str) != "Yes")].merge(T0b, on="patientunitstayid")
    ms = ms[~ms.routeadmin.astype(str).str.contains(r"topical|ophth|inhal|eye|ear|otic|rectal|intra-artic", case=False)]
    active = (ms.drugstartoffset <= ms.t0) & (ms.drugstartoffset > ms.t0 - 4320) & ((ms.drugstopoffset >= ms.t0 - 1440) | (ms.drugstopoffset <= 0) | ms.drugstopoffset.isna())
    started = (ms.drugstartoffset > ms.t0 - 1440) & (ms.drugstartoffset <= ms.t0)
    tst = tr[ts.str.contains(r"glucocorticoid|steroid", case=False) & (tr.h > -24) & (tr.h <= 0)]
    C["steroid_pre24"] = C.patientunitstayid.isin(set(ms[active | started].patientunitstayid) | set(tst.patientunitstayid)).astype(int)
    dxs = dx.diagnosisstring.astype(str)
    ic = dx.icd9code.astype(str)
    C["sepsis"] = (C.apacheadmissiondx.astype(str).str.contains("Sepsis") |
                   C.patientunitstayid.isin(set(dx.loc[dxs.str.contains(r"sepsis|septic", case=False), "patientunitstayid"]))).astype(int)
    C["card_surg"] = C.apacheadmissiondx.astype(str).str.contains(CS_RE).astype(int)
    C["cvicu"] = C.unittype.isin(CARD_UNITS).astype(int)
    pit_set = (set(dx.loc[ic.str.contains(PIT_ICD, regex=True) | dxs.str.contains(PIT_RE, case=False, regex=True), "patientunitstayid"]) |
               set(adx.loc[(adx.admitdxname.astype(str) + "|" + adx.admitdxpath.astype(str)).str.contains(PIT_RE + r"|pituitar|transsphenoid", case=False, regex=True), "patientunitstayid"]) |
               set(ph.loc[ph.pasthistorypath.astype(str).str.contains(r"insipidus|pituitar", case=False), "patientunitstayid"]))
    C["excl_pit"] = (C.patientunitstayid.isin(pit_set) | C.apacheadmissiondx.astype(str).str.contains(r"pituitar|transsphenoid|Brain death", case=False)).astype(int)
    C["neuro_any"] = (C.patientunitstayid.isin(set(dx.loc[dxs.str.contains(r"^neurologic|^burns/trauma\|trauma - CNS", case=False, regex=True), "patientunitstayid"])) |
                      C.unittype.eq("Neuro ICU")).astype(int)
    # post-T0 events and co-interventions
    ddv = tr[ts.str.contains(r"desmopressin|diabetes insipidus", case=False) & ~ts.str.contains("DDAVP injection", case=False) & (tr.h > 0) & (tr.h <= 48)]
    C["ddavp_antidiuretic_48h"] = C.patientunitstayid.isin(set(ddv.patientunitstayid)).astype(int)
    osm_m = med[med.drugname.str.contains(OSM_RE, case=False)].merge(T0b, on="patientunitstayid")
    osm_m["h"] = (osm_m.drugstartoffset - osm_m.t0) / 60
    osm_i = inf[dn.str.contains(OSM_RE, case=False)][["patientunitstayid", "infusionoffset"]].merge(T0b, on="patientunitstayid")
    osm_i["h"] = (osm_i.infusionoffset - osm_i.t0) / 60
    osm_t = tr[ts.str.contains(r"mannitol|hypertonic saline|osmotic", case=False)]
    osm = pd.concat([osm_m[["patientunitstayid", "h"]], osm_i[["patientunitstayid", "h"]], osm_t[["patientunitstayid", "h"]]])
    C["osmo_post24"] = C.patientunitstayid.isin(set(osm.loc[(osm.h > 0) & (osm.h <= 24), "patientunitstayid"])).astype(int)
    C["male"] = (C.gender.astype(str) == "Male").astype(int)
    C["age10"] = C.age_n / 10
    return SimpleNamespace(lab=lab, na=na, ev=ev, osm=osm)


def populations(C):
    """Analysis populations: base_eval (evaluable, no pituitary code, new onset), primary
    (+ no vasopressor running at T0) and rs_primary (the same without the urine requirement)."""
    C["base_eval"] = ((C.evaluable == 1) & (C.excl_pit == 0) & (C.poly_pre12 == 0)).astype(int)
    C["primary"] = ((C.base_eval == 1) & (C.press_t0 == 0)).astype(int)
    C["rs_primary"] = ((C.riskset == 1) & (C.excl_pit == 0) & (C.poly_pre12 == 0) & (C.press_t0 == 0)).astype(int)


# =====================================================================================
# The two cohort tables
# =====================================================================================
def build_primary_cohort(src, say, gap=GAP_MIN, dur=POINT_RECORD_MIN):
    """Two-group cohort (VP vs NE-only), T0 = end of the index episode.

    Only the index drug's own episode is ignored in the "running at T0" rule. Returns the
    cohort table and a namespace with flow (first rows of the flow table), ne_units,
    ev (loop-diuretic events) and osm (osmotic agents), both with h = hours from T0."""
    EP = drug_episodes(src.inf, gap, dur)
    say("episodes (episodes, stays):", {c: (len(v), v.patientunitstayid.nunique()) for c, v in EP.items()})
    neonly, flow = ne_only_stays(src, EP, say)
    C = two_group_index(EP, neonly)
    C = add_patient_columns(C, src.pt)
    pos = positive_records(src.inf, C)
    ignore = pd.DataFrame({c: (C.grp.values == "VP") if c == "vp" else ((C.grp.values == "NE") if c == "ne" else False)
                           for c in PRESS}, index=C.patientunitstayid.values)
    running_at_t0(C, pos, EP, ignore, dur)
    _, ne_units = pressor_summary(C, pos, EP)
    risk_set(C, src.trt)
    ur, intake = urine_and_intake_records()
    urine_outcomes(C, src.inf, ur)
    cov = covariates(C, src, intake)
    populations(C)
    C["grp_label"] = C.grp.map({"VP": "VP", "NE": "NE-only"})
    return C, SimpleNamespace(flow=flow, ne_units=ne_units, ev=cov.ev, osm=cov.osm)


def build_usage_cohort(src, say):
    """Usage / stop-order cohort anchored at liberation from all vasopressors (T0b).

    Groups of VP-containing stays (d = end of the last NE episode before T0b - T0a(VP)):
    VP restarted (another VP episode starting in (T0a(VP), T0b]), VP-only (no NE episode
    before T0b), NE-first (d < -60 min), Simultaneous (|d| <= 60 min), VP-first (d > 60 min),
    VP-first+other (VP-first with another vasopressor running after the VP stop).
    T0a = min(NE stop, VP stop) in NE+VP stays, T0a(VP) otherwise.

    "No vasopressor running at T0b": no vasopressor episode straddling T0b and no positive
    record within 60 min before T0b of a class outside the liberation sequence (NE-only: NE;
    VP stays: VP, NE and every class running at some step of the chain).

    Returns the cohort table and a namespace with the intermediate tables that later steps
    re-use: flow, EP (episodes per class), pos (positive vasopressor / inotrope records),
    ur (urine records), lab and na (laboratory results with h = hours from T0b)."""
    EP = drug_episodes(src.inf)
    say("episodes (episodes, stays):", {c: (len(v), v.patientunitstayid.nunique()) for c, v in EP.items()})
    neonly, flow = ne_only_stays(src, EP, say)
    VG = liberation_groups(EP)
    say("VP-containing stays by group (before exclusions):", VG.grp.value_counts().to_dict())
    C = usage_group_index(EP, neonly, VG)
    C = add_patient_columns(C, src.pt)
    pos = positive_records(src.inf, C)
    ignore = C.set_index("patientunitstayid")[[f"ign_{c}" for c in PRESS]]
    ignore.columns = PRESS
    running_at_t0(C, pos, EP, ignore)
    # NE positive record within 60 min before the VP stop (the two-group analysis counts this
    # as "NE running at T0")
    pa = pos[pos.c_ne].merge(C[["patientunitstayid", "t0a_vp"]].dropna(), on="patientunitstayid")
    pa = pa[(pa.infusionoffset - pa.t0a_vp >= -60) & (pa.infusionoffset - pa.t0a_vp <= 0)]
    C["ne_rec_at_vpstop"] = C.patientunitstayid.isin(set(pa.patientunitstayid)).astype(int)
    pre, _ = pressor_summary(C, pos, EP)
    # VP peak dose before T0b (units/min records with 0 < rate <= 0.2 only; other units = unknown)
    vr = pre[pre.c_vp].copy()
    vr["dose"] = np.where((vr.unit.fillna("") == "units/min") & (vr.rate > 0) & (vr.rate <= 0.2), vr.rate, np.nan)
    C["vp_peak_pre"] = C.patientunitstayid.map(vr.dropna(subset=["dose"]).groupby("patientunitstayid").dose.max())
    C["vp_dose_unknown"] = ((C.vp == 1) & C.vp_peak_pre.isna()).astype(int)
    risk_set(C, src.trt)
    ur, intake = urine_and_intake_records()
    hr = urine_outcomes(C, src.inf, ur)
    cov = covariates(C, src, intake)
    populations(C)
    C["grp_label"] = C.grp
    # urine output in the 6 h before T0b (mean of recorded hourly rates, bins -6..-1), mL/kg/h
    p6 = hr[(hr.h >= -6) & (hr.h < 0)]
    C["uo_pre6_rate"] = C.patientunitstayid.map(p6.groupby("patientunitstayid").rate.mean())
    C["uo_pre6_bins"] = C.patientunitstayid.map(p6.groupby("patientunitstayid").h.nunique()).fillna(0)
    # urine series anchored at the VP stop T0a(VP) (VP-containing stays)
    ua = ur.merge(C.loc[C.t0a_vp.notna(), ["patientunitstayid", "t0a_vp", "weight_kg"]], on="patientunitstayid")
    ua = ua[ua.weight_kg > 0]
    ua["h"] = np.floor((ua.intakeoutputoffset - ua.t0a_vp) / 60)
    ua = ua[(ua.h >= -24) & (ua.h < 72)]
    hra = ua.groupby(["patientunitstayid", "h"]).agg(ml=("v", "sum"), w=("weight_kg", "first")).reset_index()
    hra["rate"] = hra.ml / hra.w
    series_a = {sid: dict(zip(g.h.astype(int), g.rate)) for sid, g in hra.groupby("patientunitstayid")}
    C["a_onset_h"] = C.patientunitstayid.map({s: onset_series(v, 3.0, 2) for s, v in series_a.items()})
    C["a_poly_pre12"] = C.patientunitstayid.map({s: int(onset_series(v, 3.0, 2, -12, 0) is not None) for s, v in series_a.items()}).fillna(0).astype(int)
    C["a_obs_h_24"] = C.patientunitstayid.map(hra[(hra.h >= 0) & (hra.h < 24)].groupby("patientunitstayid").h.nunique()).fillna(0)
    C["poly_between_t0a_t0b"] = ((C.a_onset_h.notna()) & (C.a_onset_h <= C.gap_vpstop_t0b_h) & (C.gap_vpstop_t0b_h > 0)).astype(int)
    C["a_obs_after_h"] = (C.unitdischargeoffset - C.t0a_vp) / 60
    C["a_riskset"] = ((C.adult == 1) & (C.rrt == 0) &
                      ~(((C.hosp_death == 1) & (C.hospitaldischargeoffset < C.t0a_vp)) | ((C.died_icu == 1) & (C.unitdischargeoffset < C.t0a_vp))) &
                      (C.unitdischargeoffset > C.t0a_vp)).astype(int)
    # outcome / restart helpers
    C["restart24"] = (C.restart_h <= 24).astype(int)
    C["poly24_before_restart"] = ((C.polyuria_24h == 1) & ~(C.restart_h <= C.uo_onset_h)).astype(int)
    C["poly8_before_restart"] = ((C.poly8 == 1) & ~(C.restart_h <= C.uo_onset_h)).astype(int)
    C["poly824_before_restart"] = ((C.poly_8_24 == 1) & ~(C.restart_h <= C.uo_onset_h)).astype(int)
    # rule of the two-group primary analysis: T0b = T0a(VP) and no NE record within 60 min before the VP stop
    C["primary_rule"] = np.where(C.grp == "NE-only", 1, ((C.t0 == C.t0a_vp) & (C.ne_rec_at_vpstop == 0)).astype(int))
    return C, SimpleNamespace(flow=flow, EP=EP, pos=pos, ur=ur, lab=cov.lab, na=cov.na)


def columns_differing_from_stored(C, stored):
    """Columns of the stored cohort table `stored` whose values differ from the cohort C built
    in memory (numeric columns within floating-point tolerance, others as text)."""
    m = C.merge(stored, on="patientunitstayid", suffixes=("", "_o"))
    assert len(m) == len(stored) == len(C), "stored cohort table has other stays than the rebuilt cohort"
    bad = []
    for c in stored.columns:
        if c == "patientunitstayid":
            continue
        a, b = m[c], m[c + "_o"]
        if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
            ok = np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        else:
            ok = ((a.astype(str) == b.astype(str)) | (a.isna() & b.isna())).values
        if (~ok).sum():
            bad.append(c)
    return bad
