"""
mimic_common.py - MIMIC-IV v3.1: definitions, cached extraction, data loading and cohort
construction shared by the numbered MIMIC scripts.

Contents
  definitions       item identifiers, diagnosis key words, names of the cohort tables
  RunLog            prints progress messages and keeps them for the run log of a script
  check_extracts    checks that the csv extracts written by 01_extract.sh exist
  build_caches      cached extraction from the raw files (WORK_DIR/mimic/ext): all inputevents
                    rows of vasopressor-treated stays, ventilation / renal replacement
                    procedures, corticosteroid, desmopressin and vasopressin prescriptions,
                    diagnosis / procedure / DRG codes
  load_all          reads the extracts and caches into a dictionary of data frames
  episodes, anchors infusion episodes and the index time T0 of the VP and NE groups
  hourly_bins, alloc_bins, run_onset, outcomes_from_bins
                    hourly urine output and the polyuria run rule
  build             one row per stay: T0, urine outcomes, covariates, exclusion flags
  cohort, populations   exclusion rules and the primary / secondary population

Definitions
  VP group   any vasopressin (itemid 222315) in the ICU stay; T0 = end of the first VP episode
  NE group   norepinephrine (221906) and no vasopressin in the stay; T0 = end of the first NE episode
  Episode    infusion rows of one drug joined when the gap is <= 1 h (0.5 h and 2 h as sensitivity)
  Vasopressor running at T0
             a non-index vasopressor row with starttime <= T0 < endtime (a vasopressor stopped
             at exactly T0 is not running). Vasopressors: vasopressin, norepinephrine,
             phenylephrine, epinephrine, dopamine, angiotensin II. Dobutamine (221653) and
             milrinone (221986) are covariates only.
  Polyuria   urine output > 3 mL/kg/h in >= 2 consecutive recorded hourly bins (an unrecorded
             hour resets the run); onset = end of the second hour. Outcomes: onset within
             24 h, within 0-8 h and within >8-24 h after T0.
  Primary population
             adult; T0 inside the ICU stay; alive at T0; no renal replacement therapy in the
             stay; no pituitary/hypothalamic/diabetes insipidus code, brain death or pituitary
             surgery; urine recorded in >= 12 of the 24 hourly bins after T0; no polyuria in
             the 12 h before T0; no vasopressor running at T0.

All paths come from config.py (one level above this folder). Nothing is read or written when
the module is imported.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

MIMIC = config.MIMIC_DIR
DATA = os.path.join(config.MIMIC_WORK, "data")     # csv extracts written by 01_extract.sh
EXT = os.path.join(config.MIMIC_WORK, "ext")       # parquet caches built from the raw files

# cohort tables (WORK_DIR/mimic/cohort), written by the script named in the comment
PRIMARY_TABLE = "analysis_ready_mimic_primary.parquet"            # 02 (also _gap0.5 / _gap2.0)
USAGE_TABLE = "analysis_ready_mimic_usage_groups.parquet"         # 05
FLUID_TABLE_2GRP = "mimic_fluid_balance_2grp_anchor.parquet"      # 06
FLUID_TABLE_T0B = "mimic_fluid_balance_T0b_anchor.parquet"        # 06


def cohort_file(name):
    """Path of a cohort table below WORK_DIR/mimic/cohort (the folder is created on demand)."""
    return os.path.join(config.work_dir("mimic", "cohort"), name)


class RunLog:
    """Callable that prints a message and keeps it; write() saves all messages as the run log."""

    def __init__(self):
        self.lines = []

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        self.lines.append(s)
        print(s, flush=True)

    def write(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines))


# ---------------------------------------------------------------------------
# Definitions
VP, NE = 222315, 221906
VASOP = {222315, 221906, 221749, 221289, 221662, 229709, 229764}   # vasopressors
MANNITOL = {227531, 220988, 220989, 220990, 220991, 220992}
HTS = {225161, 228341, 220962}             # NaCl 3%, NaCl 23.4%, Saline 3%
GU_IRRIGANT = 227489
NA_ITEMS = {50983, 52623, 50824}
CRE_ITEMS = {50912}
UOSM_ITEMS = {51093, 52044}
LACT_ITEMS = {50813, 52442}
GLU_ITEMS = {50931, 50809}
STEROID_RX = r"hydrocort|methylpred|dexameth|prednis|fludrocort|cortisone"
TOPICAL = r"cream|oint|ophth|rectal|suppos|enema|foam|ear|eye|lotion|inhal|nasal|soln 0\.1%"
SYS_ROUTES = {"IV", "PO/NG", "PO", "NG", "ORAL", "IV DRIP", "IV BOLUS", "IM", "SC", "G TUBE", "J TUBE", "NU"}
POLY_RATE, POLY_MIN_H = 3.0, 2
KW = {"tbi": r"subdural|traumatic brain|intracranial injury|concussion|cerebral laceration",
      "ich_sah": r"subarachnoid hemorrhage|intracerebral|intracranial hemorrhage|intracranial hypertension",
      "pituitary_di": r"pituit|hypothal|insipidus|hypophys"}
KW["brain_any"] = "|".join(KW.values()) + r"|cerebral infarction|anoxic brain|cerebral edema|brain death|stroke"


def hrs(a, b):
    return (a - b).dt.total_seconds() / 3600


# ===========================================================================
# Input check and cached extraction
COPY_FILES = ["vaso2.csv", "urine.csv", "labs.csv", "labs_extra.csv", "map.csv", "icustays.csv",
              "patients.csv", "admissions.csv", "rrt.csv", "diuretic_iv.csv"]


def check_extracts():
    """Stop with a clear message when a csv extract of 01_extract.sh is missing or empty."""
    missing = [f for f in COPY_FILES
               if not (os.path.exists(os.path.join(DATA, f)) and os.path.getsize(os.path.join(DATA, f)) > 0)]
    if missing:
        raise FileNotFoundError("run mimic/01_extract.sh first; missing in %s: %s" % (DATA, ", ".join(missing)))


def build_caches(say=print):
    """Write the parquet caches in WORK_DIR/mimic/ext that do not exist yet."""
    os.makedirs(EXT, exist_ok=True)
    v = pd.read_csv(os.path.join(DATA, "vaso2.csv"), header=None, usecols=[1], names=["stay_id"])
    sids = set(v.stay_id.dropna().astype("int64"))
    f = os.path.join(EXT, "ie_pressorstays.parquet")
    if not os.path.exists(f):
        say("extract inputevents (pressor stays)")
        cols = ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount", "amountuom",
                "rate", "rateuom", "ordercategorydescription", "patientweight", "statusdescription"]
        out = []
        for ch in pd.read_csv(os.path.join(MIMIC, "icu", "inputevents.csv.gz"), usecols=cols, chunksize=2_000_000):
            out.append(ch[ch.stay_id.isin(sids)])
        d = pd.concat(out)
        for c in ("starttime", "endtime"):
            d[c] = pd.to_datetime(d[c])
        d.to_parquet(f)
    f = os.path.join(EXT, "proc_vent_rrt.parquet")
    if not os.path.exists(f):
        say("extract procedureevents")
        p = pd.read_csv(os.path.join(MIMIC, "icu", "procedureevents.csv.gz"),
                        usecols=["subject_id", "stay_id", "starttime", "endtime", "itemid", "value", "statusdescription"])
        p[p.itemid.isin([225792, 225794, 225802, 225803, 225805, 225809, 225436, 225441, 224270])].to_parquet(f)
    for t in ("diagnoses_icd", "procedures_icd", "d_icd_diagnoses", "d_icd_procedures", "drgcodes", "admissions"):
        f = os.path.join(EXT, t + ".parquet")
        if not os.path.exists(f):
            say("extract", t)
            pd.read_csv(os.path.join(MIMIC, "hosp", t + ".csv.gz")).to_parquet(f)
    f = os.path.join(EXT, "rx_steroid_ddavp.parquet")
    if not os.path.exists(f):
        say("extract prescriptions (steroids, desmopressin)")
        ic = pd.read_csv(os.path.join(DATA, "icustays.csv"))
        hadm = set(ic[ic.stay_id.isin(sids)].hadm_id)
        pat = STEROID_RX + "|desmopressin|ddavp"
        out = []
        for ch in pd.read_csv(os.path.join(MIMIC, "hosp", "prescriptions.csv.gz"),
                              usecols=["subject_id", "hadm_id", "starttime", "stoptime", "drug_type", "drug",
                                       "prod_strength", "dose_val_rx", "dose_unit_rx", "doses_per_24_hrs", "route"],
                              chunksize=2_000_000, low_memory=False):
            ch = ch[ch.hadm_id.isin(hadm)]
            out.append(ch[ch.drug.str.lower().str.contains(pat, regex=True, na=False)])
        pd.concat(out).to_parquet(f)
    f = os.path.join(EXT, "rx_vasopressin.parquet")
    if not os.path.exists(f):
        say("extract prescriptions (vasopressin)")
        out = []
        for ch in pd.read_csv(os.path.join(MIMIC, "hosp", "prescriptions.csv.gz"),
                              usecols=["subject_id", "hadm_id", "starttime", "drug", "route"],
                              chunksize=2_000_000, low_memory=False):
            out.append(ch[ch.drug.str.lower().str.contains(r"vasopressin|pitressin", regex=True, na=False)])
        pd.concat(out).to_parquet(f)


# ===========================================================================
# Loading and cohort construction
def load_all(say=print):
    """Read the extracts and caches; returns a dictionary of data frames (v vasopressor rows,
    u urine, icu, pt, ad, lb / lx laboratory values, mp arterial pressure, dv diuretics, rrt,
    ie inputevents, proc, rx / rxvp prescriptions, flag diagnosis flags per admission)."""
    t = time.time()
    data = {}
    v = pd.read_csv(os.path.join(DATA, "vaso2.csv"), header=None,
                    names=["subject_id", "stay_id", "starttime", "endtime", "itemid", "amount", "rate", "rateuom", "weight"])
    v = v[v.stay_id.notna()].copy()
    v["stay_id"] = v.stay_id.astype("int64")
    for c in ("starttime", "endtime"):
        v[c] = pd.to_datetime(v[c], errors="coerce")
    v = v.dropna(subset=["starttime", "endtime"]).sort_values(["stay_id", "itemid", "starttime"]).reset_index(drop=True)
    data["v"] = v
    stays = set(v.stay_id)
    u = pd.read_csv(os.path.join(DATA, "urine.csv"), header=None, names=["stay_id", "charttime", "itemid", "value"])
    u["charttime"] = pd.to_datetime(u.charttime, errors="coerce")
    data["u"] = u.dropna(subset=["charttime"])
    data["u"] = data["u"][data["u"].stay_id.isin(stays)]
    data["icu"] = pd.read_csv(os.path.join(DATA, "icustays.csv"), parse_dates=["intime", "outtime"])
    data["pt"] = pd.read_csv(os.path.join(DATA, "patients.csv"))
    data["ad"] = pd.read_csv(os.path.join(DATA, "admissions.csv"), parse_dates=["admittime", "dischtime", "deathtime"])
    hadm = set(data["icu"][data["icu"].stay_id.isin(stays)].hadm_id)

    def labs(fn):
        x = pd.read_csv(os.path.join(DATA, fn), header=None, names=["subject_id", "hadm_id", "itemid", "charttime", "v"])
        x = x.dropna(subset=["hadm_id"])
        x["hadm_id"] = x.hadm_id.astype("int64")
        x = x[x.hadm_id.isin(hadm)]
        x["charttime"] = pd.to_datetime(x.charttime, errors="coerce")
        return x.dropna(subset=["charttime", "v"])
    data["lb"] = labs("labs.csv")
    data["lx"] = labs("labs_extra.csv")
    mp = pd.read_csv(os.path.join(DATA, "map.csv"), header=None, names=["stay_id", "charttime", "itemid", "v"])
    mp = mp[mp.stay_id.isin(stays)]
    mp["charttime"] = pd.to_datetime(mp.charttime, errors="coerce")
    data["mp"] = mp.dropna(subset=["charttime"])
    dv = pd.read_csv(os.path.join(DATA, "diuretic_iv.csv"), header=None, names=["stay_id", "starttime", "itemid", "amt"])
    dv = dv[dv.stay_id.isin(stays)]
    dv["starttime"] = pd.to_datetime(dv.starttime, errors="coerce")
    data["dv"] = dv
    rr = pd.read_csv(os.path.join(DATA, "rrt.csv"), header=None, names=["stay_id", "s", "e", "itemid"])
    for c in ("s", "e"):
        rr[c] = pd.to_datetime(rr[c], errors="coerce")
    data["rrt"] = rr
    data["ie"] = pd.read_parquet(os.path.join(EXT, "ie_pressorstays.parquet"))
    pr = pd.read_parquet(os.path.join(EXT, "proc_vent_rrt.parquet"))
    for c in ("starttime", "endtime"):
        pr[c] = pd.to_datetime(pr[c])
    data["proc"] = pr
    rx = pd.read_parquet(os.path.join(EXT, "rx_steroid_ddavp.parquet"))
    for c in ("starttime", "stoptime"):
        rx[c] = pd.to_datetime(rx[c], errors="coerce")
    data["rx"] = rx
    data["rxvp"] = pd.read_parquet(os.path.join(EXT, "rx_vasopressin.parquet"))
    data["flag"] = dx_flags(hadm)
    say("load_all %.0fs" % (time.time() - t))
    return data


def dx_flags(hadm_all):
    dx = pd.read_parquet(os.path.join(EXT, "diagnoses_icd.parquet"))
    pr = pd.read_parquet(os.path.join(EXT, "procedures_icd.parquet"))
    drg = pd.read_parquet(os.path.join(EXT, "drgcodes.parquet"))
    ddx = pd.read_parquet(os.path.join(EXT, "d_icd_diagnoses.parquet"))
    dx = dx[dx.hadm_id.isin(hadm_all)].copy()
    pr = pr[pr.hadm_id.isin(hadm_all)].copy()
    drg = drg[drg.hadm_id.isin(hadm_all)]
    dx["code"] = dx.icd_code.str.strip()
    pr["code"] = pr.icd_code.str.strip()
    st = lambda s, p: s.str.startswith(tuple(p))
    d9, d10 = dx.icd_version == 9, dx.icd_version == 10
    p9, p10 = pr.icd_version == 9, pr.icd_version == 10
    hs = lambda m, tab: set(tab.loc[m, "hadm_id"])
    fl = pd.DataFrame(index=sorted(hadm_all))
    pit_di_icd = (d9 & (st(dx.code, ["253"]) | (dx.code == "5881"))) | (d10 & st(dx.code, ["E22", "E23", "N251"]))
    tumor = (d9 & st(dx.code, ["2273", "2370", "1943"])) | (d10 & st(dx.code, ["D352", "D443", "C751"]))
    fl["pit_di_codes"] = fl.index.isin(hs(pit_di_icd, dx))
    siadh = (d9 & (dx.code == "2536")) | (d10 & (dx.code == "E222"))
    fl["siadh"] = fl.index.isin(hs(siadh, dx))
    fl["pit_di_nosiadh"] = fl.index.isin(hs((pit_di_icd & ~siadh) | tumor, dx))   # pituitary / DI codes without SIADH
    fl["pit_tumor"] = fl.index.isin(hs(tumor, dx))
    fl["pit_di_dx"] = fl.index.isin(hs(pit_di_icd | tumor, dx))
    fl["brain_death"] = fl.index.isin(hs((d9 & (dx.code == "34882")) | (d10 & (dx.code == "G9382")), dx))
    fl["pit_surg"] = fl.index.isin(hs((p9 & st(pr.code, ["076", "077"])) |
                                      (p10 & st(pr.code, ["0GB0", "0GT0", "0GC0", "0G50"])), pr))
    sep = (d9 & st(dx.code, ["038", "99591", "99592", "78552"])) | \
          (d10 & st(dx.code, ["A40", "A41", "R6520", "R6521", "A021", "A227", "A267", "A327", "A427", "B377",
                              "T8144", "O85", "A548"]))
    fl["sepsis"] = fl.index.isin(hs(sep, dx))
    cs = (p9 & st(pr.code, ["361", "351", "352", "3961"])) | \
         (p10 & st(pr.code, ["0210", "0211", "0212", "0213", "02RF", "02RG", "02RH", "02RJ", "02QF", "02QG", "02QH",
                             "02QJ", "02UF", "02UG", "02UH", "02UJ", "5A1221Z"]))
    fl["card_surg_icd"] = fl.index.isin(hs(cs, pr))
    dm = drg[drg.drg_type == "HCFA"]
    dc = pd.to_numeric(dm.drg_code, errors="coerce")
    fl["card_surg_drg"] = fl.index.isin(set(dm.loc[dc.between(216, 236) | dc.between(266, 267), "hadm_id"]))
    dxl = dx.merge(ddx, on=["icd_code", "icd_version"], how="left")
    dxl["t"] = dxl.long_title.fillna("").str.lower()
    fl["brain_any"] = fl.index.isin(set(dxl.loc[dxl.t.str.contains(KW["brain_any"], regex=True), "hadm_id"]))
    adm = pd.read_parquet(os.path.join(EXT, "admissions.parquet")).set_index("hadm_id")
    fl["admission_type"] = adm.admission_type.reindex(fl.index)
    fl["elective"] = fl.admission_type.isin(["ELECTIVE", "SURGICAL SAME DAY ADMISSION"])
    for c in fl.columns:
        if fl[c].dtype == bool:
            fl[c] = fl[c].astype(int)
    return fl


def episodes(v, gap):
    g = v.groupby(["stay_id", "itemid"], sort=False)
    prev_end = g.endtime.cummax().groupby([v.stay_id, v.itemid]).shift(1)
    newep = prev_end.isna() | ((v.starttime - prev_end).dt.total_seconds() / 3600 > gap)
    epid = newep.groupby([v.stay_id, v.itemid]).cumsum()
    return (v.assign(ep_id=epid).groupby(["stay_id", "itemid", "ep_id"], sort=False)
            .agg(start=("starttime", "min"), end=("endtime", "max"), amount=("amount", "sum"),
                 rate_max=("rate", "max")).reset_index())


def anchors(v, gap):
    """T0 and vasopressor status of the VP and NE groups (one row per stay)."""
    ep = episodes(v, gap)
    vp_st = set(ep.loc[ep.itemid == VP, "stay_id"])
    ne_st = set(ep.loc[ep.itemid == NE, "stay_id"])
    out = []
    for grp, drug, stays in (("VP", VP, vp_st), ("NE", NE, ne_st - vp_st)):
        e = ep[(ep.itemid == drug) & ep.stay_id.isin(stays)].sort_values(["stay_id", "start"])
        f = e.groupby("stay_id").first()
        d = pd.DataFrame({"t0": f.end, "idx_start": f.start,
                          "idx_ep1_h": (f.end - f.start).dt.total_seconds() / 3600,
                          "idx_n_ep": e.groupby("stay_id").size()})
        d["grp"] = grp
        out.append(d)
    d = pd.concat(out)
    d.index.name = "stay_id"
    r = v[v.rate.notna() & (v.rate > 0)].merge(d[["t0", "grp"]], left_on="stay_id", right_index=True)
    r = r[r.itemid != np.where(r.grp == "VP", VP, NE)]
    cur_incl = r[(r.starttime <= r.t0) & (r.endtime >= r.t0)]      # inclusive boundary (endtime == T0 counts as running)
    cur = r[(r.starttime <= r.t0) & (r.endtime > r.t0)]            # primary definition: rows continuing beyond T0 only
    d["vaso_at_t0"] = d.index.isin(set(cur.loc[cur.itemid.isin(VASOP), "stay_id"])).astype(int)
    d["vaso_at_t0_incl"] = d.index.isin(set(cur_incl.loc[cur_incl.itemid.isin(VASOP), "stay_id"])).astype(int)
    d["old_other_at_t0"] = d.index.isin(set(cur_incl.stay_id)).astype(int)   # any other agent incl. dobutamine, inclusive boundary
    d["dob_at_t0"] = d.index.isin(set(cur.loc[cur.itemid == 221653, "stay_id"])).astype(int)
    sw = r[r.itemid.isin(VASOP) & (r.starttime > r.t0) & (r.starttime <= r.t0 + pd.Timedelta(hours=1))]
    d["switch_1h"] = d.index.isin(set(sw.stay_id)).astype(int)     # another vasopressor started within 1 h after T0
    ne = ep[ep.itemid == NE].merge(d[["t0"]], left_on="stay_id", right_index=True)
    d["ne_at_t0"] = d.index.isin(set(ne.loc[(ne.start <= ne.t0) & (ne.end > ne.t0), "stay_id"])).astype(int)
    d.loc[d.grp == "NE", "ne_at_t0"] = np.nan
    e2 = ep[ep.itemid.isin(VASOP)].merge(d[["t0"]], left_on="stay_id", right_index=True)
    e2 = e2[e2.start > e2.t0]
    d["restart_t"] = e2.assign(h=hrs(e2.start, e2.t0)).groupby("stay_id").h.min()
    e3 = ep.merge(d[["t0", "grp"]], left_on="stay_id", right_index=True)
    e3 = e3[(((e3.grp == "VP") & (e3.itemid == VP)) | ((e3.grp == "NE") & (e3.itemid == NE))) & (e3.start > e3.t0)]
    d["idx_restart_t"] = e3.assign(h=hrs(e3.start, e3.t0)).groupby("stay_id").h.min()
    pre = v.merge(d[["t0"]], left_on="stay_id", right_index=True)
    pre = pre[pre.starttime < pre.t0]
    pv = pre[pre.itemid.isin(VASOP)]
    d["press_h"] = (d.t0 - pv.groupby("stay_id").starttime.min()).dt.total_seconds() / 3600
    d["ne_peak"] = pre[(pre.itemid == NE) & (pre.rateuom == "mcg/kg/min")].groupby("stay_id").rate.max() \
        .reindex(d.index).fillna(0)
    d["other_vaso_pre"] = d.index.isin(set(pre.loc[pre.itemid.isin(VASOP - {VP, NE}), "stay_id"])).astype(int)
    d["dopamine_pre"] = d.index.isin(set(pre.loc[pre.itemid == 221662, "stay_id"])).astype(int)
    d["weight"] = v.groupby("stay_id").weight.median().reindex(d.index)
    return d.reset_index()


# --- urine output ---------------------------------------------------------------
def run_onset(rates, lo, hi, thr, nh=POLY_MIN_H):
    run = 0
    for hh in range(lo, hi):
        x = rates.get(hh)
        if x is None:
            run = 0
            continue
        if x > thr:
            run += 1
            if run >= nh:
                return hh + 1
        else:
            run = 0
    return None


def hourly_bins(src, d):
    m = src.merge(d[["stay_id", "t0", "weight"]], on="stay_id", how="inner")
    m["h"] = np.floor((m.charttime - m.t0).dt.total_seconds() / 3600)
    m = m[(m.h >= -12) & (m.h < 72) & (m.weight > 0)]
    hr = m.groupby(["stay_id", "h"]).agg(ml=("value", "sum"), wt=("weight", "first")).reset_index()
    hr["rate"] = hr.ml / hr.wt
    return hr


def alloc_bins(src, d, maxgap=6.0):
    """Interval apportioning: each charted volume is spread over the interval since the previous record;
    with no previous record or an interval > 6 h it is assigned to the hour of charting."""
    m = src.merge(d[["stay_id", "t0", "weight"]], on="stay_id", how="inner")
    m["x"] = (m.charttime - m.t0).dt.total_seconds() / 3600
    m = m[(m.x >= -36) & (m.x < 80) & (m.weight > 0)].sort_values(["stay_id", "charttime"])
    m = m.groupby(["stay_id", "x"], as_index=False).agg(value=("value", "sum"), wt=("weight", "first"))
    m["prev"] = m.groupby("stay_id").x.shift(1)
    S, X, P, V, W = m.stay_id.values, m.x.values, m.prev.values, m.value.values, m.wt.values
    rows = []
    for i in range(len(m)):
        gp = X[i] - P[i]
        if np.isnan(P[i]) or gp > maxgap or gp <= 0:
            rows.append((S[i], np.floor(X[i]), V[i], W[i]))
            continue
        a, b = P[i], X[i]
        rate = V[i] / (b - a)
        h = np.floor(a)
        while h < b:
            seg = min(b, h + 1) - max(a, h)
            if seg > 0:
                rows.append((S[i], h, rate * seg, W[i]))
            h += 1
    r = pd.DataFrame(rows, columns=["stay_id", "h", "ml", "wt"])
    r = r[(r.h >= -12) & (r.h < 72)]
    hr = r.groupby(["stay_id", "h"], as_index=False).agg(ml=("ml", "sum"), wt=("wt", "first"))
    hr["rate"] = hr.ml / hr.wt
    return hr


def outcomes_from_bins(hr, suffix=""):
    rows = []
    for sid, g in hr.groupby("stay_id"):
        rr = dict(zip(g.h.astype(int), g.rate))
        rows.append((sid, run_onset(rr, 0, 72, 3.0), run_onset(rr, 0, 72, 5.0),
                     int(run_onset(rr, -12, 0, 3.0) is not None),
                     sum(1 for hh in rr if 0 <= hh < 24), g.loc[(g.h >= 0) & (g.h < 24), "ml"].sum()))
    r = pd.DataFrame(rows, columns=["stay_id", "onset", "onset5", "pre_poly", "obs24", "uo24_ml"])
    r.columns = ["stay_id"] + [c + suffix for c in r.columns[1:]]
    return r


def build(data, gap=1.0, anchor_table=None):
    """Analysis table (one row per stay) from the data frames of load_all().

    anchor_table: output of anchors() with T0 and the T0-dependent columns already replaced
    (used for the analyses anchored at another time); by default anchors(v, gap) is used.
    Returns the table, the co-intervention times and the desmopressin orders."""
    v, u = data["v"], data["u"]
    d = anchors(v, gap) if anchor_table is None else anchor_table.copy()
    D0 = d[["stay_id", "t0", "weight"]]
    for o in (outcomes_from_bins(hourly_bins(u, D0)),
              outcomes_from_bins(hourly_bins(u[u.itemid != GU_IRRIGANT], D0), "_nogu"),
              outcomes_from_bins(alloc_bins(u, D0), "_alloc")):
        d = d.merge(o, on="stay_id", how="left")
    for c in [c for c in d.columns if c.startswith(("obs24", "pre_poly"))]:
        d[c] = d[c].fillna(0).astype(int)
    for c in [c for c in d.columns if c.startswith("uo24_ml")]:
        d[c] = d[c].fillna(0)
    d = (d.merge(data["icu"], on="stay_id", how="left")
          .merge(data["pt"][["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
          .merge(data["ad"][["hadm_id", "deathtime", "hospital_expire_flag"]], on="hadm_id", how="left")
          .merge(data["flag"], left_on="hadm_id", right_index=True, how="left"))
    t0 = d.set_index("stay_id").t0
    sid = set(d.stay_id)
    rr = data["rrt"][data["rrt"].stay_id.isin(sid)].copy()
    rr["h"] = hrs(rr.s, rr.stay_id.map(t0))
    d["rrt_any"] = d.stay_id.isin(set(rr.stay_id)).astype(int)
    d["rrt_pre"] = d.stay_id.isin(set(rr.loc[rr.h < 0, "stay_id"])).astype(int)
    d["rrt_start_t"] = d.stay_id.map(rr[rr.h >= 0].groupby("stay_id").h.min())
    d["death_t"] = hrs(d.deathtime, d.t0)
    d["icuout_t"] = hrs(d.outtime, d.t0)
    # laboratory values
    L = data["lb"].merge(d[["hadm_id", "stay_id", "t0"]], on="hadm_id")
    L["h"] = hrs(L.charttime, L.t0)
    L = L.sort_values("charttime")

    def agg(items, lo, hi, how):
        s = L[L.itemid.isin(items) & (L.h > lo) & (L.h <= hi)]
        return d.stay_id.map(s.groupby("stay_id").v.agg(how))
    d["na_pre"] = agg(NA_ITEMS, -24, 0, "last")
    d["na_max_pre24"] = agg(NA_ITEMS, -24, 0, "max")
    d["na_peak_24"] = agg(NA_ITEMS, 0, 24, "max")
    d["na_peak_72"] = agg(NA_ITEMS, 0, 72, "max")
    d["na_peak_7d"] = agg(NA_ITEMS, 0, 168, "max")
    d["uosm_min_7d"] = agg(UOSM_ITEMS, 0, 168, "min")
    d["na_n72"] = agg(NA_ITEMS, 0, 72, "count").fillna(0)                 # number of Na tests (detection bias)
    d["na_fix24"] = agg(NA_ITEMS, 18, 30, "max")                           # fixed time windows
    d["na_fix48"] = agg(NA_ITEMS, 42, 54, "max")
    d["na_peak_72_chem"] = agg({50983, 52623}, 0, 72, "max")              # chemistry analyser (indirect ISE) only
    cb, cp = agg(CRE_ITEMS, -720, 0, "min"), agg(CRE_ITEMS, -168, 0, "max")
    r = cp / cb
    k = np.select([r >= 3, r >= 2, (r >= 1.5) | (cp - cb >= 0.3)], [3, 2, 1], default=0).astype(float)
    k[(cp.isna() | cb.isna()).values] = np.nan
    d["kdigo_pre"] = k
    X = data["lx"].merge(d[["hadm_id", "stay_id", "t0"]], on="hadm_id")
    X["h"] = hrs(X.charttime, X.t0)
    d["lactate_pre24"] = d.stay_id.map(X[X.itemid.isin(LACT_ITEMS) & (X.h > -24) & (X.h <= 0)].groupby("stay_id").v.max())
    d["glu_max_post24"] = d.stay_id.map(X[X.itemid.isin(GLU_ITEMS) & (X.h > 0) & (X.h <= 24)].groupby("stay_id").v.max())
    P = data["mp"][data["mp"].stay_id.isin(sid)].merge(d[["stay_id", "t0"]], on="stay_id")
    P["h"] = hrs(P.charttime, P.t0)
    d["map_base"] = d.stay_id.map(P[(P.h >= -6) & (P.h < 0)].groupby("stay_id").v.median())
    Q = data["dv"][data["dv"].stay_id.isin(sid)].merge(d[["stay_id", "t0"]], on="stay_id")
    Q["h"] = hrs(Q.starttime, Q.t0)
    d["diur_pre24"] = d.stay_id.isin(set(Q.loc[(Q.h >= -24) & (Q.h < 0), "stay_id"])).astype(int)
    d["loop_post24"] = d.stay_id.isin(set(Q.loc[(Q.h >= 0) & (Q.h <= 24), "stay_id"])).astype(int)
    # inputevents
    I = data["ie"][data["ie"].stay_id.isin(sid)].merge(d[["stay_id", "t0"]], on="stay_id")
    I["h"], I["he"] = hrs(I.starttime, I.t0), hrs(I.endtime, I.t0)
    fl = I[(I.amountuom == "mL") & (I.amount > 0) & (I.h >= -24) & (I.h < 0)]
    d["fluid_pre24_l"] = d.stay_id.map(fl.groupby("stay_id").amount.sum()).fillna(0) / 1000
    man = I[I.itemid.isin(MANNITOL) & (I.amount > 0)]
    hts = I[I.itemid.isin(HTS) & (I.amount > 0)]
    d["mannitol_post24"] = d.stay_id.isin(set(man.loc[(man.he >= 0) & (man.h <= 24), "stay_id"])).astype(int)
    d["hts_post24"] = d.stay_id.isin(set(hts.loc[(hts.he >= 0) & (hts.h <= 24), "stay_id"])).astype(int)
    mil = I[(I.itemid == 221986) & (I.rate > 0)]
    dob = v[(v.itemid == 221653) & (v.rate > 0) & v.stay_id.isin(sid)].merge(d[["stay_id", "t0"]], on="stay_id")
    mil_t0 = set(mil.loc[(mil.h <= 0) & (mil.he > 0), "stay_id"])
    dob_t0 = set(dob.loc[(dob.starttime <= dob.t0) & (dob.endtime > dob.t0), "stay_id"])
    d["inotrope_t0"] = d.stay_id.isin(mil_t0 | dob_t0).astype(int)
    d["milrinone_t0"] = d.stay_id.isin(mil_t0).astype(int)
    d["inotrope_post24"] = d.stay_id.isin(set(mil.loc[(mil.he > 0) & (mil.h <= 24), "stay_id"]) |
                                          set(dob.loc[(dob.endtime > dob.t0) & (hrs(dob.starttime, dob.t0) <= 24), "stay_id"])).astype(int)
    coint = pd.concat([Q.assign(he=Q.h)[["stay_id", "h", "he"]].assign(kind="loop"),
                       man[["stay_id", "h", "he"]].assign(kind="mannitol"),
                       hts[["stay_id", "h", "he"]].assign(kind="hts")])
    # mechanical ventilation, corticosteroids, desmopressin
    V_ = data["proc"][(data["proc"].itemid == 225792) & data["proc"].stay_id.isin(sid)].merge(d[["stay_id", "t0"]], on="stay_id")
    d["vent_t0"] = d.stay_id.isin(set(V_.loc[(V_.starttime <= V_.t0) & (V_.endtime >= V_.t0), "stay_id"])).astype(int)
    rx = data["rx"]
    S_ = rx[rx.drug.str.lower().str.contains(STEROID_RX, regex=True) &
            ~rx.drug.str.lower().str.contains(TOPICAL, regex=True) & rx.route.isin(SYS_ROUTES)]
    S_ = S_.merge(d[["hadm_id", "stay_id", "t0"]], on="hadm_id")
    d["steroid_pre24"] = d.stay_id.isin(set(S_.loc[(S_.starttime < S_.t0) &
                                                   (S_.stoptime.fillna(S_.starttime) > S_.t0 - pd.Timedelta(hours=24)),
                                                   "stay_id"])).astype(int)
    Dd = rx[rx.drug.str.lower().str.contains("desmopressin|ddavp", regex=True)] \
        .merge(d[["hadm_id", "stay_id", "t0", "weight", "grp"]], on="hadm_id")
    Dd["h"] = hrs(Dd.starttime, Dd.t0)
    Dd["dose"] = pd.to_numeric(Dd.dose_val_rx.astype(str).str.extract(r"([\d\.]+)")[0], errors="coerce")
    un = Dd.dose_unit_rx.str.lower().fillna("")
    Dd["dose_mcg"] = np.where(un.str.contains("mcg"), Dd.dose, np.where(un.str.contains("mg"), Dd.dose * 1000, np.nan))
    Dd["route_c"] = np.select([Dd.route.isin(["IV", "IV DRIP", "IV BOLUS"]), Dd.route.isin(["SC", "SUBCUT"]),
                               Dd.route.isin(["NAS", "NU"]), Dd.route.isin(["PO", "PO/NG", "NG", "ORAL", "SL"])],
                              ["IV", "SC", "nasal", "oral"], default="other")
    Dd["hemostatic"] = (Dd.route_c.isin(["IV", "SC"]) & ((Dd.dose_mcg >= 15) | (Dd.dose_mcg / Dd.weight >= 0.3))).astype(int)
    Dd["antidiuretic"] = (Dd.route_c.isin(["IV", "SC", "nasal"]) & (Dd.hemostatic == 0)).astype(int)
    ad_ = Dd[Dd.antidiuretic == 1]
    d["ddavp_ad_48"] = d.stay_id.isin(set(ad_.loc[(ad_.h >= 0) & (ad_.h <= 48), "stay_id"])).astype(int)
    d["ddavp_ad_7d"] = d.stay_id.isin(set(ad_.loc[(ad_.h >= 0) & (ad_.h <= 168), "stay_id"])).astype(int)
    d["ddavp_any_48"] = d.stay_id.isin(set(Dd.loc[(Dd.h >= 0) & (Dd.h <= 48), "stay_id"])).astype(int)
    # derived variables
    d["vp"] = (d.grp == "VP").astype(int)
    d["age10"] = d.anchor_age / 10
    d["male"] = (d.gender == "M").astype(int)
    d["cvicu"] = d.first_careunit.str.contains("Cardiac Vascular", na=False).astype(int)
    d["card_surg"] = ((d.card_surg_icd == 1) | ((d.card_surg_drg == 1) & (d.cvicu == 1))).astype(int)
    d["emergency"] = 1 - d.elective.fillna(0).astype(int)
    d["log_press_h"] = np.log1p(d.press_h.clip(lower=0))
    d["ne_peak_c"] = d.ne_peak.clip(upper=2) / 0.1          # per 0.1 mcg/kg/min
    d["kdigo_cat"] = d.kdigo_pre.map({0: "0", 1: "1", 2: "2", 3: "3"}).fillna("unk")
    d["ex_age"] = (d.anchor_age < 18).astype(int)
    d["ex_notinicu"] = (~((d.t0 >= d.intime) & (d.t0 <= d.outtime))).astype(int)
    d["ex_deadpre"] = (d.deathtime.notna() & (d.deathtime < d.t0)).astype(int)
    d["ex_neuroendo"] = ((d.pit_di_dx == 1) | (d.brain_death == 1) | (d.pit_surg == 1)).astype(int)
    d["ex_neuroendo_nosiadh"] = ((d.pit_di_nosiadh == 1) | (d.brain_death == 1) | (d.pit_surg == 1)).astype(int)
    # vasopressin exposure of NE stays elsewhere in the admission (another ICU stay or a prescription)
    vp_hadm = set(data["icu"].loc[data["icu"].stay_id.isin(set(v.loc[v.itemid == VP, "stay_id"])), "hadm_id"])
    rx_hadm = set(data["rxvp"].hadm_id.dropna().astype("int64"))
    d["ne_vp_same_hadm"] = ((d.grp == "NE") & d.hadm_id.isin(vp_hadm)).astype(int)
    d["ne_vp_rx_hadm"] = ((d.grp == "NE") & d.hadm_id.isin(rx_hadm)).astype(int)
    d = add_outcomes(d)
    return d, coint, Dd


def add_outcomes(d):
    d = d.copy()
    for sfx in ("", "_nogu", "_alloc"):
        o = d["onset" + sfx]
        d["poly24" + sfx] = (o <= 24).astype(int)
        d["poly8" + sfx] = (o <= 8).astype(int)
        d["poly_8_24" + sfx] = ((o > 8) & (o <= 24)).astype(int)
    d["poly72"] = d.onset.notna().astype(int)
    d["marked24"] = (d.onset5 <= 24).astype(int)
    d["vol24"] = ((d.uo24_ml > 3000) | (d.uo24_ml / d.weight > 50)).astype(int)
    d["eval12"] = (d.obs24 >= 12).astype(int)
    d["log_lact"] = np.log(d.lactate_pre24.clip(lower=0.1))
    d["restart24"] = (d.restart_t <= 24).astype(int)
    d["na150_72"] = (d.na_peak_72 >= 150).astype(int)
    d["na_rise8_72"] = ((d.na_peak_72 - d.na_pre) >= 8).astype(int)
    d["severe_wd"] = ((d.poly72 == 1) & (d.na_peak_7d >= 145) & ((d.uosm_min_7d < 300) | (d.ddavp_ad_7d == 1))).astype(int)
    d["press_ge6"] = (d.press_h >= 6).astype(int)
    d["glu250_24"] = (d.glu_max_post24 >= 250).astype(int)
    d["rrt_start24"] = (d.rrt_start_t <= 24).astype(int)
    rs = d.restart_t.fillna(np.inf)
    d["poly24_prerestart"] = ((d.onset <= 24) & (d.onset <= rs)).astype(int)
    d["poly8_prerestart"] = ((d.onset <= 8) & (d.onset <= rs)).astype(int)
    d["poly_8_24_prerestart"] = ((d.onset > 8) & (d.onset <= 24) & (d.onset <= rs)).astype(int)
    rt = d.rrt_start_t.fillna(np.inf)
    d["poly24_prerrt"] = ((d.onset <= 24) & (d.onset <= rt)).astype(int)
    d["poly8_prerrt"] = ((d.onset <= 8) & (d.onset <= rt)).astype(int)
    d["poly_8_24_prerrt"] = ((d.onset > 8) & (d.onset <= 24) & (d.onset <= rt)).astype(int)
    return d


def cohort(d, obs_thr=12, rrt_rule="any", neuro=True, obscol="obs24", brain_any=False):
    m = (d.ex_age == 0) & (d.ex_notinicu == 0) & (d.ex_deadpre == 0)
    m &= (d.rrt_any == 0) if rrt_rule == "any" else (d.rrt_pre == 0)
    if neuro:
        m &= d.ex_neuroendo == 0
    if brain_any:
        m &= d.brain_any == 0
    if obs_thr:
        m &= d[obscol] >= obs_thr
    return m


def populations(d):
    base = cohort(d)
    return {"primary": base & (d.vaso_at_t0 == 0) & (d.pre_poly == 0),
            "secondary": base & (d.pre_poly == 0)}
