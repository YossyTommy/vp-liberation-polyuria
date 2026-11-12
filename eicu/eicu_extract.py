"""Loading of the raw eICU-CRD tables and one-off filtering of the large ones.

Stays of interest are those with a vasopressin or norepinephrine record in infusionDrug.
intakeOutput, medication, lab, vitalAperiodic and vitalPeriodic are filtered to these stays
with a gunzip | awk pipeline into WORK_DIR/eicu/ex; an existing extract is re-used.
"""
import os
import subprocess
from types import SimpleNamespace

import pandas as pd

from eicu_common import ex_dir, raw

VP_RE = r"vasopressin|pitressin"
NE_RE = r"norepineph|levophed"
# drug classes recognised in infusionDrug.drugname
CLS = {"vp": VP_RE, "ne": NE_RE, "pe": r"phenyleph|neo-?synephrine|neosynephrine",
       "epi": r"(?<!nor)epineph|adrenalin", "dopa": r"dopamine|intropin",
       "ang": r"angiotensin|giapreza", "dob": r"dobutamine|dobutrex", "mil": r"milrinone|primacor"}
PRESS = ["vp", "ne", "pe", "epi", "dopa", "ang"]     # vasopressors
INO = ["dob", "mil"]                                 # inotropes (covariate only)

AWK = "LC_ALL=C awk -v idf='%s' -F',' 'BEGIN{while((getline l<idf)>0)k[l]=1} NR==1{%s;next} ($2 in k)%s{%s}'"


def load_infusions():
    """infusionDrug with numeric rate, unit (text in the trailing parentheses of the drug name)
    and one indicator column c_<class> per drug class."""
    inf = pd.read_csv(raw("infusionDrug.csv.gz"), low_memory=False)
    dn = inf.drugname.astype(str)
    inf["rate"] = pd.to_numeric(inf.drugrate, errors="coerce")
    inf["unit"] = dn.str.extract(r"\(([^)]*)\)\s*$")[0].str.lower()
    for c, rx in CLS.items():
        inf["c_" + c] = dn.str.contains(rx, case=False, regex=True)
    return inf, dn


def extract_large_tables(ids, say):
    """Write the stay list and filter the large raw tables to it (skipped when already present)."""
    exdir = ex_dir()
    idf = os.path.join(exdir, "ids.txt")
    pd.Series(ids).to_csv(idf, index=False, header=False)
    jobs = {
        "x_intakeoutput.csv": ("intakeOutput.csv.gz", AWK % (idf, "print", "", "print")),
        "x_medication.csv": ("medication.csv.gz", AWK % (idf, "print", "", "print")),
        "x_lab.csv": ("lab.csv.gz", AWK % (idf, "print", ' && ($5=="sodium"||$5=="creatinine"||$5=="lactate"||$5=="glucose"||$5=="bedside glucose")', "print")),
        "x_vap.csv": ("vitalAperiodic.csv.gz", AWK % (idf, 'print "patientunitstayid,observationoffset,map"', ' && $6!=""', 'print $2","$3","$6')),
        "x_vp.csv": ("vitalPeriodic.csv.gz", AWK % (idf, 'print "patientunitstayid,observationoffset,map"', ' && $12!=""', 'print $2","$3","$12')),
    }
    procs = []
    for outn, (src, cmd) in jobs.items():
        dst = os.path.join(exdir, outn)
        if os.path.exists(dst):
            continue
        say("extracting", src, "->", outn)
        procs.append(subprocess.Popen(f"gunzip -c '{raw(src)}' | {cmd} > '{dst}.tmp' && mv '{dst}.tmp' '{dst}'", shell=True))
    for p in procs:
        p.wait()


def read_filtered(name, idset):
    """A medium-sized raw table restricted to the stays in `idset` (read in chunks)."""
    out = []
    for ch in pd.read_csv(raw(name + ".csv.gz"), low_memory=False, chunksize=2_000_000):
        out.append(ch[ch.patientunitstayid.isin(idset)])
    return pd.concat(out, ignore_index=True)


def load_source(say):
    """Load every table needed for the cohort construction.

    Returns a namespace with inf (infusionDrug), dn (its drug names as str), ids (sorted stay
    list), pt (patient), apr (apachePatientResult), apv (apachePredVar), dx (diagnosis),
    adx (admissionDx), trt (treatment), rc (respiratoryCare), ph (pastHistory) and
    med (medication); all but pt, apr and apv are restricted to the stays of interest."""
    inf, dn = load_infusions()
    ids = sorted(set(inf.loc[inf.c_vp | inf.c_ne, "patientunitstayid"]))
    idset = set(ids)
    say("infusionDrug rows", len(inf), "; stays with VP or NE record", len(ids))
    extract_large_tables(ids, say)
    pt = pd.read_csv(raw("patient.csv.gz"), low_memory=False)
    apr = pd.read_csv(raw("apachePatientResult.csv.gz"), low_memory=False)
    apv = pd.read_csv(raw("apachePredVar.csv.gz"), low_memory=False)
    dx = read_filtered("diagnosis", idset)
    adx = read_filtered("admissionDx", idset)
    trt = read_filtered("treatment", idset)
    rc = read_filtered("respiratoryCare", idset)
    ph = read_filtered("pastHistory", idset)
    med = pd.read_csv(os.path.join(ex_dir(), "x_medication.csv"), low_memory=False)
    med["drugname"] = med.drugname.astype(str)
    return SimpleNamespace(inf=inf, dn=dn, ids=ids, pt=pt, apr=apr, apv=apv, dx=dx, adx=adx, trt=trt,
                           rc=rc, ph=ph, med=med)
