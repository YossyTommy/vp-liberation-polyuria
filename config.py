"""Central path and run-time configuration.

Every script imports this module; no other file contains a data or output path.
Locations are taken from environment variables and default to folders inside the
repository (all of them are excluded from version control by .gitignore).

  MIMIC_DIR  MIMIC-IV v3.1 root, containing hosp/ and icu/ (*.csv.gz)   [data/mimiciv/3.1]
  EICU_DIR   eICU-CRD v2.0 root (*.csv.gz)                              [data/eicu-crd/2.0]
  WORK_DIR   row-level intermediates: extracts, caches, cohort tables   [work]
  OUT_DIR    aggregate results: tables, figure data, logs               [out]

Row-level files (anything with a stay or patient identifier) are written only
below WORK_DIR; OUT_DIR receives aggregate tables only.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _path(var, default):
    return os.path.abspath(os.environ.get(var) or os.path.join(ROOT, default))


MIMIC_DIR = _path("MIMIC_DIR", os.path.join("data", "mimiciv", "3.1"))
EICU_DIR = _path("EICU_DIR", os.path.join("data", "eicu-crd", "2.0"))
WORK_DIR = _path("WORK_DIR", "work")
OUT_DIR = _path("OUT_DIR", "out")

MIMIC_WORK = os.path.join(WORK_DIR, "mimic")      # data/ (csv extracts), ext/ (parquet caches), cohort/


def out_dir(name):
    """Aggregate-output folder of one script, e.g. out_dir("mimic_02")."""
    p = os.path.join(OUT_DIR, name)
    os.makedirs(p, exist_ok=True)
    return p


def work_dir(*parts):
    """Row-level working folder below WORK_DIR, created on demand."""
    p = os.path.join(WORK_DIR, *parts)
    os.makedirs(p, exist_ok=True)
    return p
