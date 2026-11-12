"""Paths, run-time options and the run log shared by the eICU-CRD scripts.

All locations are derived from config.py (EICU_DIR, WORK_DIR, OUT_DIR); no other module
of this folder builds a data or output path by itself.

  WORK_DIR/eicu/ex/                    raw tables filtered to the stays of interest
  WORK_DIR/eicu/ex_fluid/              intakeOutput filtered to the classified stays
  WORK_DIR/eicu/cohort/                stay-level cohort tables (parquet)
  WORK_DIR/eicu/cohort_timing_rules/   cohort tables under alternative timing rules
  OUT_DIR/eicu_NN/                     aggregate results of script NN
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import config  # noqa: E402

DB = "eICU"
COHORT_PRIMARY = "cohort_primary.parquet"            # two-group cohort (01_primary_analysis.py)
COHORT_USAGE = "cohort_usage_groups.parquet"         # usage / stop-order groups (04_usage_groups.py)
COHORT_FLUID = "cohort_fluid_balance.parquet"        # usage groups + fluid variables (05_fluid_balance.py)
RUN_LOG = "run_log.txt"


def raw(name):
    """Path of a raw eICU-CRD file."""
    return os.path.join(config.EICU_DIR, name)


def ex_dir():
    """Folder of the filtered raw tables."""
    return config.work_dir("eicu", "ex")


def ex_fluid_dir():
    """Folder of the intakeOutput extract used for the fluid balance."""
    return config.work_dir("eicu", "ex_fluid")


def cohort_path(name):
    """Path of a stay-level cohort table."""
    return os.path.join(config.work_dir("eicu", "cohort"), name)


def timing_cohort_path(tag):
    """Path of the two-group cohort table built under the timing-rule variant `tag`."""
    return os.path.join(config.work_dir("eicu", "cohort_timing_rules", tag), COHORT_PRIMARY)


def out_dir(script_number):
    """Aggregate-output folder of a numbered script, e.g. out_dir("01") -> OUT_DIR/eicu_01."""
    return config.out_dir("eicu_" + script_number)


def env_int(name, default):
    """Integer option taken from an environment variable."""
    return int(os.environ.get(name, default))


class RunLog:
    """Collects the messages of one run; they are printed immediately and written at the end."""

    def __init__(self):
        self.lines = []

    def say(self, *parts):
        text = " ".join(str(x) for x in parts)
        print(text, flush=True)
        self.lines.append(text)

    def write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
