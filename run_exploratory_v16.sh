#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_exploratory_v16.sh - runs the nine scripts added after v15 (post hoc
# exploratory analyses of manuscript v16), in order. run_all.sh (v15 pipeline)
# must have been run first: the scripts read its cohort tables and outputs.
#
# Usage:  bash run_exploratory_v16.sh
# Paths are taken from MIMIC_DIR, EICU_DIR, WORK_DIR and OUT_DIR (see config.py).
# Every script first recomputes the v15 key values and stops if one differs.
# Row-level output goes to WORK_DIR only; OUT_DIR receives aggregate tables.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
LOGS="$(PYTHONPATH=. "$PY" -c 'import config; print(config.OUT_DIR)')/logs"
mkdir -p "$LOGS"
run () { local name="$1"; shift; echo "[$(date +%H:%M:%S)] $name"; "$@" > "$LOGS/$name.log" 2>&1 || { echo "    FAILED - see $LOGS/$name.log"; exit 1; }; }
run mimic_10_early_late          "$PY" mimic/10_early_late.py
run mimic_11_early_late_extra    "$PY" mimic/11_early_late_extra.py
run mimic_12_interaction         "$PY" mimic/12_interaction.py
run eicu_08_early_late           "$PY" eicu/08_early_late.py
run eicu_09_early_late_extra     "$PY" eicu/09_early_late_extra.py
run eicu_10_interaction          "$PY" eicu/10_interaction.py
run pooled_02_early_late         "$PY" pooled/02_early_late.py
run pooled_03_early_late_extra   "$PY" pooled/03_early_late_extra.py
run pooled_04_interaction        "$PY" pooled/04_interaction.py
echo "finished: exploratory analyses of v16"
