#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_additions_v17.sh - runs the five scripts added for manuscript v17.
# No new analysis: (1) the post hoc sodium-rise models of v16 are fitted again
# to obtain clustered and model-based standard errors (mimic/13, eicu/11,
# pooled/05); (2) stay-level bootstrap confidence bands are calculated for the
# closed-cohort time course of Figure 4 (mimic/14, eicu/12).
# run_all.sh (v15 pipeline) and run_exploratory_v16.sh must have been run first.
#
# Usage:  bash run_additions_v17.sh
# Paths are taken from MIMIC_DIR, EICU_DIR, WORK_DIR and OUT_DIR (see config.py).
# Every script first checks that its point estimates equal the earlier outputs
# and stops if one differs. Row-level data stay in WORK_DIR; OUT_DIR receives
# aggregate tables only.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
LOGS="$(PYTHONPATH=. "$PY" -c 'import config; print(config.OUT_DIR)')/logs"
mkdir -p "$LOGS"
run () { local name="$1"; shift; echo "[$(date +%H:%M:%S)] $name"; "$@" > "$LOGS/$name.log" 2>&1 || { echo "    FAILED - see $LOGS/$name.log"; exit 1; }; }
run mimic_13_se_check     "$PY" mimic/13_se_check.py
run eicu_11_se_check      "$PY" eicu/11_se_check.py
run pooled_05_se_check    "$PY" pooled/05_se_check.py
run mimic_14_fig4_bands   "$PY" mimic/14_fig4_bands.py
run eicu_12_fig4_bands    "$PY" eicu/12_fig4_bands.py
echo "finished: additions of v17"
