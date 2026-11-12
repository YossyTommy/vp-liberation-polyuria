#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh - runs the pipeline in order.
#
# Usage:  bash run_all.sh [all|mimic|eicu|pooled]     (default: all)
#
# Paths are taken from MIMIC_DIR, EICU_DIR, WORK_DIR and OUT_DIR (see config.py).
# The stages "mimic" and "eicu" are independent and can be run in two shells at
# the same time; "pooled" needs the results of both.
# One log per script and a table of run times are written to OUT_DIR/logs/.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
STAGE="${1:-all}"
LOGS="$(PYTHONPATH=. "$PY" -c 'import config; print(config.OUT_DIR)')/logs"
mkdir -p "$LOGS"

run () {      # run <log name> <command ...>
    local name="$1"; shift
    local t0=$SECONDS
    echo "[$(date +%H:%M:%S)] $name"
    if ! "$@" > "$LOGS/$name.log" 2>&1; then
        echo "    FAILED - see $LOGS/$name.log"; exit 1
    fi
    local dt=$((SECONDS - t0))
    printf '%s\t%d\n' "$name" "$dt" >> "$LOGS/runtime_seconds.tsv"
    echo "    done in $((dt / 60)) min $((dt % 60)) s"
}

mimic () {
    run mimic_01_extract             bash mimic/01_extract.sh
    run mimic_02_primary_analysis    "$PY" mimic/02_primary_analysis.py
    run mimic_03_cif_bootstrap       "$PY" mimic/03_cif_bootstrap.py
    run mimic_04_window_test         "$PY" mimic/04_window_test.py
    run mimic_05_usage_groups        "$PY" mimic/05_usage_groups.py
    run mimic_06_fluid_balance       "$PY" mimic/06_fluid_balance.py
    run mimic_07_vpstop_timing       "$PY" mimic/07_vpstop_timing.py
    run mimic_08_sodium_harmonized   "$PY" mimic/08_sodium_harmonized.py
    run mimic_09_full_charting_table "$PY" mimic/09_full_charting_table.py
}

eicu () {
    run eicu_01_primary_analysis     "$PY" eicu/01_primary_analysis.py
    run eicu_02_cif_ci_timing        "$PY" eicu/02_cif_ci_timing.py
    run eicu_03_window_test          "$PY" eicu/03_window_test.py
    run eicu_04_usage_groups         "$PY" eicu/04_usage_groups.py
    run eicu_05_fluid_balance        "$PY" eicu/05_fluid_balance.py
    run eicu_06_vpstop_timing        "$PY" eicu/06_vpstop_timing.py
    run eicu_07_switch_closed        "$PY" eicu/07_switch_closed.py
}

pooled () {
    run pooled_01_pooled_estimates   "$PY" pooled/01_pooled_estimates.py
}

case "$STAGE" in
    all)     mimic; eicu; pooled ;;
    mimic)   mimic ;;
    eicu)    eicu ;;
    pooled)  pooled ;;
    *)       echo "usage: $0 [all|mimic|eicu|pooled]"; exit 2 ;;
esac
echo "finished: $STAGE"
