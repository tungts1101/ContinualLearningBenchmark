#!/bin/bash
# Run all baseline experiments across 9 methods × 7 datasets.
# Usage:
#   ./run_baselines.sh                    # run everything sequentially
#   ./run_baselines.sh aper_ssf           # one method, all datasets
#   ./run_baselines.sh aper_ssf ina       # one method, one dataset
#   ./run_baselines.sh "" ina inr         # all methods, specific datasets
#   DEVICE=1 ./run_baselines.sh           # run on GPU 1 (overrides json device)
#
# Dataset names: cifar  ina  inr  cub  cars  ob  vtab

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config ────────────────────────────────────────────────────────────────────
# Methods and their config file name prefix
METHODS=(
    aper_aperpter
    aper_ssf
    aper_vpt_deep
    l2p
    dualprompt
    coda_prompt
    slca
    ease
    mos
)

# Dataset suffixes and the config filenames they map to:
#   ""      → no suffix (cifar224 base configs, e.g. aper_ssf.json)
#   _ina    → imageneta
#   _inr    → imagenetr
#   _cub    → cub
#   _cars   → omnibenchmark (cars split)
#   _ob     → omnibenchmark (ObjectNet split)
#   _vtab   → omnibenchmark (VTAB split)
DATASET_SUFFIXES=(
    ""
    "_ina"
    "_inr"
    "_cub"
    "_cars"
    "_ob"
    "_vtab"
)

# ── Arg filtering (optional) ─────────────────────────────────────────────────
# First positional arg: method name (empty string or omitted = run all methods)
# Remaining args: dataset names — cifar ina inr cub cars ob vtab
#                 (omitted = run all datasets)

METHOD_FILTER="${1:-}"
shift || true

# Map friendly dataset names → config file suffixes
dataset_to_suffix() {
    case "$1" in
        cifar) echo "" ;;
        *)     echo "_$1" ;;
    esac
}

DATASET_FILTER=()
for ds in "$@"; do
    DATASET_FILTER+=("$(dataset_to_suffix "$ds")")
done

# ── Runner ────────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0

run_exp() {
    local config="$1"
    if [ ! -f "$config" ]; then
        echo "[SKIP] $config (file not found)"
        ((SKIP++)) || true
        return
    fi

    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo "[RUN ] $config"
    echo "══════════════════════════════════════════════════════════════════════"

    if python main.py --config "$config"; then
        echo "[DONE] $config"
        ((PASS++)) || true
    else
        echo "[FAIL] $config  (exit code $?)"
        ((FAIL++)) || true
    fi
}

should_run_dataset() {
    local suffix="$1"
    if [ ${#DATASET_FILTER[@]} -eq 0 ]; then return 0; fi
    for ds in "${DATASET_FILTER[@]}"; do
        if [ "$ds" = "$suffix" ]; then return 0; fi
    done
    return 1
}

for method in "${METHODS[@]}"; do
    # Apply method filter
    if [ -n "$METHOD_FILTER" ] && [ "$method" != "$METHOD_FILTER" ]; then
        continue
    fi

    # Iterate in user-supplied order if a filter is given, else default order
    if [ ${#DATASET_FILTER[@]} -gt 0 ]; then
        suffixes=("${DATASET_FILTER[@]}")
    else
        suffixes=("${DATASET_SUFFIXES[@]}")
    fi

    for suffix in "${suffixes[@]}"; do
        config="exps/${method}${suffix}.json"
        run_exp "$config"
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "[SUMMARY] done=$PASS  failed=$FAIL  skipped=$SKIP"
echo "══════════════════════════════════════════════════════════════════════"

# Exit non-zero if any experiment failed
[ "$FAIL" -eq 0 ]
