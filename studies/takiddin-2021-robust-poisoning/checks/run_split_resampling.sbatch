#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=robust-split

set -euo pipefail
: "${REPRO_CHECKOUT:?set isolated code checkout}"
: "${EXPECTED_COMMIT:?set frozen revision}"
: "${PREPARATION_ROOT:?set frozen preparation attempt}"
: "${BASELINE_ROOT:?set saved forest pilot}"
: "${CONTROL_OUTPUT:?set a new immutable output directory}"
: "${PYTHON_BIN:?set verified Python environment}"
cd "$REPRO_CHECKOUT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON_BIN" -m unittest tests.test_robust_split_control -q
"$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/checks/split_resampling.py \
    --preparation "$PREPARATION_ROOT/generalized-two-class-p00" \
    --baseline "$BASELINE_ROOT" --output "$CONTROL_OUTPUT"
