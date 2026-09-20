#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=robust-rf-pilot

set -euo pipefail
: "${REPRO_CHECKOUT:?set isolated code checkout}"
: "${EXPECTED_COMMIT:?set frozen code revision}"
: "${PREPARATION_ROOT:?set the unchanged preparation attempt}"
: "${PYTHON_BIN:?set the verified Python environment}"
: "${RF_OUTPUT_ROOT:?set a new immutable attempt directory}"
cd "$REPRO_CHECKOUT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON_BIN" -m unittest tests.test_robust_baseline -q
mkdir "$RF_OUTPUT_ROOT"
for poison in p00 p30; do
    "$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/reproduction/run_experiment.py \
        --data "$PREPARATION_ROOT/generalized-two-class-$poison" \
        --output "$RF_OUTPUT_ROOT/$poison"
done
"$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/reproduction/analyze_results.py \
    "$RF_OUTPUT_ROOT/p00" "$RF_OUTPUT_ROOT/p30" --output "$RF_OUTPUT_ROOT/comparison.json"
