#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=robust-svm
set -euo pipefail
: "${REPRO_CHECKOUT:?set isolated checkout}"
: "${EXPECTED_COMMIT:?set frozen commit}"
: "${PREPARATION_ROOT:?set original verified preparation}"
: "${PILOT_OUTPUT:?set new immutable output path}"
: "${PYTHON_BIN:?set verified interpreter}"
cd "$REPRO_CHECKOUT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$PILOT_OUTPUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON_BIN" -m unittest tests.test_robust_svm -v
mkdir "$PILOT_OUTPUT"
for level in p00 p30; do
    "$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/reproduction/run_experiment.py \
        --model svm --data "$PREPARATION_ROOT/generalized-two-class-$level" \
        --output "$PILOT_OUTPUT/$level"
done
"$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/reproduction/analyze_results.py \
    "$PILOT_OUTPUT/p00" "$PILOT_OUTPUT/p30" --output "$PILOT_OUTPUT/cluster_comparison.json"
"$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/checks/verify_baseline_pair.py \
    --preparation "$PREPARATION_ROOT" --attempt "$PILOT_OUTPUT" > "$PILOT_OUTPUT/artifact_audit.json"
