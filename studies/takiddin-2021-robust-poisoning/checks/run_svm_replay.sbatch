#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=svm-replay
set -euo pipefail
: "${REPRO_CHECKOUT:?set isolated frozen checkout}"
: "${EXPECTED_COMMIT:?set code revision}"
: "${PREPARATION_ROOT:?set preserved preparation}"
: "${BASELINE_ROOT:?set original SVM pair}"
: "${DIAGNOSTIC_OUTPUT:?set new output path}"
: "${PYTHON_BIN:?set original SVM environment}"
cd "$REPRO_CHECKOUT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON_BIN" -m unittest tests.test_robust_svm_replay -v
"$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/checks/svm_replay.py \
    --preparation "$PREPARATION_ROOT" --baseline "$BASELINE_ROOT" --output "$DIAGNOSTIC_OUTPUT"
