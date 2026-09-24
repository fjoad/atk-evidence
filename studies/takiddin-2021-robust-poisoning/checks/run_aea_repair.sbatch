#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=aea-repair
set -euo pipefail
: "${REPRO_CHECKOUT:?set isolated frozen checkout}"
: "${EXPECTED_COMMIT:?set code revision}"
: "${PREPARATION_ROOT:?set original preparation}"
: "${PREVIOUS_OUTPUT:?set preserved geometry attempt}"
: "${DIAGNOSTIC_OUTPUT:?set new output}"
: "${PYTHON_BIN:?set pinned CPU interpreter}"
cd "$REPRO_CHECKOUT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$DIAGNOSTIC_OUTPUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON_BIN" -m unittest tests.test_robust_aea_geometry tests.test_robust_aea_repair -v
"$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/checks/aea_repair.py \
    --preparation "$PREPARATION_ROOT" --previous "$PREVIOUS_OUTPUT" --output "$DIAGNOSTIC_OUTPUT"
"$PYTHON_BIN" -m pip freeze > "$DIAGNOSTIC_OUTPUT/runtime-packages.txt"
