#!/bin/bash
#SBATCH --partition=cpu-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=robust-prep

set -euo pipefail
: "${REPRO_CHECKOUT:?set isolated code checkout}"
: "${EXPECTED_COMMIT:?set frozen code revision}"
: "${RAW_CER_DIR:?set existing verified CER directory}"
: "${PYTHON_BIN:?set the verified Python environment}"
: "${PREPARATION_OUTPUT:?set a new immutable attempt directory}"
cd "$REPRO_CHECKOUT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON_BIN" -m unittest tests.test_robust_preparation -q
"$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/reproduction/prepare_data.py \
    --raw "$RAW_CER_DIR" --output "$PREPARATION_OUTPUT" \
    --customers 20 --days-per-customer 28 --rates 0 0.3 --include-customer-specific
