#!/bin/bash
#SBATCH -p gpu-all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --job-name=cr-iset-fcsae-01

set -euo pipefail

: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the frozen Phase-4 code commit}"

cd /export/home/fjoad/atk-evidence

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
    echo "frozen commit mismatch: expected $EXPECTED_COMMIT observed $actual_commit" >&2
    exit 2
fi

.venv/bin/python \
    studies/atk-2022-deep-autoencoder/reproduction/prepare_data.py \
    --contract clean-reader-v1 \
    --source-branch sciencedb-csv-semantic-equivalence-v1 \
    --output data/derived/atk-2022-deep-autoencoder/reproduction/clean-reader-v1-full-printed-semantic-allocation \
    --force-expensive-adasyn

.venv/bin/python \
    studies/atk-2022-deep-autoencoder/reproduction/run_experiment.py \
    --contract clean-reader-v1 \
    --data data/derived/atk-2022-deep-autoencoder/reproduction/clean-reader-v1-full-printed-semantic-allocation \
    --output data/derived/atk-2022-deep-autoencoder/reproduction/clean-reader-v1-results-semantic-allocation/runs
