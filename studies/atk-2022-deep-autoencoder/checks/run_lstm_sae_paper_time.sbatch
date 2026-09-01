#!/bin/bash
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100_16GB:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=lstm-sae-paper-time

set -euo pipefail

: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the frozen paper-time commit}"

cd /export/home/fjoad/atk-evidence

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
    echo "frozen commit mismatch: expected $EXPECTED_COMMIT observed $actual_commit" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "tracked Panther checkout is dirty; refusing paper-time run" >&2
    exit 2
fi

.venv/bin/python \
    studies/atk-2022-deep-autoencoder/checks/lstm_sae_paper_time.py \
    --attempt data/derived/atk-2022-deep-autoencoder/reproduction/remaining-paper-v1-results/feasibility/lstm_sae/seed_20260824_5f53ca7217aa \
    --data data/derived/atk-2022-deep-autoencoder/reproduction/clean-reader-v1-full-printed-semantic-allocation \
    --output data/derived/atk-2022-deep-autoencoder/reproduction/remaining-paper-v1-results
