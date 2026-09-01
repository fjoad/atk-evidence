#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=remaining-model-pilot

set -euo pipefail

: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the frozen pilot commit}"
: "${MODEL:?set MODEL to one approved remaining Table-III model}"

case "$MODEL" in
    lstm_sae|fc_vae|lstm_vae|lstm_aea) ;;
    *)
        echo "unsupported remaining pilot model: $MODEL" >&2
        exit 2
        ;;
esac

cd /export/home/fjoad/atk-evidence

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
    echo "frozen commit mismatch: expected $EXPECTED_COMMIT observed $actual_commit" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "tracked Panther checkout is dirty; refusing the pilot" >&2
    exit 2
fi

.venv/bin/python \
    studies/atk-2022-deep-autoencoder/reproduction/run_experiment.py \
    --contract remaining-paper-feasibility-v1 \
    --model "$MODEL" \
    --data data/derived/atk-2022-deep-autoencoder/reproduction/clean-reader-v1-full-printed-semantic-allocation \
    --output data/derived/atk-2022-deep-autoencoder/reproduction/remaining-paper-v1-results \
    --epochs 2 \
    --minimum-epochs 2 \
    --batch-size 32 \
    --score-batch 256
