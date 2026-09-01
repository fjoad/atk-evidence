#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --job-name=remaining-score-recovery

set -euo pipefail

: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the frozen score-recovery commit}"
: "${MODEL:?set MODEL to lstm_sae or lstm_vae}"

case "$MODEL" in
    lstm_sae)
        source_attempt="data/derived/atk-2022-deep-autoencoder/reproduction/remaining-paper-v1-results/feasibility/lstm_sae/seed_20260824_5f53ca7217aa"
        ;;
    lstm_vae)
        source_attempt="data/derived/atk-2022-deep-autoencoder/reproduction/remaining-paper-v1-results/feasibility/lstm_vae/seed_20260824_1d6360ddcead"
        ;;
    *)
        echo "unsupported score-recovery model: $MODEL" >&2
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
    echo "tracked Panther checkout is dirty; refusing score recovery" >&2
    exit 2
fi

.venv/bin/python \
    studies/atk-2022-deep-autoencoder/checks/remaining_score_recovery.py \
    --model "$MODEL" \
    --attempt "$source_attempt" \
    --data data/derived/atk-2022-deep-autoencoder/reproduction/clean-reader-v1-full-printed-semantic-allocation \
    --output data/derived/atk-2022-deep-autoencoder/reproduction/remaining-paper-v1-results \
    --batches 256 128 64 32
