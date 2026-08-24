#!/bin/bash
#SBATCH -p gpu-all
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /export/home/fjoad/atk-evidence
.venv/bin/python \
  studies/atk-2022-deep-autoencoder/exploration/phase2_discovery_sandbox.py \
  --output studies/atk-2022-deep-autoencoder/exploration/results/phase2_seed_20260824.json
