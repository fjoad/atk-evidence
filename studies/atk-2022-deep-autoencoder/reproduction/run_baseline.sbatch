#!/bin/bash
#SBATCH -p gpu-all
#SBATCH --gres=gpu:1

cd /export/home/fjoad/atk-evidence
.venv/bin/python studies/atk-2022-deep-autoencoder/reproduction/run_experiment.py \
  --model "$1" --seed "${2:-11}" --score-batch 256
