#!/bin/bash
#SBATCH -J atk_p1_base
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100_16GB:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH --chdir=/export/home/fjoad/atk-evidence

set -euo pipefail
source /etc/profile.d/modules.sh
module load slurm
export KERAS_BACKEND=torch
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

python=.venv/bin/python
study=studies/atk-2022-deep-autoencoder/reproduction
data=data/derived/atk-2022-deep-autoencoder/reproduction/p0-full-none
seed="${SEED:-11}"
output_activation="${OUTPUT_ACTIVATION:-paper}"

git rev-parse HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$python" "$study/download_data.py" iset
"$python" "$study/prepare_data.py" \
  --mode full --test-adasyn none --output "$data"
"$python" "$study/run_experiment.py" \
  --model fc_sae --seed "$seed" --data "$data" \
  --epochs 100 --batch-size 512 --test-view original --table-v \
  --output-activation "$output_activation"
"$python" "$study/analyze_results.py"
