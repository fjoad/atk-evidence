#!/bin/bash
#SBATCH -J atk_p1_base
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100_16GB:1
#SBATCH -t 2-00:00:00
#SBATCH --chdir=/export/home/fjoad/atk-evidence

set -euo pipefail
source /etc/profile.d/modules.sh
module load slurm
export KERAS_BACKEND=torch
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

python=.venv/bin/python
study=studies/atk-2022-deep-autoencoder/reproduction
data=data/derived/atk-2022-deep-autoencoder/reproduction/p0-full-none
seed="${SEED:-11}"
model="${MODEL:-fc_sae}"
output_activation="${OUTPUT_ACTIVATION:-paper}"
epochs="${EPOCHS:-}"

git rev-parse HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$python" "$study/download_data.py" iset
"$python" "$study/prepare_data.py" \
  --mode full --test-adasyn none --output "$data"
if [[ "$model" == "naive_bayes" || "$model" == "arima" || \
      "$model" == "one_class_svm" || "$model" == "multiclass_svm" ]]; then
  "$python" "$study/run_experiment.py" \
    --model "$model" --seed "$seed" --data "$data" --test-view original
elif [[ "$model" == "supervised_feed_forward" || \
        "$model" == "supervised_lstm" ]]; then
  "$python" "$study/run_experiment.py" \
    --model "$model" --seed "$seed" --data "$data" \
    --epochs "${epochs:-30}" --batch-size 512 --test-view original
else
  "$python" "$study/run_experiment.py" \
    --model "$model" --seed "$seed" --data "$data" \
    --epochs "${epochs:-100}" --batch-size 512 --test-view original \
    --output-activation "$output_activation"
fi
"$python" "$study/analyze_results.py"
