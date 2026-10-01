#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100_16GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=robust-seq-preflight
set -euo pipefail
: "${REPRO_CHECKOUT:?}"
: "${EXPECTED_COMMIT:?}"
: "${PREFLIGHT_OUTPUT:?}"
: "${LEARNING_RESULT:?}"
: "${LEARNING_SHA:?}"
: "${PYTHON_BIN:?}"
cd "$REPRO_CHECKOUT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$PREFLIGHT_OUTPUT"
export KERAS_BACKEND=tensorflow TF_CPP_MIN_LOG_LEVEL=1 TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=20260920
neural_site="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
neural_cuda_libs=""
for component in cublas cuda_cupti cuda_nvrtc cuda_runtime cudnn cufft curand cusolver cusparse nccl nvjitlink; do
    neural_cuda_libs="$neural_cuda_libs$neural_site/nvidia/$component/lib:"
done
export LD_LIBRARY_PATH="$neural_cuda_libs${LD_LIBRARY_PATH:-}"
"$PYTHON_BIN" -m unittest tests.test_robust_sequential -v
"$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/checks/sequential_preflight.py \
    --output "$PREFLIGHT_OUTPUT" --learning "$LEARNING_RESULT" --learning-sha "$LEARNING_SHA"
"$PYTHON_BIN" -m pip freeze > "$PREFLIGHT_OUTPUT/runtime-packages.txt"
