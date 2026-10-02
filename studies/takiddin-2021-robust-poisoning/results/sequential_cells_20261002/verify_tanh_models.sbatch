#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100_16GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:03:00
#SBATCH --job-name=robust-cell-verify
set -euo pipefail
cd "$REPRO_CHECKOUT"
test "$(git rev-parse HEAD)" = febc28e4aaf626302f884524c1184a79f59b2c02
test -z "$(git status --porcelain --untracked-files=no)"
test "$(sha256sum "$CLOSURE_SCRIPT" | cut -d' ' -f1)" = "$CLOSURE_SHA"
test ! -e "$CLOSURE_OUTPUT"
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
"$PYTHON_BIN" "$CLOSURE_SCRIPT" --gpu-reload --code-root "$REPRO_CHECKOUT" \
    --attempt "$DIAGNOSTIC_OUTPUT" --failed-attempt "$FAILED_ATTEMPT" \
    --previous "$PREVIOUS_PAIR" --output "$CLOSURE_OUTPUT"
