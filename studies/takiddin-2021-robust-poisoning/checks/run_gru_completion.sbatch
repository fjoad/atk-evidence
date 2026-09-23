#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100_16GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:25:00
#SBATCH --job-name=robust-gru-complete
set -euo pipefail
: "${REPRO_CHECKOUT:?set isolated checkout}"
: "${EXPECTED_COMMIT:?set frozen revision}"
: "${PREPARATION_ROOT:?set original preparation}"
: "${PILOT_OUTPUT:?set new immutable output directory}"
: "${PREVIOUS_ATTEMPT:?set preserved partial attempt}"
: "${PYTHON_BIN:?set pinned neural interpreter}"
cd "$REPRO_CHECKOUT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$PILOT_OUTPUT"
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
mkdir "$PILOT_OUTPUT"
"$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/checks/gru_completion.py \
    --previous "$PREVIOUS_ATTEMPT" > "$PILOT_OUTPUT/completion_authorization.json"
"$PYTHON_BIN" -m unittest tests.test_robust_gru -v
"$PYTHON_BIN" -m pip freeze > "$PILOT_OUTPUT/runtime-packages.txt"
for level in p00 p30; do
    "$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/reproduction/run_experiment.py \
        --model gru --gru-fit-limit-seconds 2100 \
        --data "$PREPARATION_ROOT/generalized-two-class-$level" --output "$PILOT_OUTPUT/$level"
    if test "$level" = p00; then
        "$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/checks/gru_completion.py \
            --previous "$PREVIOUS_ATTEMPT" --completed-p00 "$PILOT_OUTPUT/p00" > "$PILOT_OUTPUT/repeated_prefix.json"
    fi
done
"$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/reproduction/analyze_results.py \
    "$PILOT_OUTPUT/p00" "$PILOT_OUTPUT/p30" --output "$PILOT_OUTPUT/cluster_comparison.json"
"$PYTHON_BIN" studies/takiddin-2021-robust-poisoning/checks/verify_baseline_pair.py \
    --preparation "$PREPARATION_ROOT" --attempt "$PILOT_OUTPUT" > "$PILOT_OUTPUT/artifact_audit.json"
