#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"
STUDY_DIR="$REPO_ROOT/studies/atk-2022-deep-autoencoder"
export KERAS_BACKEND="${KERAS_BACKEND:-torch}"
export KERAS_HOME="${KERAS_HOME:-$REPO_ROOT/tmp/keras}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXECUTABLE="$PYTHON_BIN"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_EXECUTABLE="$REPO_ROOT/.venv/bin/python"
elif [[ -x "$REPO_ROOT/replication/.venv/bin/python" ]]; then
  # Compatibility with the pre-publication local workspace only.
  PYTHON_EXECUTABLE="$REPO_ROOT/replication/.venv/bin/python"
else
  PYTHON_EXECUTABLE="python3"
fi

"$PYTHON_EXECUTABLE" -m compileall -q "$STUDY_DIR"
(
  cd "$STUDY_DIR/src"
  "$PYTHON_EXECUTABLE" -m unittest -v \
    test_attacks.py \
    test_branch_lattice.py \
    test_cer_parser.py \
    test_paper_literal_data.py \
    test_paper_literal_iset.py \
    test_paper_literal_iset_runner.py \
    test_paper_literal_metrics.py \
    test_paper_literal_benchmarks.py \
    test_paper_literal_models.py \
    test_paper_source_models.py \
    test_paper_literal_runner.py \
    test_paper_literal_ddp.py \
    test_diagnose_fc_vae_first_step_ddp.py \
    test_probe_recurrent_ddp.py \
    test_aggregate_paper_tables.py
)
(
  cd "$REPO_ROOT"
  "$PYTHON_EXECUTABLE" -m unittest discover -s tests -v
)
