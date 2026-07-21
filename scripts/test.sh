#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"
STUDY_DIR="$REPO_ROOT/studies/atk-2022-deep-autoencoder"

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

"$PYTHON_EXECUTABLE" -m compileall -q "$STUDY_DIR/src"
(
  cd "$STUDY_DIR/src"
  "$PYTHON_EXECUTABLE" -m unittest -v test_attacks.py test_cer_parser.py
)
(
  cd "$REPO_ROOT"
  "$PYTHON_EXECUTABLE" -m unittest discover -s tests -v
)
