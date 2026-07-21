#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"
PYTHON_EXECUTABLE="${PYTHON_BIN:-python3}"
VENV_DIR="$REPO_ROOT/.venv"
LOCK_FILE="$REPO_ROOT/studies/atk-2022-deep-autoencoder/requirements-lock.txt"

PYTHON_VERSION="$($PYTHON_EXECUTABLE -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.12" ]]; then
  printf 'Python 3.12 is required; %s reports Python %s.\n' "$PYTHON_EXECUTABLE" "$PYTHON_VERSION" >&2
  printf 'Set PYTHON_BIN=/path/to/python3.12 and retry.\n' >&2
  exit 2
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_EXECUTABLE" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --requirement "$LOCK_FILE"
PYTHON_BIN="$VENV_DIR/bin/python" bash "$REPO_ROOT/scripts/test.sh"

printf '\nEnvironment ready: %s\n' "$VENV_DIR"
printf 'Next: bash scripts/acquire_sgcc.sh\n'
printf 'Then follow docs/GETTING_STARTED.md for authorized CER access.\n'

