#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"
SOURCE_DIR="${SGCC_SOURCE_DIR:-$REPO_ROOT/data/raw/sgcc-source}"
VERIFIED_DIR="${SGCC_VERIFIED_DIR:-$REPO_ROOT/data/raw/sgcc-verified}"
VERIFIED_CSV="$VERIFIED_DIR/data.csv"
SOURCE_URL="https://github.com/henryRDlab/ElectricityTheftDetection.git"
SOURCE_COMMIT="8db682e65422d24689a61bd044eab7235121c5df"
EXPECTED_CSV_SHA256="99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7"

checksum_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    printf 'No SHA-256 utility found (need sha256sum or shasum).\n' >&2
    return 127
  fi
}

verify_sha256() {
  local target_file="$1"
  local expected_hash="$2"
  local actual_hash
  actual_hash="$(checksum_sha256 "$target_file")"
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    printf 'SHA-256 mismatch for %s\nexpected: %s\nactual:   %s\n' "$target_file" "$expected_hash" "$actual_hash" >&2
    return 1
  fi
  printf 'verified %s\n' "$target_file"
}

if [[ -f "$VERIFIED_CSV" ]]; then
  verify_sha256 "$VERIFIED_CSV" "$EXPECTED_CSV_SHA256"
  printf 'SGCC is already acquired and verified.\n'
  exit 0
fi

if [[ -e "$SOURCE_DIR" && ! -d "$SOURCE_DIR/.git" ]]; then
  printf '%s exists but is not the expected Git checkout; refusing to overwrite it.\n' "$SOURCE_DIR" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  mkdir -p "$(dirname -- "$SOURCE_DIR")"
  git clone "$SOURCE_URL" "$SOURCE_DIR"
fi

if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
  printf 'SGCC source checkout has local changes; refusing to alter it.\n' >&2
  exit 1
fi

git -C "$SOURCE_DIR" fetch --quiet origin "$SOURCE_COMMIT"
git -C "$SOURCE_DIR" checkout --quiet --detach "$SOURCE_COMMIT"

CURRENT_COMMIT="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$SOURCE_COMMIT" ]]; then
  printf 'Unexpected SGCC commit: %s\n' "$CURRENT_COMMIT" >&2
  exit 1
fi

verify_sha256 "$SOURCE_DIR/data.z01" "c324df53c88358a50aa23fd843b1e15af06e7a20b72d901a98d957c304a52b67"
verify_sha256 "$SOURCE_DIR/data.z02" "34a30c8eea0fdfa77d58e15cd01c5593ea354ead4cc408a1b87364f6c46d4ed7"
verify_sha256 "$SOURCE_DIR/data.zip" "1e06ad5f5e13f56f2a72bea304864d259e060d6ad95b3b030a4ad050d8df82d4"

if command -v 7zz >/dev/null 2>&1; then
  SEVEN_ZIP="7zz"
elif command -v 7z >/dev/null 2>&1; then
  SEVEN_ZIP="7z"
else
  printf '7-Zip is required. See docs/GETTING_STARTED.md.\n' >&2
  exit 2
fi

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/atk-sgcc.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT
"$SEVEN_ZIP" x "$SOURCE_DIR/data.zip" "-o$TEMP_DIR" -y >/dev/null

EXTRACTED_CSV="$TEMP_DIR/data.csv"
if [[ ! -f "$EXTRACTED_CSV" ]]; then
  printf 'Archive extraction did not produce data.csv.\n' >&2
  exit 1
fi
verify_sha256 "$EXTRACTED_CSV" "$EXPECTED_CSV_SHA256"

mkdir -p "$VERIFIED_DIR"
mv "$EXTRACTED_CSV" "$VERIFIED_CSV"
printf 'SGCC ready at %s\n' "$VERIFIED_CSV"

