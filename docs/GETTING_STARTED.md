# Getting Started

These instructions take a new contributor from a fresh clone to a verified
local environment and the exact data-access boundary for Study 1. No private
machine path, unpublished file, or chat transcript is required.

## 1. Clone

```bash
git clone https://github.com/fjoad/atk-evidence.git
cd atk-evidence
```

## 2. Prerequisites

- Git
- Python 3.12
- Approximately 5 GB free space for environments, archives, and extracted data
- 7-Zip only for SGCC extraction:
  - macOS: `brew install sevenzip`
  - Debian/Ubuntu: install `7zip` or `p7zip-full`

## 3. Create the pinned environment

```bash
bash scripts/bootstrap.sh
```

The script creates `.venv`, installs the Study 1 lock file, compiles the Python
sources, and runs deterministic tests. Override the Python executable if needed:

```bash
PYTHON_BIN=/path/to/python3.12 bash scripts/bootstrap.sh
```

## 4. Acquire SGCC (anonymous/public)

```bash
bash scripts/acquire_sgcc.sh
```

This script clones the author-linked source at the recorded commit, verifies
all multipart archive SHA-256 values, extracts with 7-Zip into a temporary
directory, verifies the final CSV, and only then moves it into
`data/raw/sgcc-verified/data.csv`. It refuses to overwrite a mismatching file.

## 5. Acquire CER/ISET (authorization required)

The Irish CER Smart Metering consumption archives are restricted. They cannot
be legally or technically downloaded anonymously by this repository.

1. Open the official record: <https://doi.org/10.7929/ISSDA/BX59EU>.
2. Sign in to the UCD/ISSDA Dataverse and submit its data-access request for
   legitimate research or educational use.
3. After approval, create an API token in the account settings.
4. Put the token in the process environment without committing or printing it.
5. Run:

```bash
export ISSDA_API_TOKEN="$(python3 -c 'import getpass; print(getpass.getpass("ISSDA API token: "))')"
./.venv/bin/python studies/atk-2022-deep-autoencoder/src/download_cer.py
unset ISSDA_API_TOKEN
```

The `getpass` prompt hides the token and keeps its value out of shell history.
Do not paste a token into source code, command-line arguments, issues, or logs.

The downloader writes to `data/raw/cer-authorized/`, uses temporary `.part`
files, and verifies every archive against the official MD5 before accepting it.

If institutional policy requires downloading through the browser, put the six
archives in `data/raw/cer-authorized/` with their original names. The verifier
does not care whether the authorized download used the script or browser.

## 6. Verify data

For a status report that allows missing restricted files:

```bash
./.venv/bin/python scripts/verify_data.py
```

For the hard gate required before full Study 1 reproduction:

```bash
./.venv/bin/python scripts/verify_data.py --strict
```

Expected Study 1 files and checksums are also listed in
[`studies/atk-2022-deep-autoencoder/DATA_SOURCES.md`](../studies/atk-2022-deep-autoencoder/DATA_SOURCES.md).

## 7. Run tests

```bash
bash scripts/test.sh
```

## 8. Reproduce the study

Confirmatory reproduction commands are intentionally not published yet. Study
1 is at the contract-freezing stage: the literal algorithm, ambiguity branches,
finite hyperparameter search, tolerances, seeds, and stopping rule must be
approved and committed before any confirmatory runner is enabled. See
[`docs/STATUS.md`](STATUS.md) and the active plan under [`docs/plans/`](plans/).

This gate is part of the scientific method, not missing documentation.
