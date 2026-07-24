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
- Approximately 10 GB free space for environments, archives, and prepared data
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

## 5. Acquire CER/ISET

The official Irish CER Smart Metering consumption archives are restricted. The
safest fresh-clone route is an approved ISSDA request.

1. Open the official record: <https://doi.org/10.7929/ISSDA/BX59EU>.
2. Sign in to the UCD/ISSDA Dataverse and submit its data-access request for
   legitimate research or educational use.
3. After approval, create an API token in the account settings.
4. Put the token in the process environment without committing or printing it.
5. Run:

```bash
export ISSDA_API_TOKEN="$(python3 -c 'import getpass; print(getpass.getpass("ISSDA API token: "))')"
.venv/bin/python studies/atk-2022-deep-autoencoder/download_data.py iset
unset ISSDA_API_TOKEN
```

The `getpass` prompt hides the token and keeps its value out of shell history.
Do not paste a token into source code, command-line arguments, issues, or logs.

The downloader writes to `data/raw/cer-authorized/`, uses temporary `.part`
files, and verifies every archive against the official MD5 before accepting it.

If institutional policy requires downloading through the browser, select and
download each restricted file separately from the dataset's file table. Put
`File1.txt.zip` through `File6.txt.zip` **and**
`SME and Residential allocations.tab` in `data/raw/cer-authorized/` with their
original names. The verifier does not care whether the authorized download used
the script or browser.

The current exploratory branch was prepared from a ScienceDB deposit whose six
consumption archives are byte-identical to the official files. Its converted
allocation CSV is not the official binary; it is accepted only under the
versioned `sciencedb-csv-semantic-equivalence-v1` branch after checksum,
row-level semantic, and coverage checks. The project does not redistribute
those files or treat the depositor's license label as proof that ISSDA's
conditions were superseded. Full provenance and manual placement instructions
are in the study `DATA_SOURCES.md`.

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

The current executable track is explicitly **exploratory paper-literal**, not a
retrospectively preregistered confirmatory experiment. Normal use has four
study-root commands. First checksum-gate and prepare the exact SGCC input
without fitting a model:

```bash
.venv/bin/python studies/atk-2022-deep-autoencoder/prepare_data.py sgcc
```

The corresponding `iset` subcommand verifies the seven exact CER/ISET files,
constructs complete 48-reading residential profiles, and generates the six
paper-described synthetic attacks. For the current named ScienceDB branch:

```bash
.venv/bin/python studies/atk-2022-deep-autoencoder/prepare_data.py iset
```

The generated cache is an inspectable, checksummed data artifact. The training
runner deliberately re-verifies the raw file and reconstructs preprocessing in
each run instead of trusting a possibly stale cache; on SGCC this adds only a
few seconds relative to model fitting.

Run a declared model/seed cell with the unchanged frozen contract:

```bash
./.venv/bin/python \
  studies/atk-2022-deep-autoencoder/run_experiment.py fc_sae --seeds 11
```

Each invocation appends checksum-verified immutable attempts; it
does not overwrite a previous run or select a favorable seed. Verify retained
artifacts and run the score-separation sanity checks with:

```bash
.venv/bin/python studies/atk-2022-deep-autoencoder/analyze_results.py
```

The simple run entry point currently covers SGCC Table II. Exact ISET
preparation is complete, but Tables III--V remain explicitly unrun until their
small dataset/table-specific execution adapter exists. The internal aggregation
command and documented the cluster workflow remain available to audit or extend
the pipeline; see
local cluster configuration (not published; see the execution policy in `CONTEXT.md`). A later confirmatory runner remains gated on a
separately frozen contract; see [`docs/STATUS.md`](STATUS.md) and the active
plans under [`docs/plans/`](plans/).
