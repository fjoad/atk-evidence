# Getting started

How to set up a fresh copy of the repository and get the data.

## Install and test

Requirements: Git, Python 3.12, and enough local space for the data you need.

```bash
git clone https://github.com/fjoad/atk-evidence.git
cd atk-evidence
bash scripts/bootstrap.sh
bash scripts/test.sh
```

The bootstrap script creates `.venv`, installs the pinned environment, compiles
the Python sources and runs the tests. Set `PYTHON_BIN` if Python 3.12 isn't
the default. On macOS, run the tests with `KERAS_TORCH_DEVICE=cpu`.

## Data

### SGCC

The public SGCC acquisition helper downloads an author-linked archive and checks
its recorded hashes:

```bash
bash scripts/acquire_sgcc.sh
```

### CER/ISET

The Irish CER smart-metering data require authorized access. Open the official
record at <https://doi.org/10.7929/ISSDA/BX59EU> and follow its access terms.
Never commit a token or restricted files.

Place authorized files under `data/raw/cer-authorized/` with their original
names, then verify:

```bash
.venv/bin/python scripts/verify_data.py
.venv/bin/python scripts/verify_data.py --strict
```

Expected names and checksums are in
[the study data record](../studies/atk-2022-deep-autoencoder/DATA_SOURCES.md).
Our runs so far used a ScienceDB copy whose allocation CSV matches the
official workbook row for row but isn't the official file. That substitution is
recorded in
[the admission decision](decisions/2026-08-30-clean-reader-semantic-allocation-admission.md)
and must never happen silently.

## Where the code lives

Each paper's rebuild is in `studies/<paper>/reproduction/`. Some studies also
hold older code from earlier attempts; each study's `README.md` says what is
current, and its `RUNNING.md`, where there is one, gives the commands. Experiments on real data run on the Panther
cluster; see [CONTEXT](CONTEXT.md) for access and layout.
