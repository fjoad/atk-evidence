# Getting started

These instructions verify a fresh clone and explain where the current
experiment lives. They do not authorize a new experimental run.

## Install and test

Requirements: Git, Python 3.12, and enough local space for the selected data.

```bash
git clone https://github.com/fjoad/atk-evidence.git
cd atk-evidence
bash scripts/bootstrap.sh
bash scripts/test.sh
```

The bootstrap script creates `.venv`, installs the pinned Paper 1 environment,
compiles the Python sources, and runs deterministic tests. Set `PYTHON_BIN` if
Python 3.12 is not the default.

## Data

### SGCC

The public SGCC acquisition helper downloads an author-linked archive and checks
its recorded hashes:

```bash
bash scripts/acquire_sgcc.sh
```

### CER/ISET

The Irish CER Smart Metering data require authorized access. Open the official
record at <https://doi.org/10.7929/ISSDA/BX59EU> and follow its access terms.
Do not commit a token or restricted files.

Place authorized files under `data/raw/cer-authorized/` using their original
names, then verify:

```bash
.venv/bin/python scripts/verify_data.py
.venv/bin/python scripts/verify_data.py --strict
```

Expected names and checksums are in
[the study data record](../studies/atk-2022-deep-autoencoder/DATA_SOURCES.md).
The current completed run used an explicitly documented semantic-allocation
CSV because the official allocation file's archival serialization was
unavailable. That substitution is recorded in
[the admission decision](decisions/2026-08-30-clean-reader-semantic-allocation-admission.md);
it must never happen silently.

## Current implementation

The active scientific route is:

```text
studies/atk-2022-deep-autoencoder/reproduction/
  download_data.py
  prepare_data.py
  models.py
  run_experiment.py
  analyze_results.py
```

Study-root wrappers and `src/` are older forensic code. They do not define the
clean-reader experiment.

Do not copy an old command from a historical plan. Before any scientific run,
read [STATUS](STATUS.md), the
[clean-reader specification](../studies/atk-2022-deep-autoencoder/CLEAN_READER_SPECIFICATION.md),
and the active plan. The project is currently stopped at Checkpoint 2: no
additional seed, configuration, model, or control is approved.

The completed run and audit are explained in the
[readable finding](../studies/atk-2022-deep-autoencoder/CLEAN_READER_FINDING.md).
Its public page links the exact code revision, configuration, and small saved
records needed to inspect it.
