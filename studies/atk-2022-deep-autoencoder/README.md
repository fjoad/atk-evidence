# Study 1 — Deep Autoencoder Electricity-Theft Reproduction

**Study ID:** `atk-2022-deep-autoencoder`

This self-contained study directory contains a neutral, auditable reproduction of:

> A. Takiddin, M. Ismail, U. Zafar, and E. Serpedin, "Deep Autoencoder-Based Anomaly Detection of Electricity Theft Cyberattacks in Smart Grids," IEEE Systems Journal, 16(3), 4106-4117, 2022. DOI: 10.1109/JSYST.2021.3136683.

## Primary testable hypothesis

A faithful implementation of the named data, algorithms, preprocessing,
architectures, and evaluation described by the paper will not reproduce its
complete principal numerical result pattern reliably within a predeclared,
reasonable search over unspecified hyperparameters and documented ambiguity
branches.

The first priority is the **paper-literal track**: execute exactly the
methodology the paper describes — nothing more, nothing less — with every
omission filled by a documented sensible assumption, and test whether the
published numbers emerge. It does not silently repair
the method or add practices that the paper does not state. Each material
ambiguity receives a documented reasonable branch that is frozen before
confirmatory results are examined. Reproduction tolerances, seeds, partitions,
search budget, statistics, and stopping rules will likewise be frozen in the
Paper 1 reproduction contract.

This is a falsifiable reproduction hypothesis, not a conclusion or allegation
about intent. A stable paper-consistent reproduction would be recorded as
disconfirming evidence. Conclusions remain bounded by the finite registered
implementation and search space.

## Secondary controlled analysis

Only after the literal result is assessed may a separately labeled controlled
track test whether results change when preprocessing is fitted on training data
only, the test set is untouched, architectures are parameter-matched,
thresholds avoid test labels, and models use repeated seeds and customer splits.
Those experiments cannot substitute for the primary reproduction.

The deferred controlled-solution design is recorded in
[`../../docs/plans/2026-07-23-paper-1-controlled-solution.md`](../../docs/plans/2026-07-23-paper-1-controlled-solution.md).
It includes a transparent-baseline-to-temporal-model ladder, customer-disjoint
evaluation, a causal repair experiment for the decoder/input-domain mismatch,
and an honest comparison against the paper's numbers. Its checkpoint forbids
implementation until the Paper 1 reproduction verdict is frozen.

## Current status

- A source-first visual reconstruction of the complete paper workflow is a
  self-contained file at
  [`../../site/papers/atk-2022-deep-autoencoder/index.html`](../../site/papers/atk-2022-deep-autoencoder/index.html).
  [`PAPER_WORKFLOW.md`](PAPER_WORKFLOW.md) provides the durable study-local
  pointer.
- SGCC: official author-linked dataset downloaded, archive-tested, extracted, and checksummed.
- Irish CER/ISET: six consumption archives match the official ISSDA MD5s; the
  allocation CSV is admitted only as the named exploratory semantic branch.
- Paper specification: extracted, with missing and contradictory details recorded in `EXPERIMENT_SPEC.md`.
- Exploratory SGCC proxy experiments exist but are invalid as Paper 1 reproduction evidence because their 48 inputs are days rather than CER half-hours.
- Exploratory paper-literal runner: implemented with immutable attempts, frozen
  seeds, timing, and raw-score persistence; 20/33 Table II cells have successful
  outcomes.
- Exact CER/ISET preparation is complete. Tables III--V are data-ready; their
  model-execution path is implemented and smoke-tested; full the cluster result
  cells remain.
- Paper-literal confirmatory experiments: not started; exploratory results do
  not retroactively become preregistered evidence.
- Gate C is structurally complete: all 921 paper-consistent branches and 22
  corrected controls resolve by stable ID through the public runner. Gate D is
  partial: local cells and real SGCC preflight pass; a source-v2 exact-ISET
  cache and one real the cluster DDP smoke remain.

## Current command surface

Normal use starts with four short study-root entry points. They are wrappers
over the larger tested audit implementation in `src/`; they are **not** yet the
promised compact reference implementation. A genuine five-file reference track
will add a readable `models.py` and extract one frozen source-faithful route
without depending on the full ambiguity/DDP/evidence engine.

```bash
# 1. Download one named dataset.
.venv/bin/python studies/atk-2022-deep-autoencoder/download_data.py sgcc

# 2. Verify it, apply the paper's preprocessing/attacks, and freeze splits.
.venv/bin/python studies/atk-2022-deep-autoencoder/prepare_data.py sgcc

# 3. Run model/seed cells (or add --preflight for a no-training check).
.venv/bin/python studies/atk-2022-deep-autoencoder/run_experiment.py fc_sae --seeds 11

# Frozen branches are run by stable ID; this also selects preparation/model/evaluation.
.venv/bin/python studies/atk-2022-deep-autoencoder/run_experiment.py \
  --preflight --branch-id sgcc_fc_sae-19b26c8cdff2

# Exact ISET Table III also derives Table V from the same scores/threshold.
.venv/bin/python studies/atk-2022-deep-autoencoder/run_experiment.py \
  fc_sae --dataset iset --table 3 --seeds 11

# Table IV retrains on one or more nested benign-training sizes.
.venv/bin/python studies/atk-2022-deep-autoencoder/run_experiment.py \
  fc_sae --dataset iset --table 4 --sizes half --seeds 11

# Table V identities are independent, explicit experiment families.
.venv/bin/python studies/atk-2022-deep-autoencoder/run_experiment.py \
  fc_sae --dataset iset --table 5 --seeds 11 \
  --table-v-identity retrain_per_attack --table-v-size full_heldout

# 4. Verify artifacts and analyze score separation.
.venv/bin/python studies/atk-2022-deep-autoencoder/analyze_results.py
```

`download_data.py iset` uses the official token-safe ISSDA route.
`prepare_data.py iset` also supports the explicitly approved local ScienceDB
semantic-allocation branch. `run_experiment.py` covers SGCC Table II and exact
ISET Tables III--V. Table III retains the historical fixed-model/fixed-3,000
Table V derivation as a structural invariant-FA diagnostic. Direct
`--table 5` runs cover every registered identity branch: common model/common
benign rows, independent retraining, independent seeded benign resampling, and
retraining plus resampling, each on the full heldout set or seeded
3,000-per-class subsets. All six column-specific scores and identities are
persisted.
When `--branch-id` is used for ISET, the runner requires the matching
content-addressed cache and verifies its embedded preparation ID. The old
implementation-v1 cache is never silently accepted for a source-v2 branch.

## Directory map

- `DATA_SOURCES.md`: provenance, access conditions, file sizes, and checksums.
- `PAPER_WORKFLOW.md`: pointer to the standalone source-first visual map.
- `EXPERIMENT_SPEC.md`: literal paper protocol versus controlled replication protocol.
- `REPRODUCIBILITY_LOG.md`: chronological audit trail and environment details.
- `download_data.py`, `prepare_data.py`, `run_experiment.py`,
  `analyze_results.py`: current small command surface, not a compact codebase.
- `src/`: large forensic harness: parsers, branch implementations, evidence
  persistence, tests, and cluster adapter.
- `results/`: machine-readable metrics and run metadata.
- `../../scripts/cluster/`: short setup, CPU, and one-GPU SLURM wrappers.
- `../../data/raw/`: downloaded raw data; never modified in place.

Repository-wide setup and data acquisition are documented in
[`../../docs/GETTING_STARTED.md`](../../docs/GETTING_STARTED.md).

## Evidence principles

1. Reproduce explicit paper procedures first, including questionable choices;
   document them rather than silently correcting them.
2. Resolve omissions through predeclared reasonable ambiguity branches.
3. Predeclare search spaces, acceptance tolerances, seeds, splits, statistics,
   and stopping rules before confirmatory runs.
4. Store every configuration, failure, raw score, prediction, and seed—not only
   the best result.
5. Require the complete principal result pattern rather than one matching metric.
6. Keep corrected methodology and causal architecture comparisons in a separate
   controlled track.
