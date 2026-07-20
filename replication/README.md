# Deep Autoencoder Electricity-Theft Replication

This directory contains a neutral, auditable reproduction of:

> A. Takiddin, M. Ismail, U. Zafar, and E. Serpedin, "Deep Autoencoder-Based Anomaly Detection of Electricity Theft Cyberattacks in Smart Grids," IEEE Systems Journal, 16(3), 4106-4117, 2022. DOI: 10.1109/JSYST.2021.3136683.

## Primary testable hypothesis

A faithful implementation of the named data, algorithms, preprocessing,
architectures, and evaluation described by the paper will not reproduce its
complete principal numerical result pattern reliably within a predeclared,
reasonable search over unspecified hyperparameters and documented ambiguity
branches.

The first priority is the **paper-literal track**. It does not silently repair
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

## Current status

- SGCC: official author-linked dataset downloaded, archive-tested, extracted, and checksummed.
- Irish CER/ISET: official DOI and file manifest identified; consumption files require ISSDA authorization.
- Paper specification: extracted, with missing and contradictory details recorded in `EXPERIMENT_SPEC.md`.
- Result provenance: the component ISET point estimates and thresholds were traced to two earlier papers; their written protocols raise a concrete test-set threshold-selection concern.
- Exploratory SGCC proxy experiments exist but are invalid as Paper 1 reproduction evidence because their 48 inputs are days rather than CER half-hours.
- Paper-literal confirmatory experiments: not started; exact-data and frozen-contract gates remain.

## Directory map

- `DATA_SOURCES.md`: provenance, access conditions, file sizes, and checksums.
- `EXPERIMENT_SPEC.md`: literal paper protocol versus controlled replication protocol.
- `REPRODUCIBILITY_LOG.md`: chronological audit trail and environment details.
- `src/`: executable analysis and model code.
- `results/`: machine-readable metrics and run metadata.
- `../data/raw/`: downloaded raw data; never modified in place.

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
