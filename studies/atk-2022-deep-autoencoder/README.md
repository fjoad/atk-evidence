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

- The exact 12-page PDF was independently re-audited on 2026-08-11. Read the
  corrected [`METHOD.md`](METHOD.md) first and the compact
  [`PAPER_WORKFLOW.md`](PAPER_WORKFLOW.md) for the one-page flow.
- The exact CER/ISET consumption archives are verified. The allocation CSV is
  admitted only as the documented semantic-equivalence branch.
- The five direct scientific files exist under [`reproduction/`](reproduction/)
  and do not import the historical `src/` harness.
- Full preparation reached the paper-printed `B2+M` population. Default ADASYN
  on that 14.26-million-row test population did not complete and is preserved
  as an executability finding, not silently relabeled.
- One full no-test-ADASYN FC-SAE seed-11 cluster result exists, but its score and
  eligibility audit is unfinished. It is not a paper-level verdict.
- No experimental job is running. Work is stopped at the renewed source-freeze
  checkpoint before the five-file source-to-code audit.

## Current command surface

The primary implementation is exactly these five files:

```text
reproduction/download_data.py
reproduction/prepare_data.py
reproduction/models.py
reproduction/run_experiment.py
reproduction/analyze_results.py
```

Do not begin with the study-root wrappers or `src/`; those belong to the
historical forensic harness. No command should be run until the renewed source
checkpoint and source-to-code audit are complete. Cluster execution remains a
short resource wrapper around the same five-file scientific path.

## Directory map

- `DATA_SOURCES.md`: provenance, access conditions, file sizes, and checksums.
- `METHOD.md`: current PDF-derived source authority and straight-through route.
- `PAPER_WORKFLOW.md`: compact one-page source flow and pivotal breakpoints.
- `EXPERIMENT_SPEC.md`: literal paper protocol versus controlled replication protocol.
- `REPRODUCIBILITY_LOG.md`: chronological audit trail and environment details.
- `reproduction/`: active five-file reference implementation, pending renewed
  source-to-code audit.
- study-root command files and `src/`: preserved historical forensic harness
  with parsers, branch implementations, evidence persistence, tests, and the
  cluster adapter.
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
