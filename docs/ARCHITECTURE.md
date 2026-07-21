# ATK Evidence — Architecture

**Last updated:** 2026-07-21

## Overview

The repository is a domain-neutral evidence pipeline organized around
independent target papers. Local source PDFs and raw datasets feed versioned
paper specifications, literal implementations, immutable run outputs,
statistical assessments, and per-paper LaTeX reports. Charter preserves intent,
state, and causal evidence across Claude and Codex sessions.

## Structure

```text
atk-evidence/
  docs/                       # Charter vision, status, memory, evidence, plans
  papers/                     # Local source PDFs; ignored by Git
  data/                       # Local raw/derived datasets; ignored by Git
  scripts/                    # Environment, data acquisition/verification, tests
  studies/
    registry.toml             # Stable cross-domain paper registry
    atk-2022-deep-autoencoder/
      src/                    # Study 1 parsers, audits, and runners
      results/                # Machine-readable summaries; large arrays ignored
  reports/
    atk-2022-deep-autoencoder/# Planned standalone LaTeX report
    synthesis/                # Planned cross-paper LaTeX report
  .claude/rules/              # Claude automation; canonical facts stay shared
  AGENTS.md                   # Cross-agent operating contract
```

Every later paper receives a registered, self-contained study directory before
implementation begins.

## Evidence layers

1. **Source layer:** paper PDFs, official dataset metadata, raw files, hashes.
2. **Specification layer:** explicit paper statements, equations, target tables,
   omissions, contradictions, and ambiguity branches.
3. **Contract layer:** frozen reproduction tolerances, search space, seeds,
   partitions, statistics, and stopping rules.
4. **Implementation layer:** paper-literal code and tests; controlled variants
   live separately and cannot overwrite the literal path.
5. **Run layer:** configuration, environment, seed, scores, predictions, timing,
   and summary metrics for every attempted run.
6. **Assessment layer:** statistical comparison against the frozen contract.
7. **Report layer:** per-paper and cross-paper LaTeX sources and rendered PDFs.

## Key interfaces

### Dataset manifest

Each dataset record must identify the authoritative source, access status,
version, expected files, official checksum when available, local checksum, and
transform provenance. Code consumes verified local paths, never an unrecorded
substitute.

Acquisition scripts may download only openly available data or files for which
the invoking researcher already has authorization. Restricted datasets include
explicit access instructions and checksum verification, never embedded tokens.

### Reproduction contract

The Paper 1 contract will define a machine-readable experiment matrix plus a
human-readable rationale. Each row binds a paper claim to one explicit or
ambiguity-resolving implementation branch and its acceptance criteria.

### Run record

Every run must be reconstructable from: source revision, data hash, branch id,
hyperparameters, seed, split id, environment, raw scores/predictions, metrics,
duration, and completion/failure status. Failed runs remain part of the record.

### Verdict

A verdict compares the complete principal result pattern against the frozen
criteria over repeated runs. Static contradictions and proxy experiments are
reported separately and do not become reproduction verdicts.

## Data flow

1. Register paper and exact claimed results.
2. Acquire and checksum named datasets.
3. Translate paper text into explicit steps and ambiguity branches.
4. Freeze the reproduction contract before confirmatory search.
5. Test deterministic transformations and model interfaces.
6. Execute pilots, then confirmatory repeated runs without post-hoc selection.
7. Compute uncertainty and compare with the acceptance criteria.
8. Update the evidence ledger and generate the paper report.
9. Begin the next paper with a fresh independent contract.

## Key design decisions

- **Literal/controlled separation:** protects the primary question from being
  replaced by a better but different method.
- **Local raw inputs, versioned provenance:** avoids redistributing restricted
  or copyrighted material while maintaining auditability.
- **Predeclared finite search:** makes the non-reproduction hypothesis testable
  without claiming exhaustive proof over an infinite parameter space.
- **Per-paper isolation:** prevents conclusions and implementation assumptions
  from leaking from one target paper into another.

## How to extend

- Add a paper only after registering its PDF, claims, datasets, and independent
  reproduction contract.
- Add a methodological correction only under a clearly named controlled track.
- Add a report by committing LaTeX source and reproducible build instructions;
  generated build intermediates remain ignored.
