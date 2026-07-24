# ATK Evidence — Architecture

**Last updated:** 2026-07-24

## Overview

The repository is a domain-neutral evidence pipeline organized around
independent target papers. Local source PDFs and raw datasets feed versioned
paper specifications, literal implementations, immutable run outputs,
statistical assessments, and per-paper LaTeX reports. Charter preserves intent,
state, and causal evidence across Claude and Codex sessions.

## Structure

```text
atk-evidence/
  RUNBOOK.md                 # Canonical end-to-end paper implementation guide
  docs/                       # Charter vision, status, memory, evidence, plans
  papers/                     # Local source PDFs; ignored by Git
  data/                       # Local raw/derived datasets; ignored by Git
  scripts/                    # Environment, data acquisition/verification, tests
  studies/
    registry.toml             # Stable cross-domain paper registry
    atk-2022-deep-autoencoder/
      METHOD.md               # Fresh PDF-derived executable specification
      reproduction/           # Primary five-file scientific implementation
        download_data.py
        prepare_data.py
        models.py
        run_experiment.py
        analyze_results.py
      download_data.py        # Historical forensic command wrapper
      prepare_data.py         # Historical forensic command wrapper
      run_experiment.py       # Historical forensic command wrapper
      analyze_results.py      # Historical forensic command wrapper
      src/                    # Study 1 parsers, audits, and runners
      results/                # Machine-readable summaries; large arrays ignored
  reports/
    atk-2022-deep-autoencoder/# Standalone LaTeX report source
    synthesis/                # Planned cross-paper LaTeX report
  site/
    index.html                # Multi-paper public landing page
    papers/<study-id>/        # Self-contained readable paper maps
  .claude/rules/              # Claude automation; canonical facts stay shared
  AGENTS.md                   # Cross-agent operating contract
```

Every later paper receives a registered, self-contained study directory before
implementation begins.

The four current study-root programs are a small command surface over `src/`.
They are not, by themselves, a compact reference implementation: the Paper 1
`src/` tree contains a large forensic branch engine, evidence verifier, tests,
and the cluster adapter.

The target public architecture separates two code products:

1. a real five-file reference track (`download`, `prepare`, `models`, `run`,
   `analyze`) for one frozen source-faithful anchor; and
2. the larger forensic harness for ambiguity coverage, corrected controls,
   cluster execution, and evidence verification.

The extraction is the active Paper 1 plan and must preserve existing
fingerprints and result eligibility. Small wrappers over the forensic harness
do not satisfy it.

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

Each paper begins with a concise source-located `METHOD.md` and one declared
straight-through experiment. Before confirmatory runs, its active plan freezes
targets, eligible material interpretations, tolerances, seeds, statistics,
budget, and stopping rules. A machine-readable branch matrix is optional later
support, not a prerequisite for the first full anchor.

### Run record

Every run must be reconstructable from: source revision, data hash, branch id,
hyperparameters, seed, split id, environment, raw scores/predictions, metrics,
duration, and completion/failure status. Failed runs remain part of the record.

Historical the cluster/DDP attempts retain their existing detailed fingerprints
inside the forensic layer. New primary runs use the same five-file scientific
path locally and on the cluster; a short Slurm wrapper may select resources but
must not introduce a second scientific implementation.

### Verdict

A verdict compares the complete principal result pattern against the frozen
criteria over repeated runs. Static contradictions and proxy experiments are
reported separately and do not become reproduction verdicts.

## Data flow

1. Register paper and exact claimed results.
2. Read the complete PDF and freeze a source-located `METHOD.md`.
3. Acquire and checksum named datasets.
4. Implement the straight-through paper reading in five real scientific files.
5. Run tiny deterministic and one-step checks.
6. Obtain one eligible full anchor before generalized infrastructure.
7. Complete the reported tables and repeated exploratory seeds.
8. Test material ambiguity branches and separate corrected controls.
9. Freeze the reproduction contract before confirmatory search.
10. Execute confirmatory runs without post-hoc selection.
11. Compute uncertainty and compare with the acceptance criteria.
12. Update the evidence ledger and generate the paper report.
13. Freeze the paper-level reproduction report and verdict.
14. If authorized, execute the separately contracted controlled solution and
    publish it as an addendum that cannot alter the reproduction verdict.
15. Begin the next paper with a fresh independent contract.

The public `site/` is a separate static publication surface. It contains no raw
data, paper PDFs, credentials, or internal build state. GitHub Pages deploys
only this directory; rendered report PDFs are copied into it only after their
paper-level evidence is frozen.

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
