# ATK Evidence — Shared Agent Operational Guide

> Canonical bootstrap for Claude, Codex, and other assistants. `CLAUDE.md`
> imports this file; `.claude/rules/` automates Claude-specific rituals. No
> project fact may exist only in an agent-specific file or chat transcript.

## What this project is

This is a rigorous, extensible, paper-by-paper audit of whether selected
published numerical results can be reproduced from the methods as written. The
initial corpus is by Abdulrahman Takiddin and coauthors; the current target is
"Deep Autoencoder-Based Anomaly Detection of Electricity Theft Cyberattacks in
Smart Grids." See
[`docs/VISION.md`](docs/VISION.md) for the full thesis.

## Canonical implementation tutorial

Read [`RUNBOOK.md`](RUNBOOK.md) first. It is the single end-to-end operational
guide for implementing any paper: PDF reconstruction, exact data, the genuine
five-file reproduction, sanity checks, full experiments, ambiguity branches,
confirmation, reporting, and publication.

If another document conflicts with its execution order, the paper remains the
scientific authority and `RUNBOOK.md` remains the workflow authority.

The durable reasoning behind the current project reframe is
[`docs/decisions/2026-08-09-paper-first-minimal-instrument.md`](docs/decisions/2026-08-09-paper-first-minimal-instrument.md).
Its governing rule is: read the paper more carefully than we write the code,
then build the smallest transparent instrument that can execute the frozen
reading. Do not turn a paper reproduction into a platform.

The reporting contract for source wording that cannot execute is
[`docs/decisions/2026-08-11-non-executable-source-ladder.md`](docs/decisions/2026-08-11-non-executable-source-ladder.md).

## Non-negotiable scientific mandate

1. **Paper extraction dominates reasoning.** Read and visually inspect the
   complete PDF and freeze a source-located executable specification before
   scientific model code. Passing tests or prior contracts never compensate
   for a wrong reading.
2. **Paper-literal first.** The primary track implements only the named data,
   algorithms, preprocessing, models, and evaluation explicitly described by
   the paper.
3. **No silent repairs or extras.** Corrected splits, parameter matching,
   alternative anomaly scores, or other improvements belong only in a separate
   controlled-analysis track.
4. **Branch ambiguities.** Every material omission or contradiction becomes one
   or more documented reasonable interpretations. If the printed operation
   cannot exist or execute, preserve that literal failure, predeclare the
   smallest reasonable executable repairs, run every materially distinct
   repair, and show each result beside the reported target on the site and in
   the paper report. Never pick an interpretation after seeing that it produces
   a favorable result or relabel a repair as the literal method.
5. **Use the minimal scientific instrument.** The five direct reproduction
   files are the active implementation. Shared code performs mechanical work
   only; paper meaning remains explicit. Hypothetical reuse never justifies new
   infrastructure before eligible results.
6. **Freeze before confirmatory runs.** Record numerical targets, tolerances,
   hyperparameter envelope, seeds, partitions, statistics, and stopping rules.
7. **No cherry-picking.** Preserve all runs. Report distributions and failures,
   not only the best seed or one matching metric.
8. **Be open to falsification.** A stable paper-consistent reproduction is a
   valid result and must be reported plainly.
9. **Bound conclusions.** State what was not reproduced within the tested,
   predeclared space. Do not infer intent or claim an infinite space was proven
   impossible.
10. **Independent paper verdicts.** Do not generalize a finding from one paper to
   another before independently testing the latter.

## Reading order every session

1. [`RUNBOOK.md`](RUNBOOK.md) — how to execute a paper end to end.
2. [`docs/STATUS.md`](docs/STATUS.md) — current state and next action.
3. [`docs/CONTEXT.md`](docs/CONTEXT.md) — compact working memory and don't-repeats.
4. [`docs/EVIDENCE-AND-LEARNINGS.md`](docs/EVIDENCE-AND-LEARNINGS.md) when
   interpreting results, disputed claims, or corrected conclusions.
5. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — project and evidence flow.
6. [`docs/VISION.md`](docs/VISION.md) when scope or intent is relevant.
7. The active plan, currently
   [`docs/plans/2026-08-09-paper-1-minimal-finish.md`](docs/plans/2026-08-09-paper-1-minimal-finish.md).

If the current step has no plan, create one before experimental implementation.

## Evidence discipline

- Use **VERIFIED**, **OBSERVED**, **INFERRED**, **HYPOTHESIS**, **INVALIDATED**,
  and **OPEN** consistently.
- Separate paper statements, static audits, exploratory experiments, and
  confirmatory experiments; none substitutes for another.
- Preserve causal corrections: former belief → evidence → root cause → current
  conclusion → confidence and uncertainty.
- `STATUS.md` is current state, `CONTEXT.md` is prunable active memory, and
  `EVIDENCE-AND-LEARNINGS.md` is the durable causal record.
- Raw datasets and PDFs remain local. Commit provenance, checksums, code,
  configurations, summary results, and report sources.

## Session ritual

1. Read `RUNBOOK.md`, then the relevant documents above.
2. Identify the current step from STATUS.
3. Continue the active plan or draft one.
4. Do not cross a `CHECKPOINT` without user approval.
5. Update CONTEXT inline when a non-obvious fact or explicit user emphasis appears.
6. Update the evidence record when a result changes a belief.
7. Before any run, name the paper/table cell, interpretation, seed, exact
   question, and report result it feeds.
8. Stop if implementation or documentation is growing while no eligible result
   is advancing.

## Charter rituals

| Tier | Signal | Ritual |
|---|---|---|
| Trivial | Typo or one-line metadata | Execute and verify |
| Small | Bounded, well-understood change | State plan, execute, verify |
| Medium | Multi-file implementation or experiment | Written plan, tests, verification, finish |
| Major | New paper pipeline or evidentiary design | Written plan, user checkpoint, implementation, verification, finish |

For medium/major work, run tests, update STATUS and relevant docs, commit the
work, and report tests, docs, commits, and the next step.

## Key documents

| Question | Read |
|---|---|
| What and why? | `docs/VISION.md` |
| How do I execute a paper? | `RUNBOOK.md` |
| Where are we? | `docs/STATUS.md` |
| What must survive compaction? | `docs/CONTEXT.md` |
| Why did a conclusion change? | `docs/EVIDENCE-AND-LEARNINGS.md` |
| How do artifacts connect? | `docs/ARCHITECTURE.md` |
| What was deliberately decided? | `docs/decisions/` |
| What is the current execution contract? | `docs/plans/` |

## Project-specific commands and constraints

- Python environment: root `.venv` created by `bash scripts/bootstrap.sh`.
- Deterministic tests: `bash scripts/test.sh`.
- Data verification: `.venv/bin/python scripts/verify_data.py --strict`.
- Never modify raw downloaded files in place.
- Never commit `data/`, `papers/`, access tokens, or restricted archives.
- Do not resume the SGCC 48-day proxy as though it were Paper 1 reproduction.
- The exact CER data is a hard gate for the relevant paper-literal experiment.
- Each paper belongs under `studies/<study-id>/`; register it in
  `studies/registry.toml` before implementation.

## Decision records

Create `docs/decisions/YYYY-MM-DD-short-title.md` for a choice that changes the
evidence contract, architecture, dependency boundary, or interpretation policy.
