# Electricity-Theft Paper Reproducibility Audit — Shared Agent Operational Guide

> Canonical bootstrap for Claude, Codex, and other assistants. `CLAUDE.md`
> imports this file; `.claude/rules/` automates Claude-specific rituals. No
> project fact may exist only in an agent-specific file or chat transcript.

## What this project is

This is a rigorous, paper-by-paper audit of whether selected published
electricity-theft detection results can be reproduced from the methods as
written. The current target is Takiddin et al., "Deep Autoencoder-Based Anomaly
Detection of Electricity Theft Cyberattacks in Smart Grids." See
[`docs/VISION.md`](docs/VISION.md) for the full thesis.

## Non-negotiable scientific mandate

1. **Paper-literal first.** The primary track implements only the named data,
   algorithms, preprocessing, models, and evaluation explicitly described by
   the paper.
2. **No silent repairs or extras.** Corrected splits, parameter matching,
   alternative anomaly scores, or other improvements belong only in a separate
   controlled-analysis track.
3. **Branch ambiguities.** Every material omission or contradiction becomes one
   or more documented reasonable interpretations. Never pick an interpretation
   after seeing that it produces a favorable result.
4. **Freeze before confirmatory runs.** Record numerical targets, tolerances,
   hyperparameter envelope, seeds, partitions, statistics, and stopping rules.
5. **No cherry-picking.** Preserve all runs. Report distributions and failures,
   not only the best seed or one matching metric.
6. **Be open to falsification.** A stable paper-consistent reproduction is a
   valid result and must be reported plainly.
7. **Bound conclusions.** State what was not reproduced within the tested,
   predeclared space. Do not infer intent or claim an infinite space was proven
   impossible.
8. **Independent paper verdicts.** Do not generalize a finding from one paper to
   another before independently testing the latter.

## Reading order every session

1. [`docs/STATUS.md`](docs/STATUS.md) — current state and next action.
2. [`docs/CONTEXT.md`](docs/CONTEXT.md) — compact working memory and don't-repeats.
3. [`docs/EVIDENCE-AND-LEARNINGS.md`](docs/EVIDENCE-AND-LEARNINGS.md) when
   interpreting results, disputed claims, or corrected conclusions.
4. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — project and evidence flow.
5. [`docs/VISION.md`](docs/VISION.md) when scope or intent is relevant.
6. The active plan under [`docs/plans/`](docs/plans/).

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

1. Read the documents above.
2. Identify the current step from STATUS.
3. Continue the active plan or draft one.
4. Do not cross a `CHECKPOINT` without user approval.
5. Update CONTEXT inline when a non-obvious fact or explicit user emphasis appears.
6. Update the evidence record when a result changes a belief.

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
| Where are we? | `docs/STATUS.md` |
| What must survive compaction? | `docs/CONTEXT.md` |
| Why did a conclusion change? | `docs/EVIDENCE-AND-LEARNINGS.md` |
| How do artifacts connect? | `docs/ARCHITECTURE.md` |
| What was deliberately decided? | `docs/decisions/` |
| What is the current execution contract? | `docs/plans/` |

## Project-specific commands and constraints

- Python environment: `replication/.venv`.
- Deterministic tests:
  `cd replication/src && ../.venv/bin/python -m unittest -v test_attacks.py test_cer_parser.py`
- Compile check: `replication/.venv/bin/python -m compileall -q replication/src`.
- Never modify raw downloaded files in place.
- Never commit `data/`, `papers/`, access tokens, or restricted archives.
- Do not resume the SGCC 48-day proxy as though it were Paper 1 reproduction.
- The exact CER data is a hard gate for the relevant paper-literal experiment.

## Decision records

Create `docs/decisions/YYYY-MM-DD-short-title.md` for a choice that changes the
evidence contract, architecture, dependency boundary, or interpretation policy.

