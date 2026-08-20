# ATK Evidence — Shared Agent Operational Guide

> Canonical bootstrap for Claude, Codex, and other assistants. `CLAUDE.md`
> imports this file; `.claude/rules/` automates Claude-specific rituals. No
> project fact may exist only in an agent-specific file or chat transcript.

## What this project is

This is a rigorous paper-by-paper audit of three distinct questions: whether
published numerical results follow from the methods as written, whether the
experiments identify the mechanism claimed to explain those results, and
whether the reported targets lie inside a credible empirical performance
envelope. See [`docs/VISION.md`](docs/VISION.md) for the full thesis.

## Canonical implementation tutorial

Read [`RUNBOOK.md`](RUNBOOK.md) first. It is the single end-to-end operational
guide for auditing any paper: end-to-end reading, discovery sandbox, diagnostic
breadth, source freeze, exact data, the genuine five-file reproduction,
mechanism tests, attainability analysis, confirmation, three findings,
reporting, and publication.

If another document conflicts with its execution order, the paper remains the
scientific authority and `RUNBOOK.md` remains the workflow authority.

The durable reasoning behind the current project reframe is
[`docs/decisions/2026-08-09-paper-first-minimal-instrument.md`](docs/decisions/2026-08-09-paper-first-minimal-instrument.md).
Its governing rule is: read the paper more carefully than we write the code,
then build the smallest transparent instrument that can execute the frozen
reading. Do not turn a paper reproduction into a platform.

The reporting contract for source wording that cannot execute is
[`docs/decisions/2026-08-11-non-executable-source-ladder.md`](docs/decisions/2026-08-11-non-executable-source-ladder.md).

The project-wide evidence frame is
[`docs/decisions/2026-08-20-three-part-evidence-frame.md`](docs/decisions/2026-08-20-three-part-evidence-frame.md).
It separates numerical reproduction, mechanism identification, and
attainability; defines the discovery-sandbox boundary; and requires cheap
diagnostic breadth before execution depth.

## Non-negotiable scientific mandate

1. **Read the complete paper before building a theory about it.** The first
   end-to-end pass identifies numerical targets and writes every explanatory
   claim as `B > A because Z exploits S`.
2. **Use a discovery sandbox before a formal program.** Author code, when it
   exists, is an artifact rather than the authority. With or without it, use
   the smallest disposable script or notebook to find capability-discriminating
   questions. Sandbox outputs are exploratory and never eligible reproduction
   or confirmation.
3. **Paper extraction dominates formal reasoning.** After discovery, return to
   the complete PDF, visually inspect it, and freeze a source-located executable
   specification and causal-claim map before eligible scientific model code.
   Passing tests or prior contracts never compensate for a wrong reading.
4. **Paper-literal first for numerical reproduction.** The primary numerical
   track implements only the named data, algorithms, preprocessing, models, and
   evaluation explicitly described by the paper.
5. **No silent repairs or extras.** Corrected splits, parameter matching,
   alternative scores, mechanism ablations, and improved methods remain visibly
   separate from paper-consistent numerical evidence. A control can answer a
   mechanism question without becoming reproduction.
6. **Branch ambiguities.** Every material omission or contradiction becomes one
   or more documented reasonable interpretations. If the printed operation
   cannot exist or execute, preserve that literal failure, predeclare the
   smallest reasonable executable repairs, run every materially distinct
   repair, and show each result beside the reported target on the site and in
   the paper report. Never pick an interpretation after seeing that it produces
   a favorable result or relabel a repair as the literal method.
7. **Use the minimal scientific instrument.** The five direct reproduction
   files are the active implementation. Shared code performs mechanical work
   only; paper meaning remains explicit. Hypothetical reuse never justifies new
   infrastructure before eligible results.
8. **Establish the triviality floor.** Run zero-parameter and simple fair rules
   through the identical evaluation path before attributing value to an
   elaborate architecture.
9. **Breadth means cheap questions, not expensive cells.** Run many small,
   discriminating checks and interpret them before full data, long training,
   repeated seeds, or branch expansion. One costly run per model family is
   already depth.
10. **Earn three findings separately.** Numerical non-reproduction does not
    establish mechanism failure or unattainability. Mechanism tests must isolate
    `S`, `Z`, and fair `A/B` comparisons. Attainability requires a declared
    empirical envelope and stopping rule.
11. **Freeze before confirmatory depth.** Record the evidence question,
    competing predictions, eligible implementations, targets, tolerances,
    hyperparameter envelope, statistical units, seeds, partitions, uncertainty
    method, compute budget, promotion rule, and stopping rule.
12. **No cherry-picking.** Preserve all runs. Report distributions and failures,
   not only the best seed or one matching metric.
13. **Be open to falsification.** A stable reproduction, a mechanism-confirming
    ablation, or an attainable target is valid and must be reported plainly.
14. **Bound conclusions.** State what was tested within the finite declared
    space. Empirical saturation is not structural impossibility. Do not infer
    intent, unpublished code, or undocumented history.
15. **Independent paper findings.** Do not generalize a result from one paper to
   another before independently testing the latter.

## Reading order every session

1. [`RUNBOOK.md`](RUNBOOK.md) — how to execute a paper end to end.
2. [`docs/decisions/2026-08-20-three-part-evidence-frame.md`](docs/decisions/2026-08-20-three-part-evidence-frame.md)
   — the three questions, sandbox, and breadth/depth boundary.
3. [`docs/STATUS.md`](docs/STATUS.md) — current state and next action.
4. [`docs/CONTEXT.md`](docs/CONTEXT.md) — compact working memory and don't-repeats.
5. [`docs/EVIDENCE-AND-LEARNINGS.md`](docs/EVIDENCE-AND-LEARNINGS.md) when
   interpreting results, disputed claims, or corrected conclusions.
6. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — project and evidence flow.
7. [`docs/VISION.md`](docs/VISION.md) when scope or intent is relevant.
8. The active plan, currently
   [`docs/plans/2026-08-09-paper-1-minimal-finish.md`](docs/plans/2026-08-09-paper-1-minimal-finish.md).

If the current step has no plan, create one before experimental implementation.

## Evidence discipline

- Use **VERIFIED**, **OBSERVED**, **INFERRED**, **HYPOTHESIS**, **INVALIDATED**,
  and **OPEN** consistently.
- Separate paper statements, static audits, exploratory experiments, and
  confirmatory experiments; none substitutes for another.
- Tag every scientific analysis as numerical (`N`), mechanism (`M`), or
  attainability (`A`) when one of those questions applies. Tag implementation
  semantics separately as `P`, `I`, `C`, or exploratory `X`; the two
  classifications are orthogonal.
- A sandbox result can motivate a formal question but cannot retrospectively
  supply its preregistration, choose a favorable interpretation, or become an
  eligible paper result.
- Preserve causal corrections: former belief → evidence → root cause → current
  conclusion → confidence and uncertainty.
- `STATUS.md` is current state, `CONTEXT.md` is prunable active memory, and
  `EVIDENCE-AND-LEARNINGS.md` is the durable causal record.
- Raw datasets and PDFs remain local. Commit provenance, checksums, code,
  configurations, summary results, and report sources.

## Session ritual

1. Read `RUNBOOK.md`, then the relevant documents above.
2. Identify the current phase and evidence question from STATUS.
3. Continue the active plan only if it names the discriminating question and
   remains consistent with the three-part frame; otherwise stop and draft a
   correction.
4. Do not cross a `CHECKPOINT` without user approval.
5. Update CONTEXT inline when a non-obvious fact or explicit user emphasis appears.
6. Update the evidence record when a result changes a belief.
7. Before an adaptive `X` probe, record its question and minimal setup. Before
   any eligible formal run, name the paper/table cell or causal claim,
   `N`/`M`/`A` question, `P`/`I`/`C` track, seed, competing predictions, exact
   question, and report finding it feeds.
8. Before any expensive run, show which cheap diagnostic promoted it and what
   uncertainty remains after that diagnostic.
9. Stop if implementation or documentation is growing while no explanation is
   being discriminated or eligible result is advancing.

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
| What are the three findings and sandbox boundary? | `docs/decisions/2026-08-20-three-part-evidence-frame.md` |
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
