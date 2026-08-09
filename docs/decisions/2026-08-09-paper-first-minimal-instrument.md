# Decision: paper-first minimal scientific instrument

**Date:** 2026-08-09

**Status:** Accepted

**Scope:** Entire project; Paper 1 is the active application

## Why this decision exists

The project repeatedly turned a small scientific question into a large software
system. Paper 1 accumulated a 21,000-line forensic harness, branch machinery,
cluster tooling, and publication work before a complete eligible result existed.
Several expensive runs were later quarantined because the paper had not first
been reconstructed accurately enough: architectures, sample populations, attack
construction, and anomaly scores differed from what the PDF described.

The recurring problem was not insufficient engineering effort. It was misplaced
complexity. A generalized system was built before one transparent experiment had
proved that the scientific interpretation and measurement path were correct.

## Core principle

> Read the paper more carefully than we write the code. Then build the smallest
> transparent scientific instrument that can execute that frozen reading and
> answer one named question.

Paper reconstruction is the reasoning-intensive part of the work. It requires
complete reading, visual inspection, source locators, reconciliation of prose,
equations, algorithms, figures, and tables, and explicit treatment of omissions
and contradictions. Implementation should then be mostly direct transcription.

A perfectly engineered implementation of a misread paper is invalid evidence.
A small implementation whose relationship to the paper is obvious is preferable
to a reusable framework whose scientific meaning is hidden behind abstractions.

## What the project is

ATK Evidence is a focused paper-by-paper scientific audit. It is not:

- a production machine-learning platform;
- a general workflow or experiment service;
- an exhaustive program generator;
- a scheduler or cluster-management product; or
- a publication system whose infrastructure precedes its evidence.

The repository may contain historical forensic machinery, but that machinery is
not the active scientific implementation and is never an authority over the PDF.

## The minimal instrument

Each paper's active reproduction consists of five direct files:

```text
download_data.py
prepare_data.py
models.py
run_experiment.py
analyze_results.py
```

The global/shared layer may perform only mechanical work whose meaning is stable:

- checksums and file verification;
- ordinary parsing;
- standard metric calculations;
- result serialization; and
- a short cluster resource wrapper.

Paper-specific meaning remains explicit inside the study: preparation order,
attack equations, model architecture, loss, score, threshold, split, table
identity, and every interpretive choice.

Duplication is acceptable. Do not introduce a shared abstraction merely because
a future paper might need it. Extract common code only after repeated completed
papers demonstrate the same stable mechanical need.

## Source-freeze contract

Before scientific model code is written or changed, `METHOD.md` must let an
independent reader reconstruct the experiment without consulting old code. It
must contain:

1. every reported target number and table;
2. exact data, populations, sample units, and access identity;
3. preparation operations in printed order;
4. every model layer, activation, loss, optimizer, and anomaly score stated;
5. training and evaluation procedures, including missing settings;
6. timing claims and their reported measurement context;
7. one source locator for every material instruction;
8. explicit `EXACT`, `AMBIGUOUS`, `CONTRADICTORY`, or `NON-EXECUTABLE` status;
9. one frozen straight-through reading; and
10. the finite set of material alternative readings.

Tests, code comments, prior contracts, precursor papers, and field conventions do
not establish fidelity. They may help explain a branch only after the PDF-derived
specification is frozen.

## Step 0: prove the measuring instrument

Before a long run, the actual execution path must cheaply demonstrate:

- raw-data identity and expected counts;
- hand-checkable preparation and attack transformations;
- split identities and any intended or printed leakage;
- runtime layer inventory and parameter count;
- finite forward pass, gradient, and optimizer update;
- anomaly-score definition and direction;
- threshold behavior;
- metric formulas against constructed confusion matrices;
- result persistence and reload; and
- one tiny end-to-end execution through the same five files.

These checks establish shared capability. They do not prove every full experiment
will work. One watched full anchor remains required before parallel scaling.

## Run rule

No experimental run launches unless its record names:

- paper and table cell;
- model and dataset;
- `P`, `I`, or `C` interpretation;
- seed and configuration;
- exact scientific question; and
- report result the run will feed.

No new code is added unless it is required to obtain or validate the next named
scientific result. A run failure caused by data, environment, scoring, or metric
machinery is an execution/evaluation failure, not evidence that the paper's model
failed.

## Execution order

1. Freeze the complete paper interpretation.
2. Acquire and verify the exact named data.
3. Implement the five-file route directly.
4. Pass Step 0.
5. Run and inspect one full anchor.
6. Run the remaining predeclared seeds in parallel.
7. Add models one at a time and repeat the same gate.
8. Complete the paper's reported tables.
9. Vary material ambiguities one at a time; combine them only for justified
   interactions.
10. Run corrected controls separately.
11. Freeze and execute confirmatory experiments.
12. Write the verdict and publication artifacts.
13. Begin the next paper only after the current paper's bounded verdict.

The historical 921-configuration Paper 1 inventory is a later coverage checklist,
not an execution plan. It must not be run as a matrix.

## Compute and parallelism

Every experimental preparation, training, and scoring run executes on the
cluster. Local work is limited to source reconstruction, code, documentation,
lightweight inspection, and transfer/monitoring.

Parallelism follows evidence, not optimism:

1. one cheap capability gate;
2. one watched full anchor;
3. a measured one-epoch resource/throughput probe when resource choice matters;
4. independent seeds or models in parallel after the path is trusted.

Multiple GPUs are used only when measured throughput justifies distributed
training. Otherwise, allocating one independent seed per GPU is simpler and
usually more useful.

## Documentation and recovery

The shared Charter documents are the project memory:

- `RUNBOOK.md` governs how work is performed;
- `docs/STATUS.md` contains current truth and one next action;
- the active plan contains the finite execution contract;
- `METHOD.md` contains the paper interpretation;
- `docs/EVIDENCE-AND-LEARNINGS.md` preserves causal corrections; and
- decision records preserve choices that must survive compaction.

Human-facing documentation explains the same system in ordinary language. No
scientific fact or decision may exist only in a chat, agent-specific file, or
historical status snapshot. Avoid duplicating volatile state across documents.

## Drift checks

Stop and return to this decision when:

- implementation starts before the paper can be reconstructed from `METHOD.md`;
- the codebase grows while no new eligible result exists;
- infrastructure is justified by hypothetical future needs;
- a generalized abstraction hides a paper-specific operation;
- documentation or publication displaces experiments;
- a large sweep is proposed before one watched full anchor;
- passing unit tests are treated as proof of end-to-end validity; or
- most elapsed project time is agent engineering rather than experimental
  execution and analysis.

## Working agreement

At each phase boundary, discuss and freeze the scope, question, cost, and finish
condition. Then execute the whole agreed phase without asking for every minor
implementation choice. Stop only when evidence invalidates the plan, cost changes
materially, new authority is required, or the measuring instrument fails its
gate.

Progress updates remain short: what is running, where, what question it answers,
current result, ETA, and exact next action.
