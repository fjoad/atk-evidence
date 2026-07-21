# Paper 1 Reproduction Contract — Implementation Plan

**Goal:** Freeze a complete, auditable, paper-literal experiment contract before
any confirmatory model search or result selection.

**Status:** Draft — requires user review

## Context

Initial data acquisition, static auditing, and exploratory code exist, but the
primary hypothesis has now been clarified: reproduce the paper exactly as
described before running controlled alternatives. This plan converts the paper
into a finite testable contract and prevents already-seen exploratory results
from influencing confirmatory branch or hyperparameter selection.

## Files

**Create:**

- `studies/atk-2022-deep-autoencoder/PAPER_LITERAL_CONTRACT.md` — frozen human-readable contract.
- `studies/atk-2022-deep-autoencoder/config/paper_literal_matrix.yaml` — executable experiment matrix.
- `studies/atk-2022-deep-autoencoder/REPORTED_TARGETS.csv` — transcription of every target number.
- `studies/atk-2022-deep-autoencoder/AMBIGUITY_REGISTER.md` — explicit statements, omissions,
  contradictions, branches, and rationale.
- `studies/atk-2022-deep-autoencoder/ACCEPTANCE_CRITERIA.md` — tolerances and statistical decision rule.

**Modify:**

- `studies/atk-2022-deep-autoencoder/README.md` — make the literal track primary and controlled track secondary.
- `studies/atk-2022-deep-autoencoder/EXPERIMENT_SPEC.md` — reorganize existing audit into contract inputs.
- `docs/STATUS.md` and `docs/EVIDENCE-AND-LEARNINGS.md` — record frozen state.

## Tasks

### Task 1: Register claims without interpretation

- [ ] Transcribe each principal table, threshold, timing claim, per-attack result,
  architecture, equation, split, and preprocessing statement from Paper 1.
- [ ] Double-check every transcription against rendered PDF pages.
- [ ] Assign stable claim IDs and record page/table/equation provenance.

### Task 2: Build the paper-literal algorithm

- [ ] Express the described workflow step-by-step without correcting it.
- [ ] Identify which steps are executable as written and which are missing,
  contradictory, or dimensionally inconsistent.
- [ ] Keep SGCC and CER paths separate; do not invent a shared 48-value mapping.

### Task 3: Register reasonable ambiguity branches

- [ ] For every blocking ambiguity, list the smallest paper-consistent options.
- [ ] Justify each option using only the paper, named dependencies, or standard
  implementation semantics contemporaneous with the paper.
- [ ] Exclude options that add a method or leak information not described.
- [ ] Freeze branch IDs before evaluating their numerical outcomes.

### Task 4: Freeze finite search and acceptance criteria

- [ ] Define the small hyperparameter grid only for unspecified training details.
- [ ] Define seeds, customer splits, run count, compute budget, and stopping rule.
- [ ] Define numerical tolerances and whether reproduction requires joint matching
  of DR, FA, AUC, remaining metrics, architecture ordering, and per-attack results.
- [ ] Define statistical tests, confidence intervals, multiplicity treatment, and
  what counts as strong supporting or disconfirming evidence.
- [ ] Ensure no best-seed-only or post-hoc branch selection can count as reproduction.

## CHECKPOINT: user review

Review the complete claims table, ambiguity register, search envelope, and
acceptance criteria. No confirmatory experiment starts until the user approves
and the frozen contract is committed.

### Task 5: Exact-data gate and implementation plan

- [ ] Obtain authorized CER archives and verify all official MD5 values.
- [ ] Validate parsers and sample counts against official documentation.
- [ ] Produce the code-level plan and deterministic tests for each frozen branch.
- [ ] Keep the earlier SGCC 48-day proxy outside the confirmatory matrix.

## Verification

- [ ] Every reported target has a PDF provenance location and two-pass check.
- [ ] Every non-explicit implementation choice maps to a frozen ambiguity branch.
- [ ] Search space and stopping rule are finite and recorded before runs.
- [ ] Acceptance criteria can both confirm and reject reproduction.
- [ ] No confirmatory command is runnable without verified exact data.
- [ ] Tests pass; STATUS, CONTEXT, and evidence ledger are current; contract committed.
