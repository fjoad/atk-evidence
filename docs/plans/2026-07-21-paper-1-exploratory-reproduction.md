---
branch: codex/paper1-exploratory-reproduction
---

# Paper 1 Tables I-V — Exploratory Paper-Literal Reproduction Plan

**Goal:** Implement the authors' described pipeline as faithfully as the paper
permits, execute every table whose exact named data are available, preserve all
runs and wall-clock timings, and report our Tables I-V beside the published
targets without selecting favorable seeds.

**Status:** In progress

**Authorization:** The user explicitly requested end-to-end implementation and
execution on 2026-07-21. This satisfies the earlier contract checkpoint for an
**exploratory** run. It does not convert the results into a preregistered
confirmatory verdict; the present choices were made after static audits and an
invalidated proxy experiment already existed.

## Scientific boundary

- Primary implementation obeys questionable paper procedures when executable:
  pre-split z-scoring, paper-positioned ADASYN, published architectures,
  output activations, optimizers, thresholds, and 2:1 split.
- Missing details receive pre-outcome author-intent assumptions recorded in
  `AMBIGUITY_REGISTER.md` and the TOML run contract.
- No non-identical consumption archive or undeclared allocation substitute may
  cross the exact-data gate. The six ScienceDB archives pass the official
  MD5s; the CSV allocation crosses only the separately declared semantic branch.
- All completed and failed runs, individual seeds, raw scores, predictions,
  metrics, timing, data hashes, and environment versions are preserved.

## Tasks

### 1. Freeze sources and targets

- [x] Recover and reread the human conversation from the session JSONL.
- [x] Read and visually inspect all 12 PDF pages.
- [x] Transcribe Tables I-V into machine-readable CSV files.
- [x] Record the paper-literal workflow and ambiguity branches.

### 2. Repair the exact-data gate

- [x] Add the official CER allocation file to acquisition and verification.
- [x] Extend the CER parser for chunked archives, allocation filtering, DST,
  complete 48-slot days, and customer-disjoint partitions.
- [x] Download and verify all six exact ScienceDB archives against official MD5s.
- [x] Validate the ScienceDB allocation CSV by row-level cross-copy comparison
  and full reading-ID coverage; approve it as the named exploratory semantic
  branch in a decision record.
- [x] Implement that explicit branch in the data gate, filtering the declared
  residential IDs before validating irrelevant malformed `other` rows.
- [x] Prepare Tables III-V inputs only after the six official archive MD5s and
  the approved allocation semantic digest pass.
- [x] Implement the small dataset/table-specific immutable runner over the
  verified ISET cache; do not add a second scheduler/orchestration layer.

### 3. Implement the paper-width pipeline

- [x] Implement Keras FC-SAE, LSTM-SAE, FC-VAE, LSTM-VAE, and LSTM-AEA.
- [x] Implement Naive Bayes, ARIMA(1,1,0), one-class SVM, feed-forward,
  supervised LSTM, and multiclass SVM benchmark branches.
- [x] Implement paper metrics, fixed thresholds, ADASYN placement, timing,
  immutable run records, and aggregate tables.
- [x] Add deterministic preprocessing/model/metric tests and finite weight-update tests.

### 4. Sanity gates

- [x] Verify data hashes, split cardinalities, and customer non-overlap.
- [x] Verify model shapes, parameter counts, output ranges, finite updates,
  VAE score directions, attention normalization, and Table V FA invariance.
- [x] Time short training probes and record per-architecture rough ETAs before full execution.

### 5. Execute and assess

- [ ] Run Table II on verified SGCC for all models and all declared seeds.
  Current: 20/33 successful cells, three FC-VAE failures, ten unrun.
- [ ] Run Tables III and V and the Table IV size/timing matrix from the
  checksum-verified exact-ISET cache. The execution adapter and bounded smoke
  path are complete; the cluster cells remain.
- [ ] Generate paper-versus-reproduction Tables I-V with means, dispersion,
  individual runs, timing, and explicit unavailable cells.
- [ ] Update STATUS, CONTEXT, evidence ledger, study log, and README.
- [ ] Run the full test suite, commit, push, and leave the tree clean.

### 6. Current sanity finding

- [x] Verify the first completed LSTM-SAE seed at score level, including an
  oracle-threshold diagnostic, original-only diagnostic, cross-model score
  comparison, and reconstruction-domain check.
- [x] Reduce normal user operation to four study-root commands:
  download, prepare, run, and analyze.

### 7. Deferred post-verdict handoff

- [ ] After Tables I--V, confirmatory assessment, and the Paper 1 LaTeX verdict
  are complete and frozen, request user approval before beginning
  `2026-07-23-paper-1-controlled-solution.md`.
- [ ] Keep every corrected/new method and “beat the paper” comparison outside
  this paper-literal plan.

## Stopping rules

- Default neural budget: three fixed seeds, maximum 30 epochs, convergence
  patience 4 with minimum loss change `1e-4` after a five-epoch warm-up.
- The preflight may reduce dataset fraction or skip a branch only for a recorded
  resource failure; it may not use outcome quality as the reason.
- Failed numerical convergence is a result. No replacement seed is allowed.
- Exact ISET execution stops at the MD5 gate if access is absent.
