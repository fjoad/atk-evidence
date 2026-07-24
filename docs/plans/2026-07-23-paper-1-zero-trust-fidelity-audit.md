# Paper 1 Zero-Trust Fidelity Audit

**Created:** 2026-07-23  
**Status:** Gate C complete; bounded local Gate D passed; exact-cache/GPU gates remain
**Source of truth:** The exact 12-page Paper 1 PDF, SHA-256
`f3098e0c27ee19b27bea026aedc3d10e5dbb0c46f5cd01ed5bd5c05b7dcf850f`

## Why this plan exists

The earlier implementation audit concluded that the code matched the paper or
a registered ambiguity branch. A fresh source-first pass found concrete
counterexamples to that conclusion. Existing contracts, notes, tests, and
results are therefore evidence to audit, not authorities for what the paper
says.

The purpose of this pass is to prevent a plausible but different experiment
from being mistaken for a reconstruction of the published experiment.

## Immediate containment

1. Submit no new Paper 1 jobs.
2. Do not cancel or mutate already-running jobs merely because this audit
   started, but quarantine their outputs on arrival.
3. Retain every existing artifact with its original fingerprints.
4. Do not interpret implementation-v1 metric gaps as reproduction evidence
   until the corresponding row passes the gates below.
5. Replace the former blanket `VERIFIED` fidelity status with a causal
   correction in the durable evidence record.

## Required artifacts

1. A visually checked PDF fingerprint and page inventory.
2. A source-only experimental specification with a page, section, equation,
   figure, algorithm, or table locator for every material claim.
3. A claim-to-code traceability matrix covering:
   - source data and sample unit;
   - attack population and equations;
   - split identity and order;
   - normalization axis, fit population, and order;
   - validation construction;
   - ADASYN location and parameters;
   - every model layer, width, activation, optimizer, and loss;
   - anomaly-score definition and orientation;
   - threshold derivation and numerical threshold;
   - supervised labels, losses, and decisions;
   - benchmark completion assumptions;
   - Table IV sizing/timing boundary;
   - Table V training and test independence.
4. An eligibility map saying which old results remain usable and for what
   claim.
5. Executable structural tests for every unambiguous item before rerunning
   compute.

## Classification rules

- `EXACT`: directly specified and implemented without an added choice.
- `AMBIGUOUS-BRANCH`: the paper omits a material detail and the implementation
  is one predeclared reasonable branch.
- `PAPER-CONTRADICTION`: two paper statements cannot both be satisfied.
- `MISMATCH`: the implementation contradicts an executable paper statement.
- `ADDED`: the implementation introduces an unstated procedure.
- `NOT-IMPLEMENTED`: a stated paper operation has no implementation.
- `NON-EXECUTABLE`: the printed procedure is internally invalid or lacks the
  information needed to run.

Registering a choice does not convert it to `EXACT`. A result from an
`AMBIGUOUS-BRANCH` supports only that branch. A result affected by `MISMATCH`
or `NOT-IMPLEMENTED` is not eligible as reproduction evidence.

## Gates

### Gate A — source reconstruction

- Render and visually inspect all 12 pages.
- Reconstruct the method from the PDF without consulting the current contract.
- Record contradictions and missing information without resolving them.

**Status:** Complete.

### Gate B — code crosswalk

- Map every source claim to concrete file/function/layer behavior.
- Count actual model layers rather than accepting configuration labels.
- Inspect generated cache metadata, not only preparation source.
- Assign one classification and blast radius to every row.

**Status:** Complete. Findings and current eligibility are recorded in
`studies/atk-2022-deep-autoencoder/PAPER_TO_CODE_TRACEABILITY.md`.

### CHECKPOINT — correction contract

Before changing experimental semantics or launching replacement runs:

- present all `MISMATCH`, `NOT-IMPLEMENTED`, and `PAPER-CONTRADICTION` rows;
- agree which contradiction branches are required;
- freeze cache schema and run-eligibility rules;
- obtain user approval.

**Status:** Approved in principle on 2026-07-23 with an expanded requirement:
execute the printed method, every materially defensible interpretation of the
text, and a separately labeled scientifically corrected control. The branch
coverage must be exhaustive with respect to the PDF and a frozen finite
hyperparameter envelope. See `BRANCH_COVERAGE_CONTRACT.md` and decision
`2026-07-23-exhaustive-interpretations-and-corrected-controls.md`.

### Gate C — branch implementation and structural verification

- Machine-readable coverage closure is complete: 921 compatible paper-consistent
  configurations, 22 corrected controls, 36/36 registered ambiguity rows
  mapped, and frozen screening/promotion budgets. This is the implementation
  input, not evidence that the branches themselves exist yet.
- Implement the complete frozen paper-consistent branch lattice without
  selecting one favored repair.
- Implement the corrected-control track in a separate namespace and result
  family.
- Add tests that fail if paper-specified layer counts, widths, attack
  populations, split identities, or score definitions drift.
- Rebuild any cache whose source population or transformation changes.
- Keep old artifacts under their original fingerprints.

**Progress:** The opt-in `paper_source_v2` builders and four source-derived
runtime tests now cover printed hidden-layer mirrors, latent width/placement,
and dropout placement without changing implementation-v1 defaults. The SGCC
and ISET preparers now also execute joint, per-class, per-profile, and
training-benign scaling; paper-positioned versus absent anomaly-test ADASYN;
pre-split versus training-only supervised ADASYN; and ISET all-customer versus
B2-only attack populations. Tiny fixture tests verify untouched corrected test
sets, original-ID/meter disjointness, and the all-customer attack cardinality.
These policies are exposed through the single `prepare_data.py` interface;
existing cache defaults and artifacts remain unchanged. Source-v2 recurrent
models now execute both input layouts; repeat, first-step-then-zero, and
autoregressive SAE/VAE decoder schedules; mirrored/top-only state transfer; and
concatenate/literal-sum attention merges. VAE score branches now parse all
eight frozen IDs, compute stable Monte Carlo multivariate Gaussian
reconstruction probability for fixed or learned decoder variance, and preserve
lower-probability anomaly orientation through the ordinary runner. Three
deterministic minimal repairs of the undefined ROC/IQR threshold phrase are
also tested alongside supplied constants. All registered Attack-1 scopes,
Attack-2 granularities, Attack-3 minimal repairs, hour mappings, and
all-4,225/seeded-3,000 residential populations are fixture tested and exposed
through the same data-preparation interface. All six frozen SGCC
1,034-versus-48 representations, all four missing-data policies, and both
customer-disjoint/row-random sample splits are also runner-wired with
source-customer provenance. ISET now also executes strict/trimmed/aggregated/
interpolated 48-slot days and both customer/row split units with source-profile
provenance. Fixed-per-data-seed, per-model-seed, and per-experiment attack
regeneration schedules now resolve deterministically and are recorded in cache
metadata. Direct Table V runs now execute all four model/benign identity
readings and both full/seeded-3,000 sizes while persisting all six column score
sets. Both Eq. (10) reconstruction reductions now execute in FC/LSTM VAE
builders and are distinguished by source-derived numerical tests. Algorithm
5's autoregressive reconstructed-value feedback now also executes across both
attention merges, state policies, input layouts, and latent placements.
The ordinary runner now executes every frozen validation population, threshold
formula/scope, and validation/refit policy with source-derived fixture tests.
The distributed runner has semantic parity for neural branches. Stable branch
IDs now resolve through both dataset paths into preparation, model, classical,
validation, threshold, and Table-V arguments. Branch-specific ISET caches are
content-addressed and their preparation IDs are checked at load time. The
corrected track has a separate architecture contract, data isolation,
full-data classical branches, ISET seven-class SVM, likelihood-based VAE score,
binary sigmoid heads, and validation-selected Youden-J thresholds.

**Status:** Complete. The complete deterministic suite passes with 137 study
tests and 10 project tests. No production cache was rebuilt and no the cluster job
was submitted during Gate C.

### Gate D — bounded sanity cells

- Run tiny shape/identity/metric checks first.
- Run one complete low-cost cell per materially different pipeline.
- Verify training dynamics, score orientation, class counts, and table
  derivation before scaling.

**Status:** Partially complete. Local bounded cells pass, including finite
one-step updates for every neural family, all frozen classical completions,
ordinary/ISET/DDP structural dispatch, and a real SGCC printed-anchor
preflight. The exact-ISET printed anchor fails closed at its missing
content-addressed source-v2 cache. Required before Gate E: build and verify that
cache and run one real multi-GPU DDP smoke from the exact pushed commit. See
`results/gate_d_bounded_sanity_20260724.json`.

### Gate E — matrix execution

- Only rows passing Gates A-D may be called paper-reconstruction attempts.
- Preserve all seeds, failures, timing boundaries, and branch identifiers.
- Aggregate only comparable attempts.

## Finish conditions

- No blanket implementation-fidelity claim remains.
- Every reported reproduction result links to a passing traceability row and
  source/code/cache fingerprints.
- Tables I-V distinguish exact implementation, ambiguity branches,
  non-executable paper steps, and controlled diagnostics.
- The LaTeX report can state precisely what was and was not tested without
  relying on chat history.
