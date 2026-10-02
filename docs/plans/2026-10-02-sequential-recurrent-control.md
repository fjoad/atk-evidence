# Validate one recurrent activation alternative at full sequence length

Date:2026-10-02. State: implemented and verified; ready to freeze before learning outputs.
The user approved the next alternative/control and full48-step validation.

1. Recheck the relevant complete source pages; declare the tanh-cell
   alternative and the unresolved IV-C/Algorithm1 conflict before outputs.
2. Extend the direct model with an explicit serialized cell activation;
   preserve default ReLU, old model behavior and all old artifacts. Validate
   legacy/fresh serialization, paired weights, Sigmoid gates, bounds and
   classification gradients with small constructed checks.
3. Freeze and run the four fixed full-width48-step cases on one Panther
   V10016GB/4CPU/16GiB allocation,35min ceiling,360s/fit. Tanh normal/reversed
   first, then the matched ReLU reference;300 updates each. No CER data or
   new seed/settings. Stop on partial/nonfinite/integrity failure.
4. Copy and audit every saved output/history/weight/config/optimizer/source
   identity; record all results and the precise next decision. Update status,
   context and explanation register, verify and commit. No website work.

The full source/budget/stopping contract is
[SEQUENTIAL_RECURRENT_CONTROL.md](../../studies/takiddin-2021-robust-poisoning/SEQUENTIAL_RECURRENT_CONTROL.md).
Tanh is a separately labeled source interpretation, not a silent repair or
literal implementation of every printed cell equation. No research-data fit
follows automatically, including after a complete constructed pass.

## Prelaunch verification

Complete relevant source pages were visually rechecked and unchanged PDF
hash verified. The serialized relu/tanh option preserves default ReLU;
legacy and new archives pass fresh-process software reload. Tanh hidden
bounds, unchanged gates/Dense/readout and paired initial weights verify.
Twenty pinned neural tests pass, including the new six controls and all14
existing sequential checks. Repository405pass/45skip; strictdata and journal
checks pass on the existing declared provenance branch. A new regression
harness initially used a nested output list; selecting outputs[0] fixed the
harness, with no model numerical change. Error log preserved. No new learning
job or CER fit at this checkpoint.

## Pre-fit device-scope failure and bounded operational recovery

Job408764 at6ca174a failed1:0 in3:04 (184s). All six cluster software checks
passed. The first full-width case saved its initial arrays/hash but reached
no fitting update: the caller's broad GPU-only scope forced the CPU-only
TensorSliceDataset operation ontoGPU. No history/final model exists for that
case. All original pair files remain unchanged; the failed attempt is kept.

Correct only execution placement: explicitly create model variables onGPU,
allow the CPU input pipeline outside that scope, and assert both placements.
Keep model source/contract/data/seed/optimizer/300updates and360s fit guards
unchanged. New attempt2 must reproduce every saved initial weight, prediction
and layer array from attempt1 BEFORE fitting and verify old files before/after.
This is recovery from a pre-fit software error, not a new scientific branch,
seed retry or continuation from a trained checkpoint. The existing user
approval covers completing this unchanged comparison.

Retry ceiling31min:184s already used +1860s requested =2044s (34:04), below
the original35min allocation budget. Whole-minute downward rounding avoids
relying on Slurm second-level rounding. No budget extension or automatic
additional attempt. Validate the corrected CPU-pipeline/GPU-weight fixture,
freeze the recovery revision, inspect queue and unusedattempt2 before submit.
