# Full48-step recurrent-cell comparison

The user approved one source-explicit tanh-cell alternative and constructed
validation under [the contract](../../SEQUENTIAL_RECURRENT_CONTROL.md).
Four fixed software fits compare it with the existing ReLU reference under
normal/reversed labels, with all initial weights/settings otherwise matched.
No CER inputs or research fits. No job yet at this source freeze.

Twenty relevant pinned neural checks pass;405 repository cases pass with45
environment skips. The initial new reload harness mishandled a nested output
list; outputs[0] corrects the harness and legacy/new numerical outputs match.
Both the failure log and passing checks are preserved. Website unchanged.

## Live execution

Submitted once as408764, frozen `6ca174a8e58cd47ccb5af656f1cbfa534c29a156`.
OneV10016GB/4CPUs/16GiB,35min ceiling,360s/fit; no CER data. Check this
existing job/output before any continuation. Four cases remain fixed in
advance; no duplicate or automatic research promotion. Log:
`/export/home/fjoad/seq-cells-transfer-20261002/cells-408764.out`.

## Attempt1: pre-fit device error

408764 failed1:0 in3:04 after all six cluster fixtures passed. A broad
GPU-only caller scope forced TensorSliceDataset ontoGPU although its kernel
is CPU-only. The first full-width case saved initial weights/predictions/
layer arrays, but performed no optimizer update and wrote no fitting history
or final model. This is an execution error, not a failed learning result.
[Failure](attempt1/result.json), [log](attempt1/slurm-408764.txt), and
[accounting](attempt1/attempt1-accounting.txt) remain preserved.

The placement-only recovery retains the source model and all scientific
settings, uses an unusedattempt2, and checks exact initial-array agreement
before fitting. Retry ceiling31min plus prior184s totals at most34:04 of
nominal allocated time, within the original35min budget. No CER data or
research fit; no result-driven setting change. See the amended step plan.

## Live corrected execution

Job408778 at `febc28e4aaf626302f884524c1184a79f59b2c02`,31min ceiling,
uses unusedattempt2. Only device placement and preservation checks changed;
the scientific model/contract/settings remain at6ca174a. Inspect this job,
not a new submission. Log:
`/export/home/fjoad/seq-cells-transfer-20261002/cells-retry-408778.out`.

## Declared numerical stop

408778 stopped1:0 in18:08. Both tanh cases completed300updates with100%
held-out accuracy/BCE about1e-7 and passed their case checks. ReLU-normal
became nonfinite atupdate245, so ReLU-reversed was not run. The original
comparison result remains failed/incomplete; it is not relabeled complete.
No further fitting follows. Local saved-state/metric/configuration audit passes.
Only the planned fresh-process GPU verification of the two completed tanh
models remains:3min, zero updates, within the original cumulative budget.
