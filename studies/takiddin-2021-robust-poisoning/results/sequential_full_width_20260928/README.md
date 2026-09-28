# Published widths do not pass the fixed software learning check

Both full-width runs completed their 300 updates but stayed at **50% accuracy,
BCE log(2), probability 0.5 for every constructed example**. The learning gate
fails, so the authorized dependent GPU timing and real-data pair were not
launched. This is a software validation result on eight-step synthetic data,
not a paper-data reproduction or a result for all source interpretations.

## Authorized comparison

The user approved the whole bounded sequence, conditional on its gates.
[The plan](../../../../docs/plans/2026-09-28-sequential-execution.md) and
`checks/sequential_full_width.py` were frozen at **66db165** before either
outcome was seen. It compares the earlier narrow fixture with the published
layer widths, retaining exactly the old synthetic arrays, eight time steps,
model seed 20260920, Adam .001, no dropout, MaxNorm 1, native cells and the
declared scalar ReLU intermediate projection.

The full-width model has 9,240,802 parameters: encoder 500/300/200, mirrored
decoder, eight 300-unit GRUs, Dense 500 and scalar Sigmoid classifier. The
fixed 32 training/32 test inputs and both label orientations are unchanged.
The sequence is still eight steps to isolate width; this does not demonstrate
learning at the 48-step research layout. All model/training source settings
remain those of the frozen I-SEQ-native-IVC completion.

Each fit was allowed 300 full-batch updates and a prospectively declared
600-second CPU guard. Both completed well inside the guard. The criterion
remained at least 90% held-out accuracy and BCE<log(2)/2 in **both** orientations;
there was no threshold relaxation, seed retry or favorable checkpoint choice.

| Label orientation | Updates | Fit seconds, including compilation | Test accuracy | Test BCE | Gate |
|---|---:|---:|---:|---:|---|
| Normal | 300 | 105.63 | 50% | 0.69314718 | Fail |
| Reversed | 300 | 103.80 | 50% | 0.69314718 | Fail |

The constant-prior comparison has the same accuracy/loss. The corresponding
sign-of-mean rule classifies the constructed examples perfectly. Initial
weights are identical across orientations; their final weights are also
identical. Initial and final aggregate classification gradients are zero in
every recorded model group. All biases remain unchanged. Kernel values can
change under the weight constraint even with a zero gradient; weight changes
alone do not demonstrate learning.

## The inactive stage is an interpretation choice

A fresh-process [artifact audit](artifact_audit.json) locates the first
observed complete zero output at the scalar intermediate projection. On the
initial training profiles its input contains active decoder features, with
maximum 0.03529. But the affine projection's raw output ranges from
**-0.02035 to -0.001265**. ReLU therefore produces zeros at every time step
for all 32 training examples. The held-out set has the same sign pattern.

With zero projected inputs and zero initial states/biases, all eight GRUs
and the classifier hidden layer output zero. The last Sigmoid consequently
returns 0.5. The balanced BCE gradient of the output bias cancels across the
batch; the earlier ReLU paths also transmit zero gradients. The bridge is
still zero on all tested train/test examples after 300 updates.

This differs from the earlier narrow run, whose reversed-label trajectory
became inactive downstream after update 2. Increasing width did not rescue
the declared gate; it produced a different inactive path at the same seed.
It does not isolate width as the unique cause of either failure.

**The scalar projection and its ReLU activation are our documented completion
of omitted details.** See the intermediate-projection row and open alternatives
in [the source contract](../../SEQUENTIAL_ENSEMBLE_PILOT.md). They must not be
retrospectively described as a fully specified printed operation or evidence
that the authors used this exact bridge. Linear/Sigmoid alternatives and the
intermediate tensor shape remain open. This failure is not grounds for
silently choosing whichever alternative learns best.

The next source question is the decoder-to-GRU interface: what tensor shape
and activation are justified by Fig.2, Algorithm1 and the ensemble settings?
Resolve and separately declare a minimal alternative/control before another
fit. More epochs or another seed on this failed software check is not the
immediate next question.

## Artifacts and verification

[result.json](result.json) records seeds, dimensions, gradients and artifact
hashes; the frozen source and serialized model retain the model/optimizer
configuration. Ignored full models/arrays and histories are preserved under
`data/derived/takiddin-2021-robust-poisoning/sequential-full-width-20260928-attempt1`.
The [execution log](execution.txt) includes both complete histories' progress.
The frozen source and prior synthetic data hashes verify, every saved file
hash verifies, and metric recalculation agrees. Fresh Keras loading
reproduces every probability, intermediate output, attention tensor, weight
hash and optimizer iteration count. The initial/final diagnostics made no
optimizer updates and left all original files unchanged; their
[audit script](audit_saved_fits.py) is retained.

Both label orientations ran exactly once. No settings, layer widths, seeds or
update budgets changed after an outcome was observed. Prior narrow-model
failures, traces, research inputs/models and source records remain intact.
Repository verification passes 397 tests with 30 environment skips, and strict
data verification passes the existing ScienceDB semantic-equivalence branch.
Journal consistency passes read-only; website work remains untouched.

Stage 1 is complete with a failed gate. Stages 2-4 are held: no full 48-step GPU
timing, real-data ensemble fit, comparative research metrics or new mechanism
claim exists. The user's conditional approval is recorded, but does not
silently select a different scientific interpretation after this failure.
A noninteractive Panther connectivity probe reached authentication;
no remote job was submitted. Access was not the reason for this hold.
