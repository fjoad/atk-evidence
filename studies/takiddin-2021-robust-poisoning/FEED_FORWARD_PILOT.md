# First feed-forward pilot

Recorded 2026-09-21 before research-data fitting. User approved the next model.

## Question and source

Does the reported feed-forward architecture learn on the original verified
pilot, and does 30%-customer poisoning alter ranking, the default cutoff, or
both? This is a bounded numerical pilot under an explicit interpretation,
not full Table III reproduction or a test of all neural configurations.

The complete paper was read in the source phase. Section III-B.2(d) and
Equation(1), p.2679; Table II/training settings, p.2680; and Table III,
p.2681, were visually rechecked. Preserve six hidden layers of 500 neurons,
ReLU hidden activation, Sigmoid output, Adamax, no dropout, weight constraint 3,
50 epochs and batch 100. The paper names Keras Sequential and an RTX 2070,
with roughly 1.5-3 hours for full-data deep-model training.

Equation(1) repeats log(p) in both binary terms. Taken literally its loss is
-log(p), independent of y. Preserve that fact and verify label/gradient
cancellation in a constructed test. This pair uses the obvious intended
binary-cross-entropy repair with log(1-p) in the negative-class term. It is
explicitly an interpreted repair, not a literal implementation of Equation(1).
No research-data model is trained with the label-independent loss in this pair.

## Initial executable completion

Input: the 48 standardized half-hour features, flat and in their saved order.
Sequential: six Dense(500,activation='relu') hidden layers, then
Dense(1,activation='sigmoid'). Use biases, GlorotUniform kernels and zero
biases, float32, no dropout, normalization layer, regularizer, sample/class
weights, validation-selected checkpoint, or early stopping. Parameter count:
1,277,501. One scalar output is an explicit binary-shape completion.

Interpret constraint 3 as Keras MaxNorm(3,axis=0) on every Dense kernel,
including the output; biases unconstrained. The source does not identify
constraint type, axis or layer coverage, so retain this as a named completion.
[Keras constraint documentation](https://keras.io/api/layers/constraints/)
explains the incoming-weight norm convention. Record final per-layer maximum
column norms, and verify projection after updates on constructed weights.

Use Adamax learning_rate=.002, beta_1=.9, beta_2=.999, epsilon=1e-7,
no decay, clipping, EMA, accumulation or schedule. The
[Keras 2.3.1 source](https://github.com/keras-team/keras/blob/2.3.1/keras/optimizers.py)
uses the .002 historical default; current Keras defaults differ. The paper
does not specify its learning rate or software version. Pin TensorFlow 2.16.2,
Keras 3.4.1 and requirements-feed-forward.txt; this is not an identified
author environment or bitwise reproduction of old Keras.

Set seed 20260920 before each new model so initialization is paired. Save a
canonical hash of all initial weights and require it to agree between p00/p30.
Shuffle all training examples each epoch with a seeded TensorFlow dataset,
buffer equal to the training size, reshuffle_each_iteration=True; batch 100,
no dropped remainder, one private input thread and prefetch 1. Both conditions
use the same input ordering rule. Fifty epochs imply 2,250 optimizer updates
(45 batches/epoch, last batch 64) for this pilot. Fit all training rows after
replaying the selected published architecture; do not redo the unspecified
cross-validation/hyperparameter search or claim to have reproduced it.

Compile with standard BinaryCrossentropy(from_logits=False), reduction over
the batch, and binary accuracy. Disable XLA, mixed precision and TF32; request
deterministic operations. Native primary labels use p>0.5, with ties benign,
and the scalar p is the ROC score. Represent saved probabilities as [1-p,p]
in float64 after float32 model inference so the existing metric audit applies.

## Fixed inputs, targets and saved evidence

Reuse the original p00/p30 preparations: 4,464 training and 2,232 test rows,
20 customers, first 28 complete days each, pre-split ADASYN unchanged. The same
six poisoning customers cause 675 malicious training labels to become benign
at p30 (15.12097% of training rows). Verify the frozen metadata/array hashes
through the existing loader. No new attacks, scaling, balancing, or split.

Table III p00/p30 context (%): DR 90.8/76.0, FA 9.3/24.4, SP 90.7/75.6,
PR 90.0/75.8, ACC 90.7/75.8, F1 90.4/75.9, AUC 91.1/76.1. These values
concern the full population; a pilot match is not full-paper reproduction.

Save all seven metrics, confusion counts, per-attack and original/synthetic
breakdowns, original-only metrics, constant-prior/daily-mean controls, and
fixed-score ROC/reversal diagnostics at FA caps 9.3/24.4% and 17.6/33.3%.
Test-selected thresholds remain diagnostics, not calibrated deployment rules.
Save every epoch's loss/accuracy/time, actual update count, initial/final
weight hashes, final architecture/optimizer configuration, final model,
constraint norms, final observed/true-label training accuracy, probabilities,
GPU/driver/runtime details and memory/timing. Require exact model reload
agreement on the same device/batches. Do not choose the best epoch from test
performance; report epoch50 if completed, otherwise the interruption plainly.

## Compute and stopping

Dependency setup is a separate CPU-only allocation capped at 15 minutes,
four CPUs/16 GiB, with no research-data fitting. Constructed examples can be
tested locally and on compute nodes. The research pair gets one 20-minute
job with one V100-16 GB, four CPUs and 16 GiB host memory; no multi-GPU or CPU
fallback. Before loading research inputs, verify TensorFlow sees exactly one
V100 GPU, can perform a constructed dense update there, and passes fixtures.
An incompatible runtime or failed GPU check is an operational failure, not
a negative paper result. Preserve it and do not silently change model settings.

The wrapper prepends the isolated environment's NVIDIA library directories
to its process-only library path, avoiding accidental selection of a different
system CUDA installation. Save the complete installed-package list and actual
TensorFlow build/device metadata. Existing Python environments stay unchanged.

The pair fits exactly two models, p00/p30, one seed, at most 50 epochs each.
The per-fit training guard is seven minutes; stop after a completed epoch if
reached and preserve the partial model/history without calling it complete.
The job's outer 20-minute limit is the hard allocation bound. Record actual
hardware, wall time, epoch times and memory. The RTX 2070's 8 GB versus V100's
16 GB does not establish an exact speed ratio. This small-pilot cost does not
test or extrapolate the paper's full-data 1.5-3-hour claim.

No extra seed, optimizer, loss, layer, full-data run or next model in this
allocation. A large mismatch first triggers the saved label/update/loss and
cutoff checks, not automatic retraining. Preserve matches and adverse outcomes.
After artifact verification, stop and update the journal/site draft, noting
the remaining source/data/software uncertainty. No publication in this step.
