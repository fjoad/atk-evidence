# Where the saved sequential models lose profile differences

The three saved states exhibit different failures. At initialization, small
logit differences disappear in the float32 Sigmoid probability. In the final
unpoisoned model, the eighth GRU is almost zero and the classifier hidden
outputs are identical across profiles. In the final poisoned model, recurrent
activations become enormous, the scalar-Sigmoid bridge saturates near zero,
and GRU2 onward receives no distinguishable profile variation in native outputs.

No model was trained or updated. Panther job **408550 completed 0:0 in 2:11**
on one V100-16GB, 4 CPUs/16 GiB, within its 10-minute allocation. The diagnostic
program took 80.85s; six local and six cluster constructed observer checks pass.
The local audit matches cluster bytes, all 10 copied files verify, all 15 old
pair artifacts and 20 prepared arrays remain unchanged, and every observed
model/optimizer state hash is unchanged.

## Frozen scope

[The plan](../../../../docs/plans/2026-10-02-sequential-saved-state-diagnostic.md)
and observer were frozen at `3127b19372cd16546cbf814394ea8d52f7ee2f28`.
The user approved the bounded diagnostic after
[the constant-score pair](../sequential_pilot_20261001/README.md).

Three weight states: the shared saved initialization, finalp00, finalp30.
Each uses the first 100 original training rows and first 100 original test rows,
in preserved order and batch100. Initial gradients use both observed-label
conditions; each final model uses its own. Initial optimizer count is 0;
final counts remain 2250. No optimizer step, altered model setting, input
regeneration, seed search or test-selected subset. All research inference
and differentiation ran on a Panther compute node; local work was constructed
software tests and read-only arithmetic/artifact replay.

All three native test batches reproduce the old probabilities and intermediate
outputs exactly. The separate decoder observer also agrees with native
bridge/attention outputs. The source remains the scalar-Sigmoid/native-cell
interpretation documented in SEQUENTIAL_INTERFACE.md, with its unresolved
shape/cell/activation scope; this is not a uniquely specified paper model.

## Initial state: output precision hides small surviving differences

On the fixed100 test rows, the input's largest across-profile coordinate range
is14.38. Intermediate output range is0.01519; after eight GRUs the largest
profile range is2.13e-7. The native classifier logits still vary by1.814e-8,
but all float32 probabilities equal0.4999896287918091.

Recomputing only the final affine readout and scalar Sigmoid in float64,
from the same native float32 hidden values and frozen weights, yields a
probability range4.536e-9. Thus initial constant probabilities do not mean
that all internal profiles or gradients are identical. This arithmetic check
is not a float64 network, new detector, useful-ranking finding or trained
repair; no AUC/cutoff was selected from it.

Native stable BCE still supplies nonzero gradients through cached logits.
The missing derivative with respect to the returned probability tensor is
therefore recorded separately from a zero gradient. All 52 parameter gradients
are present and finite. The fixed training batch has 45/55 observed labels at
p00 and 62/38 atp30; these are local batch derivatives, not full-training
stationarity or gradients through the optimization trajectory.

## Final unpoisoned state: severe attenuation at the last GRU

Profile differences remain in the intermediate sequence and GRU1–7. The
maximum absolute final GRU8 output is only5.92e-18, with98.33% exact zeros.
Its largest across-profile difference is5.05e-23. The classifier hidden
output is identical across profiles, with 21 of 500 coordinates nonzero;
its output is not an entirely dead tensor. Both the native and widened final
readouts remain constant.

The training-batch input-gradient L2 falls from7.42e-11 initially to8.05e-28.
The intermediate-outputgradient is4.76e-25; the combined GRU parameter
gradient is3.26e-18, while classifier parametergradient is0.04534. These
are extremely small upstream derivatives, not exact zeros or a disconnected
computation graph. The earlier eight-step whole-layer-zero diagnosis does
not directly describe this48-step endpoint.

## Final poisoned state: recurrent growth and bridge saturation

On the fixed test batch, top-encoder activations reach3.116e9 and decoder
hidden activations1.394e15, all finite. Bridge logits range from about
-1.087e15 to-26.394. Its Sigmoid outputs are at most3.445e-12, and4,775 of
4,800 entries are exactly zero. On the training batch, the corresponding
maximum bridge output is1.977e-13, with4,782 exact zeros.

In both fixed batches, every bridge output from decoder step 2 through 48
is exactly zero. GRU1 retains tiny traces of the first step; GRU2 and all
later native stages are identical across profiles at each corresponding
step. The downstream recurrent/head outputs can be nonzero because constant
outputs are not the same as all-zero tensors. The final classifier remains
constant even under widened affine readout.

The fixed-batch input-gradient L2 is1.69e-28. Encoder parameter-gradient L2
is1.66e-24, while GRU/classifier gradients remain0.00665/0.03760. The earlier
kernel-column norm audits still pass: those constraints coexisted with the
observed large recurrent states. No claim that all activations were bounded
was justified by the kernel checks.

## Profile range through selected native stages

Each entry is the largest difference across the100 test profiles at a fixed
coordinate (including corresponding time steps for sequences). Exact zeros
are shown as0; nonzero values are not rounded into zero.

| Stage | Initial | Final p00 | Final p30 |
|---|---:|---:|---:|
| Intermediate sequence | 1.519e-2 | 1.053e-2 | 3.445e-12 |
| GRU1 | 8.639e-4 | 5.802e-4 | 8.593e-14 |
| GRU2 | 4.319e-4 | 2.402e-4 | 0 |
| GRU7 | 1.356e-5 | 1.841e-7 | 0 |
| GRU8 final state | 2.130e-7 | 5.046e-23 | 0 |
| Classifier hidden | 1.444e-7 | 0 | 0 |
| Native classifier logit | 1.814e-8 | 0 | 0 |
| Native probability | 0 | 0 | 0 |

Per-time-step reductions in [summary.json](summary.json) are post-run
arithmetic over all saved steps, produced by [derive_summary.py](derive_summary.py).
They involve no additional model call. Sub-ULP differences in auxiliary
float64 hidden-affine recomputation are not treated as native feature
variation; native outputs establish the equal-profile findings above.

## What this resolves, and what remains

The constant scores are not explained by cutoff choice or by an entirely
zero layer in every state. Initial output precision, final unpoisoned
attenuation, and final poisoned saturation are separately visible. Simply
widening the last output calculation does not restore either trained model's
profile distinctions. This diagnostic does not validate a full higher-precision
network or any activation/optimizer repair.

These are endpoints on fixed batches. They do not identify when training
entered either failure, isolate one gate/parameter as its cause, establish
an infinite-time limit, or transfer to all source interpretations or the full
population. The next named question is which saved recurrent gate/cell
operations produce the GRU8 attenuation and encoder/decoder amplification.
A bounded gate/state arithmetic check is appropriate before declaring a
repair or retraining. No follow-up job, new trained branch or website work
has been started by this result.

## Verification and artifacts

[Raw result](result.json), [execution record](execution.json),
[accounting](accounting.txt), [Slurm log](slurm-408550.txt),
[transfer hashes](artifact-sha256.txt), and [artifact audit](artifact_audit.json)
are preserved. All six native observations and their gradients are saved in
ignored raw NPZ archives locally and on Panther under
`data/derived/takiddin-2021-robust-poisoning/sequential-saved-state-20261002-attempt1`.
No scientific file from the original pair was modified.

The local auditor replays every saved summary, checks input/source/output
hashes, exact saved test outputs, and independently checks four output-bias
and output-kernel BCE derivatives by elementary arithmetic. The repository
check passed 403 cases (36 environment skips in that run), and the final
six-test observer suite passes separately in the pinned local and cluster
runtimes. Strict data verification passes the existing ScienceDB semantic-
equivalence branch; official restricted archives remain unavailable locally.
Journal consistency passes and the website is unchanged. The first repository
test command used a relative interpreter path across a directory change;
its preserved command error preceded research execution and was corrected
by using the normal test launcher. No experiment was rerun.
