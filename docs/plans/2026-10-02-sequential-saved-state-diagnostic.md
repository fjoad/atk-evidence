# Locate lost profile differences in the saved sequential models

Date: 2026-10-02. State: authorized; specification before observations.
The user approved the proposed bounded inspection after the completed pair.
No training, model repair, seed change or website work is included.

## Question and source boundary

Why do the saved 48-step initial/final classifiers give constant probabilities
although the eight-step constructed learner passed? Distinguish inactive
ReLU blocks, saturated scalar-Sigmoid output, and variation lost at float32
readout precision. This is an interpreted implementation diagnostic (I/M),
not a new numerical reproduction, causal robustness ablation or attainability
search. It feeds explanation E33 and the next experiment decision.

The complete paper and source/interface review remain the authority:
Fig.2 and III-B on p2678, IV-B/C p2682, Algorithm1/TableV p2683, and TableII
p2680. The missing readout shape/equation, native-cell choices and activation
scope remain unresolved in SEQUENTIAL_INTERFACE.md. This diagnostic does not
select another source interpretation or change the five direct files.

## Fixed observations

- Use only the original pair407294 at scientific3705bcca and its saved
  initial weights, final models, scores and representations. Bind every
  original file to the committed October1 SHA256 manifest, before and after.
- Three weight states: shared initialization (restored from saved arrays),
  final p00, final p30. Initial optimizer is newly initialized at0; final
  optimizer states retain2250. Observe without an optimizer step.
- Use the first100 prepared training rows and first100 test rows, in preserved
  order, with batch100. Selection is fixed before diagnostic outputs and
  matches the original test-scoring batch. Load and verify both original
  preparations. Initial gradients use both p00 and p30 observed training
  labels; each final model uses its own observed labels. Test labels do not
  select samples or settings. No new prediction metrics or cutoff search.
- Save native intermediate sequence, all eight GRU outputs, classifier hidden
  output and probability. Observe top encoder memory, decoder hidden sequence
  and bridge preactivation using the same cells/weights/attention/feedback.
  Require observer bridge/attention agreement with native outputs, and exact
  native test probabilities/representations against saved first100 rows.
- Record per-stage finite/zero fractions, ranges, L2 and maximum profile
  difference at a fixed coordinate. Training BCE gradients cover weights,
  input and native stage tensors. Preserve missing gradients separately from
  exact zeros (native stable BCE can bypass the final probability tensor).
  Save arrays and gradients so summaries can be replayed without inference.
- Compare classifier affine readout in float32 and float64 from the SAME
  native hidden outputs and frozen weights; evaluate its scalar Sigmoid in
  float64 descriptively. This is only arithmetic inspection, not a float64
  network, new trained branch or promoted detection score. No AUC/cutoff
  selection follows. Compare hidden ReLU preactivations too.
- Hash all model/optimizer state before and after each observation; unchanged
  state is mandatory. Report nonfinite diagnostic values explicitly and stop
  rather than silently repair or omit them. Preserve partial outputs.

## Predictions and decisions

An inactive stage has exact zero native output despite a varying input;
its corresponding upstream gradient may vanish. A saturated bridge has
extreme finite preactivations and outputs near0/1 with attenuated derivatives.
A precision explanation requires variation in native features/affine arithmetic
that disappears in the declared float32 probability; float64 readout is not
proof that another full-network implementation would learn. More than one
explanation may apply. Endpoints cannot date the onset or reconstruct an
unobserved training trajectory. If evidence is inconclusive, preserve that
uncertainty rather than adding interventions or fitting a repair.

## Execution and verification

Implement one small direct observer plus discriminating constructed checks
for zero-layer detection, bridge saturation, profile-versus-time variation,
precision loss, state preservation and saved-output parity. Small fixtures
may run locally; all research inference/gradients run on a Panther compute
node. Existing pinned TF2.16.2/Keras3.4.1 environment and deterministic float32
settings; no dependency installation or alternate device/model.

One GPU-short allocation: oneV100-16GB,4CPUs/16GiB,10min ceiling. Internal
observation guard7min checked between blocks; retain partial artifacts on
failure. Three states/two fixed batches only; zero model fits/optimizer updates,
no extra sample/seed/setting. Constructed observer checks run before loading
research inputs. Inspect scheduler and the new output path before submitting
once. Freeze source/plan/tests first. This allocation is authorized by the
user's go-ahead for this bounded diagnostic; no further permission checkpoint
is required for these steps.

Afterward copy outputs, replay summary/gradient/state/source/input hashes,
record findings and limits, update STATUS/CONTEXT/explanation register, verify
and commit. Stop before any subsequent training or branch. Original artifacts,
contracts and website remain unchanged.

## Freeze verification

Six constructed observer tests pass in the local pinned neural runtime,
including active-path label-gradient signs and native-output/state parity.
The repository run passes403 cases (36 environment-specific skips before
adding the final active-path TensorFlow fixture); that additional fixture
also passes in the pinned run. Strict data verification passes the existing
ScienceDB semantic-equivalence branch; restricted official archives remain
unavailable locally. Journal consistency and shell/Python syntax pass.
Five direct scientific files are unchanged from the original pair. No
research observations have been made yet. The read-only auditor additionally
checks output-layer BCE bias/kernel derivatives against elementary arithmetic.
