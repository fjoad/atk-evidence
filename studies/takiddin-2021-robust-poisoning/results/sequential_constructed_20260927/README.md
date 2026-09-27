# Sequential ensemble implementation: software passes, learning gate fails

The declared I-SEQ-native-IVC model is implemented in the direct
`reproduction/models.py`. Its full runtime parameter count is **9,240,802**,
and full-size constructed forward shapes/attention are verified. The frozen
reduced-width learning test failed in one label orientation. **Do not promote
this implementation to GPU timing or research fitting yet.**

This is constructed software validation, not a CER/ISET experiment, numerical
reproduction, mechanism result, or limitation on the paper's model family.
The full-size model has not been trained. There was no cluster allocation,
research-data scoring, alternative architecture, extra seed or website work.
The research runner has not been extended to launch the new model; that
integration remains behind the failed constructed learning gate.

## What was implemented and checked

Source/design contract: [SEQUENTIAL_ENSEMBLE_PILOT.md](../../SEQUENTIAL_ENSEMBLE_PILOT.md),
frozen at a351652; [implementation plan](../../../../docs/plans/2026-09-27-sequential-implementation.md).
Model and initial constructed runner are preserved at b9d8223. The subsequent
2f9b793 changes only the report writer/recovery organization and its regression
test; `models.py` is byte-identical between those commits. Per-layer initializer
seeds were declared in the implementation plan before observing fit results.

The model has six native ReLU LSTM encoder/decoder layers, causal decoder-query
attention and scalar ReLU feedback, eight native ReLU GRU layers, a 500-unit
ReLU hidden layer and scalar Sigmoid output. Adam, joint corrected BCE, no
dropout and MaxNorm 1 implement the declared Section IV-C interpretation.
No standalone MAE/MSE objective or 0.51 cutoff is inherited. The native-cell
and intermediate-activation alternatives remain explicit source ambiguities.

All pre-existing model functions are AST-identical to their a351652 versions.
The old standalone AEA prototype and prior scientific artifacts are unchanged.
The custom ensemble layer is registered lazily to keep classical-only use
independent of TensorFlow; a fresh load process calls
`models.register_sequential_layer()` before Keras deserialization.

Nine ensemble/gate tests pass in the pinned TensorFlow runtime:

- Full parameter count, layer settings and actual 48-step forward dimensions.
- Paired initialization and rejection of incorrect shape/width counts.
- Decoder-query intervention with fixed memory and a disabled-query control.
- Feedback intervention changing decoder state, removed by disabling the
  feedback input weights; the actual loop also passes previous output onward.
- Mirrored encoder-to-decoder state transfer, finite connected gradients,
  an update and constraints. Finite or nonzero gradients alone do not pass
  the learning gate.
- Fresh-process reload with exact probabilities, intermediate outputs,
  attention, every weight and optimizer count.
- Rejection of constant, partial and nonfinite learning results; deterministic
  balanced constructed data with an easily verified mean-based solution.

The six existing standalone AEA tests also pass. These fifteen software tests
must not be described as fifteen successful learning experiments.

## Fixed learning pair and retained failure

Both runs use the contract's model seed 20260920 and data seed 20260926,
32 training and 32 fresh test rows, 8 time steps, encoder 8/6/4 with mirrored
decoder, eight 8-unit GRUs and Dense 16. Thus the learning instrument has 5,082
parameters while retaining the full layer order/depth. Inputs are
`.75*(2*y-1) + .05*Normal(0,1)`. The second fit reverses train/test labels
on exactly the same inputs and starts from identical weights. Each completed
300 full-batch updates within its 300-second guard; no fitted case was retried.

The predeclared gate requires at least 90% test accuracy and BCE below
log(2)/2 (approximately 0.34657) in both label orientations.

| Case | Updates | Fresh-test accuracy | Fresh-test BCE | Gate |
|---|---:|---:|---:|---|
| Original labels | 300 | 100% | 0.22528521 | Pass |
| Reversed labels | 300 | 50% | 0.69314718 | Fail |
| Constant probability 0.5 | 0 | 50% | 0.69314718 | Comparison |
| Sign of daily mean, with corresponding label orientation | 0 | 100% | Not a probability model | Comparison |

The original fit took approximately 11.69 seconds including training-function
compilation; the reversed fit took 11.53 seconds. These small CPU fixture times
are not estimates for the full model or paper dataset. Initial hashes match;
both optimizers reached exactly 300 updates. All weights/losses remain finite,
and the declared final norm checks hold.

Initially every layer group receives a nonzero classification gradient,
although attention gradients are about 2.08e-12. Every group has some changed
weights at the end. Nevertheless the reversed fit's final encoder, attention,
decoder, intermediate and GRU gradient norms are exactly zero on the 32 training
rows. Its final classifier gradient norm is about 5.96e-8. This is why software
connectivity/weight-change assertions cannot replace an actual learning check.

## Read-only diagnosis of the saved fits

The independent [artifact audit](artifact_audit.json) verifies all saved
hashes, the generated input identity, historical source bytes, initial/final
weights, optimizer counts, metrics and exact fresh-reload outputs. It also
checks that the original reporting-failure artifacts remain unchanged.

On the reversed fit's 32 test rows, the last GRU and classifier hidden outputs
are identically zero. All final probabilities are 0.5000000596. The preceding
AEA intermediate outputs are active, with range 0.00102015-0.08715413; their
raw preactivations are positive. Therefore these observations do not identify
the scalar ReLU intermediate projection as the dead stage. They locate a
downstream collapse in the saved state, without isolating the training event
that caused it.

As a separate capacity sanity check, complementing the successful normal
model's probabilities yields 100% reversed-label accuracy and BCE 0.22528522.
For a Sigmoid classifier this corresponds to negating the final affine
weights/bias. It involves no additional optimization and shows that the
architecture can represent the reversed answer. It is not a recovered
training run, seed selection or a new eligible model result.

We have not tested full-width learning, a different activation, another seed
or a wider update budget. The failed small configuration does not establish
that the full ensemble or the paper cannot learn. It does establish that this
declared implementation has not passed its prerequisite learning gate.

## Reporting defects and recovery provenance

The normal fit saved its complete history, final Keras model and predictions,
then its final JSON write failed because the gate returned `numpy.bool_`.
The original attempt remains at the ignored path
`data/derived/takiddin-2021-robust-poisoning/sequential-constructed-20260927-attempt1`.
Its started-status report is not relabeled in place. See [failure log](report-failure.log).

The writer now returns a Python bool, with a serialization regression test.
The [recovery script](recover_report.py) copies the original artifacts to
`sequential-constructed-20260927-attempt1-recovery`, reloads and audits the
normal fit without a single optimizer update, and completes only the previously
unrun reversed-label fit. Original input/model/source hashes and every original
file hash are retained in [result.json](result.json). Source snapshots resolve
against b9d8223; the recovery runner hash resolves against 2f9b793. The audit
and recovery scripts are records of this specific attempt, not research runners.

Two read-only audit-helper attempts stopped on Keras output-list handling and
an observation-hook signature before a final report was produced; see
[audit-failure.log](audit-failure.log). Both were fixed without fitting or
changing saved models. The final [audit script](audit_saved_fits.py) completed.

## Verification and next decision

Repository checks passed: 140 study tests plus 284 project cases, with 30
environment skips: 394 passed. The final pinned run passes all 9 ensemble/gate
tests plus 6 historical AEA tests. Strict data verification passes the already
recorded ScienceDB semantic-equivalence branch; no dataset was substituted.
Journal consistency passes read-only. No website generation or publication.

Stop before constructed GPU timing, as the frozen learning gate requires.
The next useful question is why the reversed-label trajectory collapses at
the last GRU/classifier, and whether this is specific to the reduced-width
instrument. Any replay instrumentation or source alternative needs a small
declared test; do not silently relax the gate, change the seed or call the
positive original-label result sufficient.
