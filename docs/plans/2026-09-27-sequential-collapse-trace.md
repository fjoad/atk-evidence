# Trace the failed constructed ensemble trajectory

Date: 2026-09-27. State: completed and audited; learning gate remains failed.

The user approved tracing when the reversed-label training collapses, before
changing settings. This is one diagnostic replay of a software fixture, not
another research model, seed trial or attempt to improve a result.

## Fixed question and limits

Does the final GRU go inactive first, or does a later classifier layer stop
passing gradients before the GRU becomes inactive? Could an input collapse,
an affine parameter change, or an instrumentation mismatch explain the event?

- Use the preserved constructed data, original model seed20260920, all model
  settings and the reversed labels from sequential_constructed_20260927.
- Replay only the first50 of its300 updates, using the same `model.fit`,
  one32-row batch/epoch and pinned CPU runtime. One replay, five-minute
  batch-boundary guard. No normal-label refit, setting change or extra seed.
- Before training and after each update, record both train/test layer ranges,
  variation and positive fractions, BCE/accuracy, and gradient norms. Save
  model/optimizer snapshots so the event can be inspected without refitting.
- Verify identical initial weights and every replayed epoch loss against the
  original history. Any discrepancy is an instrumentation/replay limitation,
  not an explanation of the old fit; preserve it and stop interpretation.
- Identify the first all-zero output in each ReLU stage. Distinguish a
  layer that was already zero, a transient zero, and a new zero transition.
- At the classifier-hidden layer's first new zero transition, if present,
  inspect its input and affine preactivations before/after. Evaluate a fixed
  two-by-two input/parameter comparison and separate weight/bias substitutions
  using those saved states only. These are local forward counterfactuals,
  not trained repairs or tests of eventual predictive performance.
- Check that observation leaves model/optimizer values unchanged. Preserve
  inputs and all original artifacts. Budget includes recording; no GPU job,
  CER input, full-width fitting, alternate activation or website change.

Save a short direct trace script and meaningful checks of zero-transition
classification and affine decomposition; freeze before replay. Audit results,
record the precise local finding/remaining cause, update status and commit.
If no transition occurs in50 updates, report that limit without extending.

## Read-only follow-up specified after the trace

The50-update replay completed with exact original loss-prefix agreement.
Both last GRU and classifier-hidden outputs first become zero after update2.
The head's updated parameters also zero its output using the old GRU inputs.
Before interpreting that as a unique cause, inspect only snapshots1 and2:
evaluate GRU8 with the fixed2x2 combination of old/new GRU7 inputs and old/new
GRU8 parameters. Then swap only its candidate bias in each direction, holding
the other parameters/input fixed. Independently calculate the reset-before
GRU recurrence and compare it with the native layer and saved outputs.
Six block forwards, no optimizer update, new data, seed or training. This
tests local parameter/input sufficiency, not whether any modified model would
learn or pass the original gate. Preserve the source plan's pre-run version
through commit9368a90; this paragraph does not retroactively preregister the
additional saved-state question.

## Outcome

The replay finished50 updates in12.37s from identical initial weights. Every
original training loss matches exactly; observations preserve model/optimizer
values. GRU8 and classifier hidden outputs first become all zero after
update2 on both constructed train/test sets and remain zero through50.
Upstream classification gradients are zero from2 onward; the earlier
intermediate and GRU1-7 stages remain active.

The head's changed parameters kill its ReLU output even with old inputs;
old bias restores positive activations with the new inputs/weights. GRU8's
candidate-bias update likewise is sufficient to zero its output with old
inputs/other parameters, and reverting that bias alone restores positive
activation with the new input/other parameters. Independent recurrence
replay matches all six native block cases within8.93e-12. These are local
forward witnesses, not trained repairs. Both stages change within update2;
no finer causal ordering is claimed.

All51 snapshot hashes and prior artifacts verify. Fresh-state replay at
0/1/2/50 matches saved values exactly and gradient zero patterns agree.
Three new checker fixtures pass;397 repository tests pass with30 skips;
strict data verification and read-only journal consistency pass. Scientific
model/training code, old results and website are unchanged. See the
[result record](../../studies/takiddin-2021-robust-poisoning/results/sequential_trace_20260927/README.md).

Next decision: a separately bounded full-width constructed learning check
can test whether the reduced-width instrument is unrepresentative, before
any optimizer/activation repair or extra seed. It has not run. This trace
does not promote the failed learner or authorize GPU timing/research fits.
