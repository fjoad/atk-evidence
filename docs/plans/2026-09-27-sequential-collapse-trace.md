# Trace the failed constructed ensemble trajectory

Date: 2026-09-27. State: specified before replay.

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
