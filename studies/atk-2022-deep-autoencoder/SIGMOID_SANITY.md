# Sigmoid: complete-evaluation bound and two no-training controls

Date: 2026-08-31

Status: recorded before new scoring; user approved a quick investigation,
without full retraining, after publication of the discussed source checks.

## Question

Can changing the FC-SAE output range to Sigmoid's reach the Table III target
on the **complete unchanged prepared evaluation**, rather than just original
rows? If an optimistic bound permits it, do two concrete, label-blind
reconstructions provide any corresponding evidence of attainability?

Classification: exploratory `X/A`, extending a source-motivated controlled
output-range check. This is not a new numerical reproduction or a test of
the paper's recurrence/attention mechanism. The adaptive origin is explicit:
the original-row Sigmoid bound has already been observed and published.

Competing outcomes:

1. The full-data high-error bound excludes the target too. No training under
   this exact input/range/score combination can rescue it.
2. Synthetic benign examples make the full bound permit the target. The
   earlier original-row exclusion must not be generalized to that evaluation.
3. Reversing the score permits the target, but feasible controls do not reach
   it. This leaves a changed recipe open; it does not demonstrate a match.
4. A label-blind control reaches the target pair. Report that counterevidence
   plainly, while checking all seven metrics and its non-paper status.

## Fixed inputs and target

- Original scientific commit: `a88d17477ad96b01ffa44a50d8ce051dd8d2b5ca`.
- FC-SAE attempt: `seed_20260824_2f483335536c`, original seed `20260824`.
- Original result SHA-256:
  `ae07b42ef6c84242ca9b39db8b8828694d6d4df6859abdee090fc0a613a69154`.
- Published original-row reference SHA-256:
  `3ef1cf59cae7bc7e9dcec2f2d8119b65b221f26b5d2b455b314b14546362dbd8`.
- All 8,884,989 prepared rows: 750,767 original benign days, six attack
  siblings per day, and the unchanged synthetic benign rows. The secondary
  view is the first 5,255,369 original rows. No regeneration or rescaling.
- Verify consumed array hashes and original/synthetic block identities.
- Target pair: detection >=81%, false alarms <=15%; rounding-tolerant pair:
  detection >=80.5%, false alarms <=15.5%. Report balanced accuracy and AUC
  ceilings too. A passing pair does not establish the complete reported row.
- No training seed is added; row selection is deterministic, not stochastic.

## Bound and concrete controls

The closed cube `0 <= r_j <= 1` contains all Sigmoid outputs. For MSE:

```
L(x) = mean((x - clip(x, 0, 1))**2)
U(x) = mean(max(x**2, (x - 1)**2))
```

Each squared coordinate distance is minimized by clipping and maximized at
one of the two interval endpoints. Thus these are real-arithmetic extrema.
Use the unchanged cube helper and threshold-enumeration helper; evaluate in
float64, outward-pad by `1e-5*(1+U)`, and clip the lower endpoint at zero.
This is not certified interval arithmetic.

Primary rule: error >0.58 means attack. Control: error <0.58 means attack.
For both directions enumerate all distinct score boundaries. The bound may
select outputs with knowledge of the labels and independently for each row;
an actual model may not realize this selection. Confirm that original-row
bounds reproduce the existing record within 1e-7 percentage points.

Two fixed label-blind reconstructions, neither using training or test labels:

1. `r(x) = clip(x, 0, 1)`: the nearest cube reconstruction, a no-learning rule.
   It includes exact endpoints that finite Sigmoid logits only approach; label
   it a closed-range control, not a trained Sigmoid network.
2. `r(x) = 0.5`: a valid constant Sigmoid reconstruction (zero logits).

For both views and directions, record all seven metrics at the corresponding
0.58 cutoff and descriptive all-cutoff maxima. Label-aware cutoff selection
is only a diagnostic, never a chosen reproduction threshold. Keep every
outcome, including reversals and unfavorable controls. Do not infer that a
nearest reconstruction is necessarily the best classifier.

## Budget, pilot, and stop

One CPU allocation: four cores, 8 GiB, ten-minute allocation ceiling. No GPU.
Pilot: 64 evenly spaced rows per original block and 64 synthetic rows (512
total). Full pass: streaming chunks of 32,768, then one-dimensional sorting.
Each process has a 240-second hard analysis cap, with partial/failure records
preserved. The allocation cap also covers setup and manual pilot inspection.

Promote exactly one full pass only if the pilot's hashes, source identities,
finite scores, interval containment, and output schema pass, and its measured
geometry throughput plus a conservative 120-second sorting allowance fit the
240-second cap. The prior full cube check supplies an additional runtime
reference. Otherwise stop; do not resize or retry automatically.

Stop after these checks, regardless of their direction. In particular, a
printed-direction exclusion does not promote training that cannot beat the
bound. If an alternative remains open, discuss the smallest further question
before fitting. A head swap on Softmax-trained weights alone is not an honest
test of a model trained with Sigmoid: Softmax is invariant to a shared logit
offset, whereas Sigmoid is not. No full retraining, tuning, additional seed,
publication of new outcomes, or broader search follows automatically.

Save code/contract/input hashes, wall times, full counts, pilot and full
records, and failures. Update the internal finding and explanation register;
discuss new results before changing the public report again.
