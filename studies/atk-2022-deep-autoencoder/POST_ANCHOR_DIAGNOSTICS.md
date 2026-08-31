# Post-anchor checks: output limits and useful information

Date: 2026-08-31

Status: approved by the user; contract recorded before the new measurements

The user approved the three proposed next questions after reading the initial
report. This opens Checkpoint 2 for the bounded checks below, not another full
training run, seed sweep, model family, or paper table. The original numerical
finding and all its files remain unchanged.

## Questions and competing outcomes

1. **A — output-domain limit.** Can any allowed reconstruction, even selected
   separately for each example with knowledge of its label, reach the paper's
   Table III FC-SAE detection/false-alarm target on the frozen evaluation?
   Failure of this deliberately generous relaxation excludes the target for
   every weight/seed choice under the fixed input, output, and score assumptions.
   Success of the relaxation is inconclusive: shared, trainable model weights
   may not realize these independently selected reconstructions.
2. **M — incremental information.** Does the trained score materially improve
   on no-learning input magnitude and a valid constant Softmax reconstruction?
   A positive gain weakens “training adds nothing”; negligible gain within a
   stated margin supports only that limited comparison. Per-attack and
   similar-energy checks can reveal differences hidden by aggregate correlation.
3. **Conditional small controls.** Only if the first checks leave a question
   they cannot answer, record a capped positive-control sandbox that separates
   missing signal, objective/output restrictions, and optimization failure.
   No training is automatically promoted by completion of this contract.

These are source-relevant diagnostics on the previously frozen `P+I` inputs,
with non-paper oracle and baseline controls classified `C/A` and `C/M`.
They are not another numerical reproduction. The questions were motivated by
the observed anchor, so this is not an independent confirmation of a previously
unseen finding. New quantities below are fixed before their inspection.

## Fixed source and scope

- Source: Table I and Section IV-C, printed p. 4115 (Softmax output); Section
  II-C, p. 4109 (standardization); Equation (7), p. 4109 and the frozen MSE
  reduction; Table III, p. 4116 (targets). Complete paper and these passages
  were already visually verified for the source freeze and public report.
- Scientific code: `a88d17477ad96b01ffa44a50d8ce051dd8d2b5ca`.
- Attempt: `seed_20260824_2f483335536c`, model FC-SAE, seed `20260824`.
- Source result SHA-256:
  `ae07b42ef6c84242ca9b39db8b8828694d6d4df6859abdee090fc0a613a69154`.
- Primary population: all 8,884,989 post-ADASYN rows from that run.
- Secondary population: the original held-out customer days and their six
  attack derivatives, excluding synthetic benign rows. This is a diagnostic
  view, not a replacement reproduction score.
- No preparation, training, weights, saved scores, threshold, or source choice
  changes. Verify hashes of every consumed artifact; never write to it.

## Analytic bound

For a fixed 48-vector `x`, let the reconstruction `r` lie in the closed simplex
`r_j >= 0`, `sum(r_j) = 1`. This includes the Softmax interior and is therefore
an optimistic relaxation. The score is `s(x,r) = ||x-r||² / 48`.

The exact real-arithmetic range is `[L(x), U(x)]`, where

```
L(x) = ||x - projection_simplex(x)||² / 48
U(x) = (||x||² + 1 - 2 min_j(x_j)) / 48.
```

The minimum is Euclidean projection. For the maximum, any simplex point is a
convex combination of vertices. Convexity gives
`||x-r||² <= max_j ||x-e_j||²`; the farthest vertex has the smallest `x_j`.
Continuity on the connected simplex fills the intermediate score range.

With the printed rule `attack iff score > t`, the best possible detection at
cutoff `t` is `fraction_attack(U > t)`; the least possible false-alarm rate is
`fraction_benign(L > t)`. Equivalently assign attacks their upper endpoint and
benign examples their lower endpoint, then enumerate every distinct score
boundary. This one label-aware score vector simultaneously maximizes ranking
AUC and supplies the optimistic threshold region. Allowing different outputs
even for identical inputs only enlarges the feasible set.

Evaluate in float64, outward-pad each endpoint by
`epsilon = 1e-5 * (1 + U)` and clip the lower endpoint at zero. Report the padding,
validate the saved scores lie within the padded interval, and distinguish this
numerical evaluation of an analytic bound from an interval-arithmetic proof.
Test hand-sized minima/maxima, ties, and brute-force threshold enumeration.

Report, for both evaluation views:

- the optimistic DR and FA at the printed cutoff 0.58;
- maximum DR at FA <= 15%, and also at FA <= 15.5% allowing rounded targets;
- maximum balanced accuracy and ranking AUC over the relaxation;
- feasibility of DR >= 81%, FA <= 15%, and the rounding-tolerant
  DR >= 80.5%, FA <= 15.5% pair;
- a separately labeled reversed-direction relaxation, not a repair.

A failed pair alone is sufficient to exclude the complete reported row under
these assumptions. A passing pair does not establish that every metric matches.
No claim about another preprocessing, output domain, loss/score, or dataset is
licensed. The result says nothing about author intent.

## Useful-information measurements

- Compare saved trained scores with `E = mean(x²)`, the saved projection-floor
  control, and `mean((x - 1/48)²)`, a valid constant Softmax reconstruction.
- Keep cutoff 0.58 and high-error orientation fixed for these comparisons.
  Report balanced accuracy, DR, FA, AUC, and beneficial/harmful decision changes.
- Report original benign versus each attack separately; synthetic benign rows
  form a separate descriptive group.
- On original rows only, use 100 pooled energy-quantile bins with label-blind
  edges. Report within-bin AUCs of energy, trained score, and their difference,
  weighted by the number of positive/negative pairs in each bin. This checks
  ranking within energy bands; it is not an exact conditional-independence test.
- Also compare each attack's score to its own benign source day's score,
  retaining ties and reporting customer-cluster uncertainty.
- Primary useful-work uncertainty: 2,000 paired customer-cluster bootstrap
  resamples, analysis seed `20260831`, all days and attack siblings retained
  together. Calculate pooled balanced-accuracy differences on the original
  view and each attack; percentile 95% intervals. Keep the fitted model,
  scaling, attack generation, and split fixed. These intervals do not measure
  training-seed or preprocessing uncertainty, and assume exchangeable customer
  clusters under that conditioning.
- Predeclare +/-1 percentage point as the negligible balanced-accuracy region,
  matching the paper's reported precision. A whole 95% interval inside that
  region supports practical equivalence only for that comparison. This choice
  follows the already observed aggregate gain and is not retrospective
  preregistration or a universal definition of usefulness.
- No bootstrap on synthetic rows without their full generation dependencies.
  Their full-population metrics remain descriptive only.

## Execution and stopping

Use one short direct analysis file, fixtures, and at most two static figures.
No changes to the five reproduction files or shared experiment framework.

1. Local hand-checkable software fixtures only; no local experimental scoring.
2. One CPU compute allocation: four cores, at most 12 GiB and 20 minutes.
   No GPU, training, automatic retry, or heartbeat.
3. Pilot: 64 deterministic evenly spaced rows from each original block and the
   synthetic block (512 total), proving geometry, grouping, and output shape.
   Pilot findings are not full-population results.
4. Inspect the pilot. Promote one full saved-array pass only if interval and
   identity checks pass and measured throughput supports the remaining budget.
   Full pass uses streaming chunks of 32,768, sorted one-dimensional score
   vectors, and aggregated customer counts; no pairwise distance matrix.
5. Preserve pilot, full success/failure, code/contract hashes, runtime, and
   figures. Stop rather than expand if the budget or checks fail.
6. Interpret before further compute. If the generous output-domain bound
   already excludes the target, another training seed cannot change that bound
   and is not promoted. Any different scientific assumption remains a separate
   proposed control.

Outputs feed a dated finding, explanation-register updates, and the existing
public report. Preserve contrary evidence as plainly as a failure to reach the
target. No stronger conclusion is written before the measured outputs exist.
