# Which assumptions carry the limit?

Date: 2026-08-31

Status: checks complete; discussed with the user and approved for publication

The original stopping point below records the completed round. After discussion,
the user approved publishing these findings, followed by a separately recorded,
quick Sigmoid investigation. That follow-up does not alter these records or
authorize publishing its new results before discussion.

## Short answer

The paper really does specify Softmax output and high-error MSE scoring for
FC-SAE. Those were not arbitrary choices introduced by our implementation.
The normalization recipe is less precise. Two additional interpretations of
that recipe still cannot reach the reported detection rate at the printed
cutoff, on the fixed attack population.

Changing the allowed output range has a much larger effect on the optimistic
limits. But that control is not the final FC-SAE model in Table I, and it does
not reproduce the paper. No model was trained in this round.

## What the paper says, and what remains a choice

The complete source was re-read. Printed pp. 4109, 4114, 4115, and 4116 were
visually checked, including Table I's FC-SAE Softmax entry and Table III's
81% detection / 15% false-alarm target. The
[frozen source map](SOURCE_ASSUMPTION_CHECK.md) gives code locations and scope.

- **VERIFIED:** Softmax is stated both in Table I and adjacent prose. MSE and
  high-error anomaly direction are explicit on pp. 4109 and 4114. The final
  FC-SAE threshold is explicitly 0.58 on p. 4115.
- **VERIFIED omission:** standardization to zero mean and unit variance is
  stated before the split, but its fitted statistics and axis are not given.
- **INTERPRETATIONS TESTED:** existing joint feature-wise scaling; joint scalar
  scaling over all readings; and the weaker reading that normalizes B and the
  pooled six-attack M separately by feature. The last uses class information
  and is not endorsed as deployable preprocessing or claimed author behavior.
- **CONTROL, NOT FINAL-MODEL REPRODUCTION:** Sigmoid appears among the paper's
  searched output choices, but Table I selects Softmax for FC-SAE. Replacing
  the simplex by [0,1]^48 tests output-range sensitivity only, not LSTM or a
  learned Sigmoid model.

## Detection at the printed cutoff

Each row below grants every attack its most favorable permitted
reconstruction, using the label. These are upper limits, not trained-model
results. Limits are rounded upward.

| Preparation and output range | Maximum detection at MSE > 0.58 | Can reach the reported 81% detection? |
|---|---:|---|
| Current joint feature-wise scaling + Softmax | 29.58% | No |
| Joint scalar scaling + Softmax | 29.81% | No |
| Separate-class feature-wise scaling + Softmax | 33.96% | No |
| Current scaling + Sigmoid-range control | 95.73% | Not excluded by detection alone |

Both additional Softmax readings remain below even the rounding-tolerant
80.5% target. This is not seed variability or a statistical extrapolation:
the reconstruction range itself limits every possible set of weights under
the stated input and scoring assumptions.

**Why this particular result survives benign resampling:** the check uses all
4,504,602 original attack rows, not a sample. Adding synthetic benign examples
cannot change the fraction of attacks detected. The conclusion therefore
survives any benign-only ADASYN regeneration with those same attacks and the
printed 0.58 cutoff. It does not cover different attack populations or cutoffs
under the alternative normalizations.

## Original-row all-cutoff diagnostics

There are 750,767 original benign source days and six attack siblings for each,
totaling 5,255,369 rows. The alternative scalers were fitted using the full
pre-split original population, not this evaluation subset. We did not reuse or
regenerate synthetic benign rows under changed scaling: the neighbor geometry
would change. Consequently the table below is NOT a new Table III result or a
full post-ADASYN bound. Again, upper limits are rounded upward.

| Preparation/output | Maximum balanced ACC, any cutoff | Maximum AUC | Maximum DR with FA <=15% |
|---|---:|---:|---:|
| Current joint feature-wise / Softmax | 50.16% | 36.85% | 5.60% |
| Joint scalar / Softmax | 50.08% | 34.44% | 4.58% |
| Separate-class feature-wise / Softmax | 51.72% | 49.19% | 17.58% |
| Current joint feature-wise / Sigmoid control | 80.84% | 83.84% | 59.99% |

The larger output range raises the original-row balanced-accuracy ceiling from
about 50.16% to 80.84%. That is a large change caused by a different allowed
score range, not evidence of a trained-model improvement. Sigmoid still cannot
achieve the paper's combined detection/false-alarm operating point on this
original-row evaluation in the printed high-error direction. At 0.58, its
optimistic detection is 95.72275%, but its unavoidable original-benign FA is
41.11582% (both approximate). Passing the detection-only test is not passing
the full target. Its possible AUC above 81% likewise does not match all metrics.

The reversed-direction controls are preserved too:

| Preparation/output | Maximum balanced ACC | Maximum AUC | Maximum DR with FA <=15% |
|---|---:|---:|---:|
| Current joint feature-wise / Softmax | 69.87% | 74.09% | 43.06% |
| Joint scalar / Softmax | 70.80% | 75.50% | 45.11% |
| Separate-class feature-wise / Softmax | 63.41% | 65.48% | 41.60% |
| Current joint feature-wise / Sigmoid control | 94.71% | 96.01% | 94.04% |

Sigmoid plus reversal does not exclude DR >=81% with FA <=15% in the
label-aware relaxation. This is important counterevidence to a universal
claim about all bounded-output detectors. It changes two explicit FC-SAE
choices, remains only a relaxation, and does not demonstrate attainable
performance of any shared trainable model.

## What did not require another data experiment

Replacing mean squared error with summed squared error multiplies every score
by 48; taking its square root gives RMSE. Both are strictly increasing on
nonnegative scores. Rankings and the all-cutoff ROC region are unchanged.
Thus mere error-unit changes cannot escape the previous bound on the complete
fixed post-ADASYN evaluation. Fixtures verify ties and corresponding cutoffs.
This does not cover a genuinely different score, weighting, or reduction unit.

## Three separate conclusions

- **Numerical:** unchanged. No new trained reproduction was performed.
- **Mechanism:** unchanged. No recurrence/attention or matched learning test
  was performed. Output-range sensitivity is not a learned-mechanism finding.
- **Attainability:** the fixed-cutoff detection shortfall now covers these
  three explicit normalization readings on the same attack population. The
  all-cutoff alternative results cover original rows only. Other populations,
  normalization scopes, attack completions, and scoring procedures remain open.

The source review did not reveal that we accidentally invented the two main
FC-SAE operations responsible for the bound. It also did not establish that
every reasonable interpretation fails, or identify what the authors ran.

## Checks, runtime, and stopping

Code was frozen locally in `b76cb02ccffd27339f537d4859851f87deed1487` after
212 deterministic tests passed. The same code and frozen contract were
transferred directly to the cluster; nothing was pushed to GitHub.

CPU job `385119` completed `0:0`. The 448-row pilot took 8.20 seconds; the full
analysis took 56.76 seconds, including 38.42 seconds for geometry. Allocation
time, including manual inspection, was 3:18. The full process's maximum RSS
was 2,399,932 KiB. Every consumed artifact hash and source identity passed;
256 benign round-trip checks had maximum error 2.53e-7. The previous complete
original-row reference bounds were reproduced within the frozen 1e-7 tolerance.
Bounds use outward-padded float64, not certified interval arithmetic.

Post-run verification passed all 214 deterministic tests (140 study, 74 root),
including eight source-assumption checks. Local documentation links resolve
and `git diff --check` passes. README, website, and reproduction files remain
unchanged from the pre-round commit `149cde7`. The additional tests validate
saved-result provenance and conservative display rounding; they did not
change the frozen scientific code or contract.

[All saved records](results/source_assumption_20260831/) include the pilot,
full results, execution details, source/code/contract hashes, and fitted
normalization statistics. Original data, weights, scores, and previous results
are unchanged. Both printed and reversed outcomes are retained.

**Stop here for discussion.** No website, README, or report change; no push;
no training, new seed, or another branch. The next decision is whether these
source-supported extensions are sufficient for the current finding or whether
a specifically identified remaining assumption warrants another cheap test.
