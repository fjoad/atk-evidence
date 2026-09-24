# AEA score/scale repairs: finite zero-fit follow-up

Recorded before new repair outcomes. User approval follows the prior fixed-
cutoff exclusion and covers this bounded no-neural-training comparison.
The original contracts, prepared arrays and result remain immutable.

## Question and source

Could the observed obstruction be removed by a cutoff/score-unit explanation,
an alternative error norm or a small scaling correction? A removed bound is
not evidence that the reported neural result is reproduced.

The target was previously read completely. Pages2677 (III-A.1) and2680
(TableII, III-D.2(b)) were visually rechecked: training is standardized and
the same scale applied to testing; the normalization axis and AEA error norm/
units are not fully defined; Sigmoid output and threshold0.51 are printed.
Reference[23]'s [0,1] scaling was recorded in AEA_SPECIFICATION.md; its model
and training procedure are not imported here. No new source PDF was altered.

## Finite alternatives, not a Cartesian sweep

Each numerical branch preserves raw rows, split, contamination and test labels.
Reconstruction remains coordinatewise [0,1] in model-input space.

| ID | Track | Single change from the earlier MSE reading |
|---|---|---|
| feature_z_mse | I/A, baseline | Original featurewise-standardized arrays and MSE; now allow every cutoff. |
| feature_z_mae | I/A | Mean absolute error, since AEA reconstruction error has no explicit formula. |
| raw_unit_mse | I/A | Same model inputs/output range; inverse-transform the reconstruction before calculating MSE against raw readings. Raw reconstruction lies in [stored_mean,stored_mean+stored_scale] coordinatewise. This tests score units, not a new model. |
| global_z_mse | I/A | One scalar training mean and population SD across all373x48 observed-benign training values, instead of48 feature statistics. The unspecified scaling axis is the question; it is not asserted to be the authors' choice. |
| minmax_mse | C/A | Per-feature training-only min/max scaling to[0,1], not the target's explicit zero-mean/unit-variance scaling. Test values are NOT clipped; values outside training extrema remain outside[0,1]. |

For the two new input transforms: compute statistics in float64 from the
same observed-benign training inputs in each poisoning condition, then cast
transformed inputs to float32 like the original instrument. Zero SD/range
uses scale1 and is reported. Never use test extrema/statistics. New views and
parameters are separate artifacts; do not replace original files. There are
no model fits, although normalization statistics are computed from training.

Also report two limited diagnostics:

- Opposite error orientation for feature_z_mse only, explicitly contrary to
  the printed higher-error-is-malicious rule. Rebuild the optimistic bound for
  that orientation; do not simply negate the original oracle's scores.
- A linear/unbounded reconstruction head, analytically only: lower error0
  and no finite upper bound make this range-only argument inconclusive. This
  changes the selected Sigmoid head and is not a successful detector.

No combinations, extra scalers, per-profile normalization, arbitrary score
normalization, model seed, new data or performance-selected branch follows.

## Interval relaxation and proof

For each row, let [L_i,U_i] be the minimum/maximum possible reconstruction
error under its declared coordinate output box. For MSE use squared distance
to the box and the mean squared distance to the farther endpoint. For MAE
use the corresponding absolute distances. These are closed-box relaxations.

For higher-error detection define a label-informed score:

`z_i = L_i if benign; U_i if malicious`.

Every actual bounded reconstruction at any common threshold has at least
the oracle's false positives and at most its true positives. Each malicious/
benign pair's rank comparison is also at most the oracle's comparison, so
its AUC is an upper bound on any actual AUC under those fixed conditions.
This differs fundamentally from the prior clipped-input score, whose AUC
was NOT a model-AUC bound. The oracle deliberately uses true test labels,
ignores shared-network/input-identity coupling and is not deployable.

For reversed orientation the optimistic score is instead -U_i for benign
and -L_i for malicious. A pass means only not excluded by the relaxation;
it does not prove a realizable, learnable or generalizing model. Even identical
inputs with conflicting labels can appear separable to this loose oracle.

Enumerate all strict score>threshold partitions, moving ties together, and
report best DR upper bounds at FA caps5.2/18.4 and favorable caps5.25/18.45.
Report AUC upper bounds separately against printed94.0/80.3, allowing0.05
percentage points for rounding. Do not infer joint full-row reproduction
from separately passing necessary conditions. Compare numerical rates with
an additional1e-10 percentage-point allowance against false exclusions.

Repeat bound summaries with no guard and with expanded score intervals
`[max(L-1e-6,0),U+1e-6]` in each branch's units. Expanding the intervals
favours attainability. Fixed-cutoff counts use0.51 and the favorable rounding
endpoints0.505/0.515 as in the earlier check.

## Threshold and score-scale explanation

For a desired FA cap, allow `floor(cap*n_benign/100)` false positives. The
next largest benign lower bound is the minimum possible strict-error cutoff.
For a desired DR, require `ceil(DR*n_attack/100)` positives; that order
statistic of attack upper bounds is the exclusive maximum possible cutoff.
Record the resulting necessary interval [minimum,maximum), exact tie rules,
and whether it is empty. This is still a relaxation, not a chosen test cutoff
for deployment. Report each condition and their common-cutoff intersection.

An empty interval cannot be repaired by a positive constant score rescaling
or any strictly increasing scalar transformation: allowing all cutoffs already
covers those ranking-preserving changes (including RMSE, SSE and half-MSE).
If an interval survives, report necessary positive scale-factor intervals
mapping printed0.51 to an effective MSE cutoff; no factor is fitted or selected.
Restrict these factor calculations to the original MSE branch. Affine score
shifts are covered by the all-cutoff exclusion/not-exclusion, not a new search.

## Inputs, scope and budget

Reuse the exact novelty metadata identities from AEA_GEOMETRY_CHECK.md and
all56 input arrays. Prior geometry result must hash to
`d6e8b877ce31bd6ccaccbfa9ed478fad9587665782a9192590243f58834a42e5`;
its input/artifact audit must pass before the new calculations. Preserve
373train/6704test,0/108 contamination,108 exact p30 train/test overlaps,
raw test equality and original/synthetic strata. No dependence correction or
independent-row confidence interval. These remain20-customer pilot results.

One10-minute CPU job,1 CPU/8GiB/noGPU, numerical threads1, existing pinned
CPU environment. Real-data calculations stay on Panther compute; local work
is constructed fixtures and read-only saved-artifact audits. Save all views,
transform parameters, intervals, oracle scores, identities and outcomes in
ignored new output; commit nonrestricted summaries and hashes. Freeze code
before submission; inspect job/output state first. Preserve failures and
stop rather than automatically retry or expand alternatives.

Artifact replay checks integer identities/labels exactly and floating arrays
with rtol/atol1e-12, much smaller than the1e-6 score guard. Recompute report
metrics from saved interval arrays and require exact JSON agreement. This
allows tiny platform arithmetic differences without changing recorded scores.

After auditing, stop and identify the next discriminating question. No AEA
implementation/training, ensemble or website publication follows automatically.
