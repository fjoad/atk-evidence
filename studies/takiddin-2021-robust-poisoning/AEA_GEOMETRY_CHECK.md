# Proposed zero-fit AEA input and score-geometry check

This is the next proposed bounded experiment, not a submitted job. Source
specification and constructed tests are authorized; real-data scoring under
this contract awaits approval. Do not train a model or regenerate inputs.

## Named question

Under the standardized-input, Sigmoid-reconstruction and MSE interpretation
in AEA_SPECIFICATION.md, does the output range already constrain the printed
0.51 operating point, and do simple untrained scores explain substantial
detection before crediting attention/reconstruction?

Competing outcomes: range-forced false alarms can exceed the printed cap;
range bounds can be inconclusive; or simple scores can already discriminate
well. None alone identifies author intent, attention's causal contribution,
or another output/error interpretation's performance.

## Inputs and scope

Use only the existing generalized novelty p00/p30 cases in
`setup-20260920-attempt1`, seed20260920,20 customers,28 days:

- p00 metadata SHA256
  `dfb6df1a5e5223569682f0066e8d1a2cb4a0a33aa3c13e7974becd463e2f4367`;
- p30 metadata SHA256
  `18c4527896cf2aaba5591b9dbf2b9a869f04788e955316e869f68c3d56c6e449`.

Before scoring, validate every consumed array's recorded hash/shape/dtype,
373 training and6704 test rows, observed labels all zero, true contamination
0/108 rows, six p30 customers, test counts3344 benign/3360 malicious,
and exact train/test row overlaps0/108. Verify matching raw test values,
identities and labels across levels. Recompute the stored scaler transform
as an artifact check only; do not refit it. Scaled inputs may differ between
levels because the scaler was trained on different inputs. Preserve the
3,157 synthetic benign test rows and report original-only strata too.

The duplicated p30 attack rows and shared source days are deliberate retained
consequences of the frozen preparation, not new clean validation data. Treat
the novelty preparation as its own experiment; do not use the classifier
test set or silently fix this dependence.

## Measurements, frozen before outcomes

On training and test rows, summarize negative/>1 coordinates and per-row
MSE box bounds L/U from checks/aea_geometry.py. On test rows, for each
class and original/synthetic stratum count guaranteed alarms (L>tau),
guaranteed nonalarms (U<=tau), and unresolved rows. Use float64 arithmetic,
and separately report a1e-6 comparison guard for near-boundary quantities.

Report tau0.51 and the favorable two-decimal-rounding interval[0.505,0.515].
A range-forced-FA exclusion robust to this rounding requires L>0.515 plus
the numerical guard; do not select the favorable side after seeing outcomes.
Also allow favorable rounding of the printed percentage metrics: one-decimal
FA5.2/18.4 permits upper limits5.25/18.45, and DR94.1/80.8 permits lower
limits94.05/80.75. Report both literal and rounding-favorable comparisons;
do not describe a rounding-sized discrepancy as an exclusion.
Check score-unit alternatives RMSE=sqrt(MSE) and SSE=48*MSE separately at
their printed0.51 threshold, clearly distinct interpretations. Do not infer
an unspecified MAE result from MSE bounds or choose the best interpretation.

For the primary MSE completion, calculate these explicitly untrained scores:

- constant reconstruction0 (mean squared input magnitude);
- constant reconstruction0.5;
- clipped-input reconstruction (the lower bound itself, an oracle of output
  range that has direct input access, not a trained attention model);
- negative mean raw daily consumption, the existing no-parameter comparison.

Preserve scores, identities, true labels and all results. Report seven metrics
for reconstruction scores at0.51, AUC and exhaustive cutoff/reversal checks
at common FA caps5.2/18.4%, plus original/synthetic strata. Daily-mean has no
paper-defined0.51 threshold: report only AUC/cutoff diagnostics for it.
Bounds are certificates/relaxations, not predictions or a network guaranteed
to attain every row's allowed endpoint simultaneously. AUC of L is a baseline
score's AUC, not an upper bound on every model's AUC.

Use the printed AEA p00/p30 context DR94.1/80.8 and FA5.2/18.4. No independent-
row confidence intervals from this dependent pilot, no full-population claim,
no neural parameter, seed or threshold optimization trial.

## Compute and stopping

One approved CPU job only:10 minutes,1 CPU,8 GiB,no GPU,zero model fits.
Numerical threads1. Research-data scoring stays on a Panther compute node;
local work is constructed fixtures and saved-artifact verification only.
Freeze the actual runner revision before submission and inspect scheduler/
output state to prevent duplicates. Stop on any input/provenance violation;
preserve failed/partial output. No automatic resampling repair or retry.

After the single check, audit identities, bound arithmetic, metrics and
original-file nonmutation. Discuss whether the next discriminating step is
AEA implementation fixtures, an output-range control, or clarification of
score/poisoning semantics. Do not automatically fit the AEA or ensemble.
No website edits/publication are included.
