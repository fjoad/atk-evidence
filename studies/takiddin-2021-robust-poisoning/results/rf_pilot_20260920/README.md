# First random-forest pilot

**Date:** 2026-09-20

**Scope:** exploratory behavior and instrument check on 20 customers and their
first 28 complete days. This is not full Table III reproduction.

Frozen code 48e979a ran once as CPU job 397217, completed 0:0 in 16 seconds.
It used four CPUs, no GPU, and a 16-GiB allocation. The 100-tree forest fit in
0.649 seconds at p00 and 0.566 seconds at p30. The paper's approximately
one-hour full-data claim cannot be assessed from these small fits.

## Primary result

Each fit used the same 4,464 input training rows and the same 2,232 evaluation
rows, with different observed training labels. Reported values below concern
the paper's larger population and are context only.

| Metric (%) | Paper p00 | Pilot p00 | Paper p30 | Pilot p30 |
|---|---:|---:|---:|---:|
| DR | 82.2 | 92.30769 | 66.8 | 61.17647 |
| FA | 17.6 | 3.01686 | 33.3 | 0.44366 |
| SP | 82.4 | 96.98314 | 66.7 | 99.55634 |
| PR | 82.1 | 96.77419 | 66.6 | 99.26579 |
| ACC | 82.3 | 94.66846 | 66.7 | 80.55556 |
| F1 | 82.1 | 94.48819 | 66.7 | 75.69989 |
| AUC | 81.4 | 98.55119 | 66.7 | 94.36071 |

ACC is ordinary accuracy. The default decision is scikit-learn's probability
argmax, with exact class ties assigned to 0. Training accuracy against the
observed labels was 99.978% and 99.933%. Under poison, accuracy against the
uncorrupted training labels was 84.946%, consistent with learning the noisy
labels rather than secretly fitting the clean ones.

## What the saved-score checks explain

Default detection falls 31.131 percentage points, while AUC falls 4.190 points
and false alarms decrease. The cutoff diagnostics establish that substantial
discrimination remains:

| False-alarm cap | p00 best detection | p30 best detection | Drop, points |
|---|---:|---:|---:|
| 17.6% | 97.28507% | 92.03620% | 5.24887 |
| 33.3% | 99.18552% | 95.02262% | 4.16290 |

The p30 cutoffs are 0.19 (actual FA 16.14907%) and 0.12 (actual FA 31.49956%).
All 119 p00 and 136 p30 ROC boundaries were considered, including the
no-alarm boundary. Test labels select these favorable diagnostic cutoffs.
They are not validation-selected deployment thresholds, new fits, or evidence
of the authors' actual cutoff rule.

The training-prior constant score has AUC 50%; negative raw daily mean has AUC
66.19624%. The forest supplies discrimination beyond those two controls.
Original-row AUC is also high: 97.79715% without poisoning, 92.38434% with it.

The evaluation population still matters. At p00, original benign FA is
6.55738% (12/183), versus 2.33051% on synthetic benign rows (22/944).
At p30 they are 2.18579% (4/183) and 0.10593% (1/944). Removing synthetic
test rows from the metric does not undo their influence on training, the
pre-split synthesis, or shared source-day/customer dependence.

## Interpretation and next question

The baseline works well on this particular construction. That weakens the
expectation of uniformly poor baseline behavior in the pilot; it does not
validate the paper's complete tables or identify the authors' procedure.
The default-cutoff decline alone cannot be treated as the loss of all useful
discrimination. Calibration/threshold effects remain material here.

The most useful next diagnostic is a matched check of the split/resampling
effect before treating the high pilot scores as independent generalization.
Additional seeds of this same construction would not answer that question.
All other reported models remain in the reproduction scope, but none was
silently fitted as part of this pair. No confirmatory interval, population
verdict, mechanism conclusion, or full-data time estimate is earned.

## Verification and files

- All 296 repository tests passed before code freeze.
- All ten baseline fixture tests passed on the compute node.
- Inputs passed their frozen metadata and consumed-array hashes.
- Each model was saved, loaded again, and produced exactly identical full
  test probabilities and predictions.
- Output model/score hashes and every metric/diagnostic were rechecked after
  transfer. Local and cluster comparison JSON files are byte-identical.
- An early fixture assertion compared equivalent repeating-decimal rates
  bit-for-bit. It was corrected before freezing to a 12-decimal-place
  comparison; the implementation was unchanged.

Records: [p00.json](p00.json), [p30.json](p30.json),
[cluster comparison](cluster_comparison.json),
[artifact audit](artifact_audit.json), [execution](execution.json), and
[original job output](slurm-397217.out).
Models, labels, identities, and probability arrays remain in the ignored
rf-pilot-20260920-attempt1 data directory.
