# First AdaBoost pilot

**Date:** 2026-09-20. **Scope:** the original twenty-customer, 28-day pilot,
one seed; not full Table III reproduction.

AdaBoost learns useful discrimination. Poisoning lowers both ranking quality
and default detection, but a substantial part of the default-detection decline
can be changed by moving the cutoff. The complete printed pattern is not
recovered in this pilot; neither is universal failure demonstrated.

## Source and frozen choices

[ADABOOST_PILOT.md](../../ADABOOST_PILOT.md) was frozen with code at
d47a6de4a60840b6806b409f88f47ac6a102a118 before fitting. The paper specifies
decision-tree weak learners and 50 estimators (pp.2678-2679); Table III is on
p.2681. A fresh visual check corrected our earlier source note: the tree
family is specified, though its depth is not.

The declared completion uses stock scikit-learn 1.5.2, SAMME.R, depth-one
trees, learning rate 1, and seed 20260920. This preserves the historical
default algorithm choice, not an identified author environment or bitwise
equivalence with a 2020 library. NumPy 1.26.4, SciPy 1.13.1, joblib 1.4.2,
and threadpoolctl 3.5.0 are pinned separately from the forest's environment.
The expected SAMME.R deprecation warning is preserved in each record.
SAMME and other tree depths remain untested alternatives, not silently tried.

No inputs were regenerated: 4,464 training and 2,232 test rows, 48 features,
the same prepared p00/p30 hashes as the forest. At p30, the same six customers
contribute 675 flipped training labels, 15.12097% of all training rows.
Pre-split ADASYN and its known dependence are retained in this original route.
The corrected forest controls are not evidence that corrected AdaBoost has
the same performance; that separate experiment has not been run.

## All seven metrics

Percentages. Paper values concern its full population and are context, not
matched-population targets for declaring this small pilot a reproduction.

| Metric | Paper p00 | Pilot p00 | Paper p30 | Pilot p30 |
|---|---:|---:|---:|---:|
| DR | 85.7 | 81.08597 | 70.1 | 46.42534 |
| FA | 14.1 | 15.17303 | 29.9 | 5.05768 |
| SP | 85.9 | 84.82697 | 70.1 | 94.94232 |
| PR | 85.3 | 83.97376 | 70.0 | 90.00000 |
| ACC | 85.8 | 82.97491 | 70.1 | 70.92294 |
| F1 | 85.5 | 82.50460 | 70.0 | 61.25373 |
| AUC | 85.0 | 90.70692 | 70.0 | 83.73956 |

ACC is ordinary accuracy. Default labels use native predict(), verified to
agree with probability argmax. TP/FN/FP/TN are 896/209/171/956 at p00 and
513/592/57/1070 at p30. The p30 accuracy is near the printed 70.1%, but
the other metrics do not match its operating point; one matching number is
not a reproduced row.

Both fits used all 50 trees, all depth one, without early stopping. Training
accuracy against observed labels was 86.31272/79.74910%; against true labels
it was 86.31272/73.31989%. The weak learners, errors, weights, and varied
scores are preserved, and positive/corrupted-label fixtures passed. This is
not a failed fit or a constant predictor.

## Ranking and cutoff effects

Default detection drops 34.66063 points; AUC drops 6.96736 points. On the
same saved scores, the best detection at each predeclared false-alarm cap is:

| FA cap | p00 DR | Actual p00 FA | p30 DR | Actual p30 FA |
|---|---:|---:|---:|---:|
| 14.1% | 80.72398 | 14.01952 | 67.33032 | 14.01952 |
| 17.6% | 82.53394 | 17.56877 | 71.40271 | 17.56877 |
| 29.9% | 88.77828 | 29.54747 | 80.45249 | 29.72493 |
| 33.3% | 90.22624 | 32.91925 | 82.98643 | 32.56433 |

The p30 scores can exceed the paper's 70.1% detection within its 29.9%
false-alarm allowance. Conversely, at p00's 14.1% cap, no cutoff on these
scores reaches the paper's 85.7% detection; the best is 80.72398%. These
are distinct observations about fixed scores and different sample populations,
not a claim that either full table row has been reproduced or excluded.
Reversing scores does not improve these operating points.

At the same 14.1% cap the poisoning-related DR decline is 13.39367 points;
at 29.9% it is 8.32579 points. Useful ranking survives, but genuine ranking
deterioration remains. Cutoffs were chosen using test labels as diagnostics;
there is no independently validated calibration result.

The constant-prior score has AUC 50%, and negative raw daily mean has
AUC 66.19624%. Original-only AUC is 88.73872/82.48770. Original benign
FA is 17.48634/6.01093% (32/183 and 11/183), compared with synthetic
FA 14.72458/4.87288% (139/944 and 46/944). The 1,288 original rows still
inherit training dependence; removing synthetic evaluation rows does not
undo pre-split synthesis.

On the identical full test rows the original forest had AUC 98.55119/94.36071,
above AdaBoost's 90.70692/83.73956. This reverses their printed AUC ordering
in this pilot, not necessarily at the full population or under other defaults.
The libraries differ (forest sklearn 1.9.0, AdaBoost sklearn 1.5.2); this is not a
controlled isolation of algorithm alone. Neither forest was refitted.

## Execution and audit

Experimental CPU job 398348 completed 0:0 in 14 seconds, with four CPUs,
16 GiB and no GPU, under a 15-minute limit. Fits took 1.494/1.503 seconds;
process peak RSS was 117,848/117,888 KiB. Slurm sampled only 3,388 KiB
for this short job, so that sample is not the Python memory peak. Pilot
timings do not test the paper's approximate one-hour full-data training claim.

Before freezing, the ordinary repository suite ran 311 tests: 307 passed and
four AdaBoost fit tests were skipped in the incompatible default environment.
All seven AdaBoost tests, including those four, passed in the isolated pinned
environment; the compute job repeated and passed all seven. Both fitted
models passed exact probability/prediction save/reload checks.

After transfer, all 20 consumed input-array hashes, both preparation metadata
hashes, unchanged paired features/test identities, 675 label changes, saved
prediction-to-input identities and labels, model/score hashes, and every
metric/cutoff diagnostic passed. The comparison recomputed in the local pinned
environment is byte-identical to the cluster comparison. The new read-only
pair checker was also applied successfully to the old forest artifacts;
the old matched-control audit still matches its preserved result.

Environment setup is separate: login-node venv creation was killed before
any research fit. CPU setup job 398338 then completed 0:0 in 67 seconds,
using one CPU/4 GiB with a 10-minute limit. No data fitting/scoring occurred in
that setup job. The experimental pair ran once, without a retry or settings
change after inspecting outcomes.

## Decision and next question

Stop the pair here. No extra seeds, deeper weak learners, alternate AdaBoost
algorithm, or resampling control follows automatically from the p30 detection
gap. Two shallow models now supply useful pilot signals, but all-model and
full-population reproduction remain incomplete.

The proposed next model is the paper's SVM (C=1, sigmoid kernel). Freeze its
omitted gamma/coefficient and decision-score choices before a bounded pair on
the same original data. This would test a distinct reported baseline, not
repeat AdaBoost hoping for a worse outcome. No SVM has been fitted here.

No confidence interval, universal bound, hidden author implementation, or
intent conclusion is inferred from one seed and twenty dependent customers.

## Preserved records and verification commands

[p00](p00.json), [p30](p30.json), [cluster comparison](cluster_comparison.json),
[artifact audit](artifact_audit.json), [execution record](execution.json),
[fit-job output](slurm-398348.out), and [environment-setup output](environment-398338.out).
Arrays, identities, and fitted models remain ignored in
data/derived/takiddin-2021-robust-poisoning/adaboost-pilot-20260920-attempt1.

Using an environment installed from requirements-adaboost.txt, the read-only
checker is checks/verify_baseline_pair.py with --preparation pointing to the
original setup attempt and --attempt to this AdaBoost attempt. The direct
reproduction/analyze_results.py command takes its p00 and p30 directories;
its stdout must match cluster_comparison.json. No new fitting is needed.
