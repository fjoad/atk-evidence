# First sigmoid-kernel SVM pilot

**Date:** 2026-09-21. **Scope:** one implementation, original twenty-customer
pilot, two poisoning levels. Not full Table III reproduction or a claim about
all SVM parameters.

Both fits completed and passed artifact checks, but their ranking is weak.
No threshold or favorable score reversal on either saved score vector reaches
its corresponding printed detection/false-alarm corner. The mismatch survives
the cutoff explanation for these fitted models and this sampled population.

## Frozen setup

[SVM_PILOT.md](../../SVM_PILOT.md) and code were frozen at
e69817379a28752c909ac4fc45b15e380dbed1c4 before fitting. The paper states
C=1 and a sigmoid kernel (III-C, p.2679); Table III is on p.2681.
The initial completion uses stock sklearn 1.9.0 SVC, gamma='scale', coef0=0,
tol=.001, no class weights, no probability calibration, and the other explicit
defaults recorded in the contract. Use native labels and raw decision scores,
not invented probabilities. Seed 20260920 is recorded but is not an active
randomness source with probability calibration disabled.

Inputs are the unchanged original p00/p30 arrays: 4,464 training rows,
2,232 test rows, 48 features, twenty customers and 28 complete days each.
At p30, the same six customers supply 675 malicious labels changed to benign
(15.12097% of all training rows). No attacks, scaling, ADASYN, or split were
regenerated. The known pre-split synthesis and source-day dependence remain.

## All seven metrics

Percentages. Published values describe a different, larger population;
comparison is diagnostic context, not a full-population reproduction test.

| Metric | Paper p00 | Pilot p00 | Paper p30 | Pilot p30 |
|---|---:|---:|---:|---:|
| DR | 89.2 | 62.08145 | 73.7 | 39.45701 |
| FA | 10.2 | 39.39663 | 25.7 | 31.14463 |
| SP | 89.8 | 60.60337 | 74.3 | 68.85537 |
| PR | 89.0 | 60.70796 | 74.0 | 55.40025 |
| ACC | 89.5 | 61.33513 | 74.0 | 54.30108 |
| F1 | 89.1 | 61.38702 | 73.8 | 46.08879 |
| AUC | 89.5 | 65.64194 | 74.0 | 63.38126 |

ACC is ordinary accuracy. TP/FN/FP/TN are 686/419/444/683 at p00 and
436/669/351/776 at p30. Original-only AUC is 66.56479/63.89635.
Original benign FA is 37.70492/30.60109% (69/183 and 56/183), compared with
synthetic FA 39.72458/31.25% (375/944 and 295/944).

The no-learning negative-daily-mean control has AUC 66.19624% on this same
full test set, slightly above unpoisoned SVM and above poisoned SVM. This is
an observed comparison, not an equivalence/significance claim. A constant
score has AUC 50%. Original forest and AdaBoost AUCs were 98.55/94.36 and
90.71/83.74 respectively on the same test rows; weak SVM performance is
not a finding that this entire prepared task lacks learnable signal.

## Every cutoff and score reversal

Each model produces 2,223 distinct test margins; all 2,224 cutoff boundaries,
including no alarms, were inspected in both directions.

| False-alarm cap | p00 best DR | Actual FA | p30 best DR | Actual FA |
|---|---:|---:|---:|---:|
| 10.2% | 19.36652 | 10.11535 | 20.27149 | 10.11535 |
| 17.6% | 27.42081 | 17.56877 | 25.97285 | 17.03638 |
| 25.7% | 35.83710 | 25.64330 | 33.48416 | 25.64330 |
| 33.3% | 51.31222 | 33.18545 | 46.06335 | 33.18545 |

At the corresponding printed corners, p00 misses 89.2% DR by 69.83348
points and p30 misses 73.7% by 40.21584 points. Favorable reversal reaches
only 1.90045% at p00's 10.2% FA cap and 5.70136% at p30's 25.7% cap;
it does not rescue either corner. Best balanced accuracy over all cutoffs is
65.10453/63.36986 in the native direction and 51.92294/52.10570 reversed.

Poisoning lowers default DR by 22.62443 points while AUC falls 2.26068.
Threshold shifts still explain part of the *within-SVM poisoning change*, but
the absolute performance is weak even before poisoning and the reported
corners remain unattainable by threshold changes on these fixed scores.
These test-selected thresholds are diagnostic limits, not calibrated deployment
decisions. The limit says nothing about scores learned with other parameters.

## Fit diagnostics

Both fits report fit_status_=0, with 1,168/1,115 solver iterations. They use
1,689/1,901 support vectors (844+845 and 950+951). Training accuracy against
observed labels is 63.32885/59.72222%; against true labels it is 63.32885/
55.48835%. Poor behavior therefore already appears on training rows; it is
not merely a large train/test generalization gap.

The numeric gamma is 0.020833333524383626 in both fits, with training
variance 0.9999999908295861. The older gamma='auto' value would be
1/48=0.020833333333333332, a relative difference of about 9.17e-9.
Those values are nearly identical here, but this does not prove refitting
with the other spelling would produce identical results.

Intercepts are -13.78551846/-21.36272143. Test margins range from
-99.80037 to 317.36957 at p00 and -73.62863 to 287.99039 at p30;
there are no zero-score test ties. Raw scores are finite and varied, native
predictions agree with their orientation, and reload preserves every margin
and label exactly. The positive and corrupted-label fixtures pass.

Solver success is not proof of a globally best sigmoid-kernel solution, nor
does it identify the missing source settings. The kernel parameters and their
effect on the fitted solution remain a concrete unresolved explanation.

## Execution, verification, and preserved mistakes

CPU job 398709 completed 0:0 in 2 minutes, within 15 minutes/4 CPUs/16 GiB,
no GPU. Fits took 5.01877/5.10622 seconds; scoring all training/test rows
took 8.37490/8.67055 seconds; reload and test rescoring took 2.89422/3.21766
seconds. Process peak RSS was 221,816/222,396 KiB; Slurm sampled 224,700 KiB
for the batch. These small-pilot costs do not assess the printed one-hour
full-data claim.

Before freezing, 314 repository tests passed and four AdaBoost fit tests were
skipped in the main environment; all seven AdaBoost tests passed in their
separate pinned environment. All seven SVM tests passed locally and again on
the compute node. The deliberately max_iter=1 fixture emits a convergence
warning; neither research-data fit does. Their recorded warnings concern
the probability-parameter and NumPy/joblib deprecations, not failed convergence.

The first full suite exposed an old provenance-test mismatch: yesterday's
corrected METHOD.md no longer had the document hash in the September 2 source
audit. Every arithmetic result still agreed. The test now verifies the original
and corrected documents at their frozen Git revisions while requiring exact
agreement on all other hashes and calculations. Neither historical evidence
nor the corrected note was overwritten; the rerun passed before SVM fitting.

The cluster and local audits verified 20 consumed input arrays, both metadata
hashes, unchanged paired features/test identities, 675 changed labels, saved
predictions bound to the correct labels, output/source hashes, and every
metric/cutoff calculation. Both the local comparison and complete pair-audit
output match their cluster files byte for byte. Historical forest/AdaBoost
audits also remain exact matches. No experimental retry or settings change ran.

## Next decision and limits

Stop this pair. The next proposed check is a bounded, **read-only diagnostic
of the existing sigmoid-kernel solution**, before another fit: reconstruct its
margins from support vectors/dual coefficients, and examine the kernel on a
fixed small training subset for conditioning and negative eigenvalues. This
can distinguish a scoring/implementation discrepancy from properties of the
declared kernel setup. It would not by itself establish the cause of poor
performance or rule out other gamma/coef0 values. Specify its sampling rule,
checks, numerical tolerances, budget, and stopping before running it.

This question is motivated by the [LIBSVM authors' FAQ](https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html):
sigmoid kernel matrices need not be positive definite. That general warning
is a reason to inspect this setup, not evidence that we have measured its
kernel spectrum or established the cause of this result.

No kernel-geometry measurement, alternative kernel/parameter, extra seed,
larger sample, or next model was run here. Other sigmoid settings, software
versions, full data, statistical replication, and the paper's proposed models
remain outside this result. Do not turn a fixed-score exclusion on one pilot
into an all-SVM impossibility or an inference about author intent.

## Records

[p00](p00.json), [p30](p30.json), [cluster comparison](cluster_comparison.json),
[artifact audit](artifact_audit.json), [execution](execution.json), and
[original job output](slurm-398709.out). Raw arrays and fitted models remain
in the ignored svm-pilot-20260921-attempt1 directory.

To repeat verification without fitting, use checks/verify_baseline_pair.py
with --preparation pointing to the original setup attempt and --attempt to
the SVM attempt, in requirements-svm.txt's environment. Its stdout must match
artifact_audit.json. reproduction/analyze_results.py with p00/p30 directories
must likewise reproduce cluster_comparison.json.
