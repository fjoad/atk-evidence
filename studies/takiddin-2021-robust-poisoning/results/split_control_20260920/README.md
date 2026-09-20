# Matched resampling and source-day controls

**Date:** 2026-09-20. **Scope:** one 20-customer pilot, one seed; not full
Table III reproduction or a population-level statistical test.

Moving ADASYN into training lowers AUC and raises false alarms on the same
original test examples. Keeping entire source days out of training does not
produce an additional collapse in this pilot. Useful discrimination survives.
Both findings matter; neither establishes the authors' implementation or intent.

## What was fixed before observing results

The [contract](../../SPLIT_RESAMPLING_CHECK.md) was frozen at
e6e03599f9f6d5274c2c52121563d69d599aa403. Arm A reuses the first forest's
saved predictions without refitting. B preserves A's original row split but
generates synthetic training rows using training observations only. C instead
holds out complete source days, keeping all seven versions of each training
day together, then applies training-only synthesis. The 100-tree forest,
model settings, seed, existing attack vectors, and poisoning-customer selection
are unchanged. These are controls, not silently corrected reproductions.

The primary common evaluation was selected from identities, not scores:
445 original rows, comprising 386 attacks and 59 benign rows from 175 source
days and all 20 customers. They are the intersection of A's original test
rows with C's 187 held-out days. Every arm uses exactly these same rows and
true labels in the same order. No synthetic test rows are included. C is not
evaluated on the larger A/B test set because some of those days train C.

## Matched results

All figures below are percentages. ACC is ordinary accuracy; decisions use
the forest's native probability argmax. AUC measures ranking across cutoffs.

| Poison | Arm | DR | FA | SP | PR | ACC | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0% | A: saved pre-split synthesis | 92.48705 | 3.38983 | 96.61017 | 99.44290 | 93.03371 | 95.83893 | 98.31606 |
| 0% | B: training-only synthesis | 94.04145 | 33.89831 | 66.10169 | 94.77807 | 90.33708 | 94.40832 | 91.84157 |
| 0% | C: training-only, grouped days | 94.30052 | 32.20339 | 67.79661 | 95.03916 | 90.78652 | 94.66840 | 93.18082 |
| 30% | A: saved pre-split synthesis | 63.21244 | 1.69492 | 98.30508 | 99.59184 | 67.86517 | 77.33756 | 90.98094 |
| 30% | B: training-only synthesis | 63.73057 | 15.25424 | 84.74576 | 96.47059 | 66.51685 | 76.75507 | 81.98823 |
| 30% | C: training-only, grouped days | 62.43523 | 11.86441 | 88.13559 | 97.17742 | 65.84270 | 76.02524 | 82.97181 |

B minus A changes AUC by -6.47449 points without poisoning and -8.99271
points with poisoning. C minus B changes it by +1.33925 and +0.98358 points.
These are observed paired contrasts, not significant-difference tests or
proof that grouping never matters. Grouping also changes training membership
and synthetic geometry. There are only 59 benign evaluation rows: one false
alarm changes FA by about 1.69 percentage points.

For the larger, secondary A/B comparison, all 1,288 original test rows are
shared (1,105 attacks, 183 benign):

| Poison | A AUC | B AUC | A FA | B FA |
|---|---:|---:|---:|---:|
| 0% | 97.79715 | 92.00949 | 6.55738 | 28.41530 |
| 30% | 92.38434 | 84.37752 | 2.18579 | 8.74317 |

The larger comparison gives the same direction for the resampling effect:
AUC falls 5.78765/8.00682 points. Detection alone would obscure this change:
B's default DR increases slightly in both comparisons while false alarms rise.

## Keeping false-alarm constraints comparable

The best detection on the common saved scores at each declared FA cap is:

| Poison | Arm | DR at FA <= 17.6% | Actual FA | DR at FA <= 33.3% | Actual FA |
|---|---|---:|---:|---:|---:|
| 0% | A | 96.63212 | 13.55932 | 97.92746 | 32.20339 |
| 0% | B | 86.78756 | 16.94915 | 93.78238 | 32.20339 |
| 0% | C | 91.19171 | 15.25424 | 94.30052 | 32.20339 |
| 30% | A | 84.97409 | 16.94915 | 91.45078 | 28.81356 |
| 30% | B | 70.72539 | 16.94915 | 83.16062 | 30.50847 |
| 30% | C | 69.94819 | 16.94915 | 82.38342 | 32.20339 |

These cutoffs use the test labels and are diagnostics, not independently
calibrated deployment decisions. In C, the 17.6%-cap DR falls 21.24352
points with poisoning, versus 31.86528 points at the default decision.
Threshold behavior still matters, but meaningful ranking degradation remains:
C's AUC falls 10.20901 points. Reversal does not rescue these outcomes.
The zero-parameter negative-daily-mean score has AUC 67.62975% on the common
evaluation; a constant score has AUC 50%. C remains above both references.

## Actual training populations

ADASYN's rounded allocations were retained, not trimmed to manufacture equal
counts. Original and synthetic rows have separately recorded ancestry.

| Arm | Original training rows | Synthetic training rows | Total | True benign / attack | Labels flipped at p30 | Fraction of all training |
|---|---:|---:|---:|---|---:|---:|
| A | 2,632 | 1,832 | 4,464 | 2,209 / 2,255 | 675 | 15.12097% |
| B | 2,632 | 1,900 | 4,532 | 2,277 / 2,255 | 675 | 14.89409% |
| C | 2,611 | 1,927 | 4,538 | 2,300 / 2,238 | 696 | 15.33715% |

C trains on 373 complete source days. All arms select the same six poisoning
customers at nominal 30%; that is not 30% of all training labels. In B/C every
synthetic parent is an original benign training row and none is an evaluation
row. C has zero training/evaluation source-day overlap, including ancestry.
B still shares 528 source days with its full original test set.

## Execution and verification

CPU job 398164 completed 0:0 in 1:48, under the 15-minute, four-CPU,
16-GiB limit, with no GPU. Exactly four experimental forests were fitted;
reference A was never retrained. The model fits took 0.667, 0.583, 0.567,
and 0.614 seconds. The control program took 6.313 seconds; the entire
allocation also includes imports, fixtures, and startup. Process-reported
peak RSS was 248,240 KiB. This is not a full-data runtime comparison.

All 304 repository tests passed before freezing. The compute node repeated
all eight control fixtures. Cluster checks passed for 112 saved input arrays,
input/output hashes, finite values, training-fitted scaling, correct test
labels, matching evaluation identities/values, parent ancestry, and grouped
source-day exclusion. Each fitted model also passed exact save/reload
prediction and probability agreement. All comparisons were recomputed from
the saved scores.

After transfer, the same artifact audit passed locally without preparing data,
fitting models, or generating new scores. Its complete output matches the
cluster audit, including the saved summary's SHA-256. Public result JSONs are
byte-preserving copies of the cluster records.

Login and file transfer had transient failures before submission; one bundle
fetch was attempted before transfer completed. It was retried only after the
bundle passed verification. No experimental job or fit preceded job 398164,
and no outcome-dependent retry occurred.

## Decision and limits

The pre-split synthesis policy materially benefits this pilot's apparent
performance. The contrast also changes generated values and training counts,
so it is not an isolated estimate of leakage alone. Grouping days does not
produce the hypothesized further collapse in this one controlled comparison.
Both corrected arms still learn useful discrimination, including under poison.

We stop this diagnostic here: no automatic seed repeats or larger sweep.
The next distinct question is whether another reported shallow baseline has
the same poisoning/threshold behavior on the now-verified original pipeline.
An explicitly specified AdaBoost pair is a reasonable next model, after
resolving its library/version choices. No additional model is launched here.

Full-population results, all reported models, other seeds, unseen customers,
later time periods, the proposed reconstruction mechanism, and the paper's
training-time claims remain untested by this check. These sampled dependent
rows do not justify a population confidence interval or universal bound.

## Files

[Complete matched summary](summary.json), [cluster artifact audit](artifact_audit.json),
[execution record](execution.json), and [original job output](slurm-398164.out).
Each new fit has its complete result record: [B-p00](B-p00.json),
[B-p30](B-p30.json), [C-p00](C-p00.json), [C-p30](C-p30.json).
Per-attack results, all seven metrics, controls, cutoff/reversal diagnostics,
counts, and hashes remain in those machine-readable records. Raw observations,
identities, predictions, and fitted models remain in the ignored
split-control-20260920-attempt1 directory, not Git.
