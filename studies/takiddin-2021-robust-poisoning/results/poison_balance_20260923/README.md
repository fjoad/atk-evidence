# Poisoning before balancing does not rescue this forest pilot

**Date:** 2026-09-23. **Scope:** one controlled forest fit, one seed, twenty
customers; not a full-paper reproduction or a population-level test.

Moving the label corruption before training-only ADASYN restores roughly
equal observed class proportions, but does not increase default detection.
Ranking and detection at matched false-alarm limits are worse in this pilot.
This is an effect of the whole resampling policy, not an isolated class-prior
effect or evidence of which implementation the authors used.

## What was held fixed

The [contract](../../POISON_BALANCE_CHECK.md) was frozen at
`9b23b329aaefe76c9a1559691366ecdb424acd3f`. A startup-only version-lookup
repair, described below, was frozen at
`579a3f5cafc0f8943f230ff7458a1c14103a6718`. No scientific settings changed.

B is the [earlier training-only forest control](../split_control_20260920/README.md),
not the original pre-split-synthesis forest A. Its clean and poisoned models
and scores were reused without fitting again. D first corrupts the same 675
original attack labels from the same six customers, then applies stock ADASYN
to the observed training labels. Both arms use the same 2,632 original
training rows, attack vectors, customer selection, forest parameters and seed
20260920. D alone adds one new 100-tree fit.

Every comparison below uses the same 1,288 original test rows in the same
order: 1,105 attacks and 183 benign examples. There are no synthetic test rows.
Before the new fit, all 28 zero-poison prepared arrays matched saved B-p00
exactly. The test raw values and labels remain fixed; standardized test values
can change with the training-fitted scaler. Training/test source-day and
customer dependence remain; no new grouping or unseen-customer split was added.

## Results

All figures are percentages; accuracy is ordinary accuracy. The first row is
clean reference context, not an additional fit.

| Preparation | DR | FA | SP | PR | ACC | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B, no poison (saved) | 93.30317 | 28.41530 | 71.58470 | 95.19852 | 90.21739 | 94.24132 | 92.00949 |
| B, balance then poison (saved p30) | 63.89140 | 8.74317 | 91.25683 | 97.78393 | 67.77950 | 77.28517 | 84.37752 |
| D, poison then balance (new p30) | 63.16742 | 12.56831 | 87.43169 | 96.80999 | 66.61491 | 76.45126 | 79.80343 |

D minus B at p30 changes detection by -0.72398 points, false alarms by
+3.82514 points, and AUC by -4.57409 points. Counts are B: TP706/FN399/FP16/TN167;
D: TP698/FN407/FP23/TN160. D misses eight more attacks and flags seven more
benign rows at the default decision rule.

| Saved-score constraint | B-p30 best DR | Actual FA | D-p30 best DR | Actual FA | D minus B DR |
|---|---:|---:|---:|---:|---:|
| FA <= 17.6% | 75.65611 | 17.48634 | 68.32579 | 17.48634 | -7.33032 |
| FA <= 33.3% | 85.42986 | 32.78689 | 75.56561 | 32.24044 | -9.86425 |

Thus the change is not only a shift in default cutoff behavior. These best
cutoffs inspect the test labels; they are diagnostics, not calibrated decisions
validated on future observations. No significance or equivalence is claimed.
With 183 benign rows, one extra false alarm moves FA by about 0.54645 points.

Useful ranking still remains: D's AUC 79.80343 exceeds the unchanged
negative-daily-mean control's 65.43382 and constant score's 50. Its per-attack default
detection is 54.21053/56.49718/78.01047/62.57310/67.19577/59.89305 for attacks 1-6.
Those mixed changes are preserved, not summarized as every attack worsening.

## What changed in training

| Quantity | B-p30 | D-p30 |
|---|---:|---:|
| Original training rows | 2,632 | 2,632 |
| Synthetic training rows | 1,900 | 611 |
| Total training rows | 4,532 | 3,243 |
| Observed benign labels | 2,952 | 1,663 |
| Observed attack labels | 1,580 | 1,580 |
| Observed attack proportion | 34.86320% | 48.72032% |
| Direct original attack-label flips | 675 | 675 |

The sampler requested equal classes and produced 611 rows after its normal
rounded allocations; none were trimmed or replaced. Both parents of every
synthetic row are observed-benign original training rows. Of D's 611 synthetic
rows, 157 have two truly benign parents, 132 have one truly malicious parent,
and 322 have two. The last 454 have unknown true class/attack ID (-1), remain
training-only, and are never treated as clean test truth.

The 675 direct flips are 25.64590% of 2,632 originals, or 20.81406% of D's final
3,243 training rows. Neither denominator is the nominal 30% of selected
customers. Unknown-truth synthetic descendants are not silently counted as
known additional poisoned labels. Observed-label training accuracy is 99.87666%;
all-row true-label training accuracy is null because 454 truths are unknown.

Order changes synthetic geometry/counts, forest bootstrap samples and fitted
scale as well as observed proportions. This comparison does not identify a
class-prior-only causal effect, including effects that might offset each other.

## Execution, failure and verification

Job 402290, frozen 9b23b32, passed the ten existing software fixtures but failed
in the version preflight: NumPy was imported as `np`, while a lookup requested
`numpy`. It stopped before research-input loading, directory creation,
preparation or fitting. Its 57-second failure and log are preserved, not erased.
The repair changes only that lookup, with two added preflight regression tests.
The scientific contract, primary implementation and data/model settings are
unchanged. The old frozen revision remains in Git.

Job 402291, repaired freeze 579a3f5, completed 0:0 in 20 seconds on
`crimv3srv024`, four CPUs/16 GiB/no GPU. All 12 fixtures and the exact zero-poison
guard passed. The program took 4.75547 seconds, preparation 0.39657, fit 0.74316,
scoring 0.22883 and reload 0.17560. Process peak RSS was 242,148 KiB; Slurm sampled
only 3,388 KiB, so its coarse sampled value is not the process peak.

The resumed submission requested 14:03 to reserve the first 57 seconds inside
a 15-minute total. **Slurm recorded a 15:00 limit**, not 14:03. Actual combined
allocation time was 77 seconds, well below 15 minutes, but a cumulative 15-minute
hard limit was not enforced by that request. Both scheduler attempts are reported.
There was exactly one research-data fit and no outcome-driven retry.

All 28 new input arrays, both reference results and their 56 input arrays verify.
Provenance/interpolations, unchanged original labels and evaluation, actual
training scaling/counts, model settings, save/reload scores, all metrics and
matched comparisons pass. The local read-only artifact audit matches the
cluster audit byte-for-byte. No local research-data model inference occurred.
The main test suite passes 339 tests with 10 isolated-environment skips; all 12
targeted fixtures passed locally and on the compute node.

## Decision

Stop this diagnostic. The proposed simple rescue by changing order was not
observed; the policy instead worsened ranking in this matched pilot. We have
not shown that changing observed proportions alone cannot matter, that every
ordering or configuration fails, or that this resolves the paper's results.
The unchanged daily-mean comparison and surviving AUC also rule out describing
this as total loss of useful signal.

Return to model coverage with a separately specified GRU pilot as the next
proposed step. Resolve its sequence shape, activations/output/loss and training
defaults before any run. No GRU implementation, fit, extra seed, larger study,
or publication was started here. SVM parameter sensitivity, full population,
other poison levels and the proposed reconstruction mechanism remain open.

## Preserved files

[Complete result and comparisons](result.json), [artifact audit](artifact_audit.json),
[execution record](execution.json), [failed preflight](preflight_failure.json),
[first job log](slurm-402290.out), and [completed job log](slurm-402291.out).
Result SHA-256: `e65e397c387263299f0f0e451d5e7c5d1640dbf969c6fa64965bebf6d1004aea`.
Models, raw observations, identities and score arrays remain ignored under
`data/derived/takiddin-2021-robust-poisoning/poison-balance-20260923-attempt2`.
The result and audit JSONs are byte-preserving copies, not regenerated summaries.
