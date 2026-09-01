# Paper-time LSTM-SAE finding

**Date:** 2026-09-02

**Status:** complete, transferred, independently audited, and stopped for
discussion before publication or another experiment.

**Evidence scope:** one faithful declared LSTM-SAE completion, one seed, the
full prepared ISET data, one V100-16GB, and exactly the paper's reported
183-minute training budget. This is direct numerical and bounded attainability
evidence. It is not a universal proof about every undocumented procedure or
unlimited training time.

## Question

Table IV reports 183 minutes for LSTM-SAE training on full ISET. What does the
written method attain when given that exact fitting time on one favorable,
period-appropriate V100?

The frozen answer used `CR-ISET-LSTMSAE-01`: 1,500,523 benign fitting profiles,
8,884,989 scored profiles, seed 20260824, batch 32, the printed architecture and
MSE score, and no retry, additional GPU, mixed precision, search, or
outcome-driven change. See `PAPER_TIME_BUDGET_CONTRACT.md`.

## Execution and integrity

Panther job `385632` ran frozen commit `46f0ddd` on one
`Tesla V100-PCIE-16GB` and completed `0:0` in 3:43:23 including startup,
verification, fitting, persistence, full scoring, and threshold enumeration.

- Fitting stopped after 10,980.464 seconds, only 0.464 seconds beyond the
  10,980-second boundary because the already-started batch was completed.
- 59,464 updates completed: one full 46,892-batch epoch and 12,572 batches
  (26.81%) of epoch two.
- Full epoch one took 8,632.961 seconds (143.883 minutes); the partial second
  epoch had the same approximately 0.18-second batch throughput.
- Loss was finite and changed from 1.372814 for the first complete epoch to
  1.301080 for the observed part of epoch two.
- Initialized and fitted weight digests differ; saved and freshly reloaded
  fitted-weight digests are identical.
- All 8,884,989 saved scores are finite `float32`, and their class counts,
  shape, checksum, confusion counts, and AUCs independently reproduce the
  recorded result.
- Every transferred artifact hash equals the hash written by the job.
- Post-result verification passes 140 study tests, 123 root tests, and the
  strict data gate.

The reload warning concerned absent optimizer-state variables in a fresh
scoring model. No further optimization followed reload, and the reloaded model
weight digest exactly matched the fitted digest.

## Numerical result

At the paper's printed high-MSE cutoff of 0.61:

| Metric | Paper | Measured | Difference (percentage points) |
|---|---:|---:|---:|
| Detection rate | 85.00% | 16.62% | -68.38 |
| False-alarm rate | 13.00% | 31.91% | +18.91 |
| Specificity | 87.00% | 68.09% | -18.91 |
| Precision | 85.00% | 34.87% | -50.13 |
| Balanced accuracy | 86.00% | 42.35% | -43.65 |
| F1 | 85.00% | 22.51% | -62.49 |
| AUC | 82.00% | 40.30% | -41.70 |

The result is not a near miss. The printed direction ranks attacks worse than
chance (`AUC=40.30%`). Reversing the direction improves AUC only to 59.70%.

## Every-cutoff check

Changing the cutoff cannot recover the reported operating point for this
score vector.

- In the paper direction, the maximum detection rate at `FA<=13%` is 7.00%,
  78.00 percentage points below the reported 85%.
- Even after reversing the score direction, the maximum detection rate at
  `FA<=13%` is 23.02%, still 61.98 percentage points below the target.
- The reversed direction can reach 79.34% detection only by accepting 61.91%
  false alarms, not the reported 13%.
- Across all 7,036,998 distinct score boundaries, the closest complete
  seven-metric row still has a maximum absolute error of 41.70 percentage
  points in the paper direction and 30.99 points after reversal.

This closes threshold choice and score-direction reversal as explanations for
the fixed fitted model. It does not transfer to an undocumented model that
would produce a different score ranking.

## Runtime result

The direct V100 measurement resolves the earlier hardware concern.

- The V100 full epoch took 143.883 minutes; the measured A16 projection was
  166.379 minutes. In this workload the V100 was about 1.16 times faster, not
  remotely an order of magnitude faster.
- The 183-minute budget held 1.268 observed epoch-equivalents on the V100.
- Ten epochs at the stable measured V100 throughput project to 23.98 training
  hours before scoring: 7.86 times the paper's entire 183-minute budget.

The paper does not report an epoch count. Therefore the exact statement is
that the predeclared ten-epoch completion cannot fit inside 183 minutes on the
measured V100 implementation; it is not that the paper explicitly claimed ten
epochs.

## What additional time can and cannot explain

This execution proves failure inside the reported time and a very large gap
for every cutoff of the resulting score vector. It does not mathematically
prove that unlimited further training can never change the ranking. The loss
was still decreasing, and no intermediate full-population score checkpoints
were taken.

Nevertheless, there is currently no positive evidence that ordinary extra
training would close the gap:

- The target requires a 61.98-point detection improvement even after giving
  the model the favorable reversed direction at the paper's false-alarm rate.
- The separate two-epoch, 32,768-row recurrent feasibility fit showed the same
  qualitative inversion and only 26.35% detection at `FA<=15%` after reversal;
  the full-data run gives 24.80% at the same cap.
- Approximately eight paper-time budgets would be required merely to execute
  ten epochs at the measured V100 rate, without evidence that reconstruction-
  loss improvement produces the required anomaly ranking.

The defensible statement is therefore that ordinary additional training has
no observed trajectory toward the reported operating point and is unlikely to
explain this discrepancy. “Impossible for all training durations” is not
earned by these checkpoints.

## Conclusion boundary

> The reported LSTM-SAE result did not arise from this declared
> implementation within the paper's reported 183-minute training budget on
> one V100. No cutoff or score-direction reversal rescues the fixed fitted
> scores. The target remains far outside the observed performance envelope,
> and the measured runtime would require about eight reported-time budgets for
> ten epochs. These results make ordinary additional training an implausible
> explanation, while not excluding an undocumented implementation or proving
> impossibility under unlimited time.

This finding does not identify author intent, fabrication, or the unpublished
implementation. Stop here for discussion before changing the website or
authorizing another run.

## Preserved records

- Machine-readable summary:
  `results/lstm_sae_paper_time_20260901.json`.
- Immutable derived attempt:
  `data/derived/atk-2022-deep-autoencoder/reproduction/remaining-paper-v1-results/paper_time/lstm_sae/v100_b92c9fa92866`.
- Slurm log SHA-256:
  `aae340c8b0fbad8d5bed37d271cf59374cf0d69352d18a613c7c716d3ba49423`.
- Score SHA-256:
  `4593437681d36c6565f94ec9e67643dca01c1ae6eea40ac90f3e330974ef942a`.
- Weight SHA-256:
  `bbf45d2751b81b0be0d40acac5de8f37f541982cea244335d3e83245fdee54ce`.
