# Recurrent score differences were decision-stable at the printed cutoffs

**Date:** 2026-09-01

**Evidence type:** operational `X`; no numerical (`N`), mechanism (`M`), or
attainability (`A`) conclusion about the paper.

## Plain-language question

The LSTM-SAE and LSTM-VAE feasibility attempts trained successfully for two
epochs, but the frozen pilot stopped because scores changed slightly when the
same saved weights were evaluated in different batch sizes. Before changing
that gate or spending days on full training, we asked a cheaper question:

**Do those numerical differences change the classifications or performance
limits that matter for this feasibility decision?**

No model was trained again. Jobs `385583` and `385584` reloaded the exact saved
weights and exact 12,119-row selection, then scored batches 256, 128, 64, and
32 under strict deterministic CUDA. Both jobs completed successfully.

| Job | Saved model | Slurm outcome | Allocation time | Scorer time |
|---:|---|---|---:|---:|
| 385583 | LSTM-SAE | completed `0:0` | 1:14 | 53.44 s |
| 385584 | LSTM-VAE | completed `0:0` | 2:33 | 133.40 s |

## What changed numerically

The scores are not bitwise batch-invariant. Across all six batch-size pairs:

| Saved model | Largest primary-score difference | Largest primary p99 difference | Largest difference / primary score range |
|---|---:|---:|---:|
| LSTM-SAE | `2.0113942e-5` | `3.5801721e-8` | `6.4136022e-7` |
| LSTM-VAE | `3.0121445e-5` | `1.4246101e-6` | `3.0247617e-5` |

LSTM-VAE's auxiliary deterministic MSE-plus-KL score differed by as much as
`6.1134268e-4` between batches 64 and 32. The original all-score absolute
`1e-6` gate therefore remains a real, preserved failure; this diagnostic does
not rewrite it after seeing the outcome.

## What did not change

At each model's printed paper cutoff, every tested batch produced exactly the
same labels. Consequently DR, FA, specificity, precision, accuracy, F1, and
AUC were identical at that cutoff.

The complete ROC comparison was also stable at the decision level:

- LSTM-SAE's AUC, best balanced accuracy, and best achievable DR/FA at both
  FA<=15% and FA<=15.5% were identical across all four batches.
- LSTM-VAE's largest AUC difference was `0.0000217922` percentage points.
  Its best balanced accuracy and its best achievable DR/FA at both false-alarm
  caps were identical across batches.

The test also froze each batch-256 FA-capped cutoff and transferred that exact
number to every other batch. This exposes one real boundary effect rather than
hiding it: no transfer changed more than one of 12,119 labels. For example,
LSTM-SAE changed one benign decision, moving FA from `14.99582%` to
`14.97908%` with unchanged DR. LSTM-VAE changed one malicious decision in the
printed FA<=15% view, moving DR from `9.24479%` to `9.22852%` with unchanged
FA. Other one-row effects are preserved in the machine record. They are tiny,
but they mean the result is near decision-invariance, not exact invariance for
every possible transferred threshold.

## The pilot scores in context

These two-epoch pilot weights are far from the paper's 81% detection / 15%
false-alarm target on this selected evaluation:

| Saved pilot / direction | Printed-cutoff DR | Printed-cutoff FA | Best DR at FA<=15% |
|---|---:|---:|---:|
| LSTM-SAE, paper direction | 21.76107% | 44.00000% | 9.01693% |
| LSTM-SAE, reversed control | — | — | 26.35091% |
| LSTM-VAE, paper direction | 10.07487% | 17.32218% | 9.24479% |
| LSTM-VAE, reversed control | — | — | 26.52995% |

This table explains why batch arithmetic cannot rescue these particular
scores. It is **not** a Table-III reproduction result: the weights came from a
deliberately truncated two-epoch feasibility fit, not the paper's full
training schedule. It cannot establish that longer training, another source
interpretation, or another fitted model will remain at these values.

## Decision boundary

The observed recurrent batch differences are operationally negligible for
the printed-cutoff and ROC-envelope conclusions on these saved weights and
rows. They do not explain the poor pilot scores. That supports discussing a
prospective decision-level feasibility rule, but it does not automatically
replace the frozen `1e-6` all-score gate or promote either model.

All source identities, implementation hashes, input hashes, weight and
selection hashes, deterministic-runtime checks, artifact hashes, shapes, and
finite-value checks passed. The two allocations used 227 GPU-seconds in total,
or about 0.0631 GPU-hours. Post-result verification passes all 252 repository
tests (140 study and 112 root) and strict data verification. Exact outputs and
hashes are recorded in
[`results/recurrent_score_recovery_20260901.json`](results/recurrent_score_recovery_20260901.json).

Stop for discussion. Do not start recurrent retraining, the FC-VAE full anchor,
AEA changes, mechanism work, or publication based on this result alone.
