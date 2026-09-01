# Recurrent score-only recovery contract

**Date:** 2026-09-01

**Status:** approved for LSTM-SAE and LSTM-VAE score-only execution; no
training, promotion, full anchor, AEA work, or publication is authorized by
this contract.

**Evidence type:** adaptive operational `X`. The result may determine whether a
feasibility gate failure is decision-relevant. It is not numerical (`N`),
mechanism (`M`), or attainability (`A`) evidence about the paper.

## Question

The two preserved recurrent pilots completed both training epochs but stopped
because scores from inference batches 256 and 128 were not identical within
the frozen absolute `1e-6` tolerance. Do those small floating-point differences
change any evaluation decision that matters for this feasibility screen?

## Frozen inputs

Do not train or alter weights. Use the exact preserved attempts from commit
`052ac373b77786ad58829b0ffe35568e971bb92d`:

| Model | Attempt | Weight SHA-256 | Selection SHA-256 |
|---|---|---|---|
| LSTM-SAE | `5f53ca7217aa` | `b26dc724...ac01b62` | `5e9a718e...620e93a` |
| LSTM-VAE | `1d6360ddcead` | `11953b8d...cb5344e` | `5e9a718e...620e93a` |

Verify the complete hashes from
`results/remaining_pilot_feasibility_20260901.json`, the original config and
failure records, unchanged `remaining_models.py` and `run_experiment.py`, the
prepared metadata identity, every consumed input checksum, and the saved row
identities before scoring.

## Frozen execution

- exact saved 12,119 score-row selection and labels;
- batch sizes 256, 128, 64, and 32;
- strict deterministic CUDA and the original scoring seed;
- one deterministic reconstruction for LSTM-SAE and the original
  deterministic-draw MC10 score family for LSTM-VAE;
- one GPU, 16 CPUs, 96 GiB RAM, `gpu-short`, and a 30-minute allocation per
  model; and
- separate immutable output attempts.

The scorer must contain no fitting, optimizer update, or weight write. A
timeout, checksum mismatch, nonfinite score, load mismatch, or unsupported
deterministic operation is preserved as an outcome.

## Comparisons to preserve

For every score family and every pair of batch sizes, record maximum, mean,
and 99th-percentile absolute difference, score-range-normalized maximum
difference, and exact-equality count.

For the primary score in both the paper's direction and the reversed control,
record separately for every batch size:

- strict printed-cutoff labels and all seven metrics;
- complete ROC/AUC and best balanced accuracy across all distinct boundaries;
- maximum detection at FA <=15% and <=15.5%; and
- the threshold and confusion counts at those points.

For every batch-size pair, count printed-cutoff label changes and compare the
above metrics. Also take each FA-capped threshold chosen from batch 256 and
apply that unchanged threshold to the other batches, recording any changed
labels or confusion counts.

No post-outcome tolerance defines automatic promotion. The result returns to
discussion. The original gate remains a preserved failure unless a later,
separately approved contract replaces it prospectively.

## Stop rule

After both recoveries and artifact audits, stop. Do not launch FC-VAE's full
anchor, rerun recurrent training, modify the score gate, or update the public
site before discussing the measured decision stability.
