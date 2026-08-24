# Phase 2 discovery sandbox

**Authorized:** 2026-08-24

**Status:** First wave pre-recorded; execution pending

**Classification:** exploratory `X`; candidate questions touch `N`, `M`, and
`A`, but no sandbox outcome is eligible evidence for any of them

**Source anchor:**
[`CLEAN_READER_ORIENTATION.md`](CLEAN_READER_ORIENTATION.md)

## Boundary

This is a disposable source-derived sandbox, not a reimplementation of the
paper's numerical experiment. It uses no named data, historical project model,
production runner, branch machinery, prior score vector, or prior result. It
cannot select a primary source completion or become confirmation
retrospectively.

Experimental computation must run on a cluster compute node. Local work is
limited to writing the script, syntax/static checks, documentation, transfer,
and result inspection.

## First-wave question

Can the smallest transparent observations distinguish four explanations that
remain conflated by the paper?

1. the synthetic attacks are largely separable through simple statistics;
2. temporal order is the one structure that can require a sequential
   capability;
3. the printed output activations impose a reconstruction-domain floor; and
4. Gaussian reconstruction probability must be low, not high, for anomaly.

## Competing predictions

### X1 — triviality floor

**Minimal setup:** Generate positive 48-step daily profiles from a fixed
asymmetric shape with random amplitude, offset, phase, and noise. Apply toy
versions of the six printed attack descriptions. Fit benign-only median/MAD
calibrations for energy, mean, variance, range, zero count, roughness, and
linear trend. Preserve every feature's AUC and oracle balanced accuracy; do not
select a favorable feature as a paper result.

- If attacks 1--5 contain obvious shortcuts, at least one simple feature will
  strongly separate most of them.
- If the task genuinely requires elaborate representation learning, these
  rules will remain near chance on most attacks.
- Reversal preserves each profile's multiset, mean, variance, range, and total;
  order-insensitive features should therefore tie exactly or nearly exactly.

### X2 — temporal witness

**Minimal setup:** Create benign cyclic shifts of an asymmetric sawtooth-like
sequence. Compare amplitude reduction, block disruption, and reversal. Train
one small undercomplete dense autoencoder and one similarly sized seq2seq LSTM
autoencoder on the same benign training rows with linear reconstruction heads,
one seed, fixed epochs, and no tuning. Linear heads deliberately remove the
known Softmax/sigmoid domain confound so this toy question focuses on temporal
capability.

- If recurrence supplies a task-relevant capability, the LSTM should react
  more strongly than the dense model when local order is necessary, while
  both may react to amplitude changes.
- If both models rank the order-sensitive anomalies similarly, this toy
  benchmark does not demonstrate a recurrence-specific advantage.
- If neither reacts, either the training/setup suppresses the capability or
  the witness is not discriminating; that outcome is a sandbox failure, not
  mechanism-absence evidence.

### X3 — output-domain consequence

**Minimal setup:** Standardize the complete toy profile collection. For benign
and each attack, compute the fraction of coordinates outside each decoder's
range and the exact squared-distance lower bound to the probability simplex
and unit box.

- Softmax and sigmoid lower bounds should be positive whenever standardized
  rows lie outside their output sets.
- The bound may differ between benign and attacked rows; if it does, a
  reconstruction score can be driven by domain geometry rather than learned
  structure.
- This cannot bound DR, FA, AUC, or the published result.

### X4 — VAE score direction

**Minimal setup:** Evaluate fixed-unit Gaussian reconstruction density for an
explicit increasing sequence of reconstruction errors.

- Probability must decrease monotonically with error.
- Therefore low probability is anomaly-consistent; a shared
  greater-than-threshold anomaly rule applied directly to probability points
  in the opposite direction.

## Fixed execution contract

- Script:
  `exploration/phase2_discovery_sandbox.py`.
- Result:
  `exploration/results/phase2_seed_20260824.json`.
- Seed: `20260824`.
- Synthetic sizes: 512 benign training profiles, 256 benign test profiles, 256
  profiles per attack/witness class.
- Sequence width: 48.
- Neural comparison: one dense AE and one seq2seq LSTM AE; fixed architecture,
  25 epochs, batch 64, Adam `1e-3`, MSE, linear outputs.
- Budget: one cluster job, one accelerator at most, 20 minutes wall time, no
  hyperparameter search, no rerun for a more favorable outcome.
- Promotion rule: a question may enter Phase 3 only if the observation differs
  across competing predictions and remains directly relevant to a
  source-located paper claim.
- Stopping rule: stop after this one complete result, inspect every section,
  and decide explicitly whether any failed diagnostic needs redesign. Do not
  proceed automatically to VAE training, attention training, named data, or
  production code.

## Results

Pending. The eventual record must include failures and warnings as well as
successful measurements.
