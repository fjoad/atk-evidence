# Paper-Literal Exploratory Contract

**Frozen:** 2026-07-21, before new Paper 1 model outcomes were observed.

> **Historical implementation-v1 contract.** The blanket fidelity conclusion
> attached to this contract was invalidated on 2026-07-23. This file remains
> immutable evidence of what the first implementation attempted; it is no
> longer the complete Paper 1 experiment definition. The replacement requires
> every printed and materially defensible interpretation plus separate
> corrected controls. See `PAPER_TO_CODE_TRACEABILITY.md` and
> `BRANCH_COVERAGE_CONTRACT.md`.

This contract reconstructs Takiddin et al. (2022) as written. It is exploratory
because prior static audits and an invalid SGCC proxy were already observed.

## Written workflow retained

1. Use the named SGCC and CER/ISET datasets.
2. Merge customer profiles into benign `B` and malicious `M`.
3. Normalize to zero mean and unit variance before splitting.
4. For anomaly detectors, split benign customers 2:1, train on `B1`, test on
   `B2 + M`, and run ADASYN inside the test set.
5. For supervised detectors, concatenate `B + M`, run ADASYN, then split 2:1.
6. Use the architectures and hyperparameters printed in Table I.
7. Use reconstruction MSE for SAE/AEA and the registered VAE score branches.
8. Use the paper's fixed thresholds because its “median of IQR of ROC” rule is
   not executable from the prose.
9. Compute DR, FA, SP, PR, balanced accuracy (called ACC), F1, and ROC-AUC.

## Primary author-intent assumptions

- SGCC: one customer is one full, chronologically ordered 1,034-day sequence;
  input/output length adapts to 1,034. This adds no undocumented windowing but
  conflicts with the paper's generic 48-input sentence.
- SGCC missing values: interpolate within each customer; unresolved edges use
  benign-training feature medians. The zero-fill branch is retained as a
  preprocessing sensitivity check.
- ISET: residential allocation code selects meters; retain only meter-days with
  exactly slots 1-48. Days with 46/50 DST slots or other gaps are excluded.
- Attacks are generated only from their partition's benign customer-days.
- Attack 1 draws one factor per daily profile. Attack 3 uses a valid integer
  start and 4-24 hour duration, mapped to two half-hours per hour.
- Z-scoring is feature-wise and fit jointly to complete `B + M` before the
  split. This deliberately retains the written leakage.
- Current `imbalanced-learn` ADASYN defaults are used with recorded seeds.
- Table I layer counts are mirrored total depth; the last listed encoder width
  is the bottleneck/latent width.
- Keras LSTM `activation` receives Table I's hidden activation; recurrent gates
  keep their default sigmoid. `dropout` is Keras input dropout; recurrent
  dropout is zero.
- VAE loss is mean reconstruction MSE plus mean analytic KL with unit weight.
  Reconstruction MSE and an explicitly labeled MSE-plus-KL surrogate are
  reported. Neither is the paper's undefined reconstruction probability, so
  applying its printed thresholds is assumption-bound rather than exact.
- Training uses Keras defaults not contradicted by the paper, batch size 512,
  maximum 30 epochs, and the frozen convergence rule in the plan. Batch 32 is a
  small-data sensitivity branch because it is Keras's likely implicit default.
- The supervised feed-forward benchmark uses two Softmax outputs, integer class
  labels, sparse categorical cross-entropy, and argmax classification. The
  supervised LSTM uses one Sigmoid output, binary cross-entropy, and threshold
  0.5. The paper does not state these output/loss/label details.
- Seeds are `11`, `22`, and `33`; data/ADASYN seed is `20260721`.
- SVM wording is interpreted as `kernel=sigmoid, gamma=scale`; omitted one-class
  `nu` uses the library default 0.5. ARIMA uses the smallest completion `(1,1,0)`.

## Claims this run can and cannot support

A completed table is evidence about this registered interpretation, not the
unique unknowable author code. Matching one row/seed is not reproduction of the
complete result pattern. Missing restricted data remain unavailable cells.
Intent, fabrication, and an infinite hyperparameter space are outside scope.
