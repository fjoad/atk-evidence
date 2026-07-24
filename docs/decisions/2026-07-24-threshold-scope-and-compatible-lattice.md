# Threshold Scope and Compatible Lattice

**Date:** 2026-07-24

## Decision

Treat threshold computation and threshold derivation scope as independent
experimental dimensions:

- formula: supplied printed constant or one of the three deterministic
  repairs of “median of IQR of ROC”;
- scope: derive on ISET and transfer, or derive separately on each dataset.

The branch generator must cover every option and every pair that can be
embedded in a complete executable configuration. It must not count impossible
pairs as interpretations.

## Source basis

Sections III-D and IV-B provide numerical thresholds and an undefined
ROC/IQR phrase. Section IV-B describes cross-validation and threshold
derivation on ISET; it does not describe a separate SGCC derivation. Dataset
specificity therefore says where a formula is applied, not which formula is
used.

## Incompatible pairs

1. Each ROC-derived formula is incompatible with
   `printed_threshold_no_derivation`, because ROC threshold selection requires
   both benign and malicious validation labels.
2. `printed_constant` is incompatible with `dataset_specific`, because the
   paper supplies only one ISET-derived constant per detector and no separate
   SGCC constant.

## Consequences

- The previous encoding of `dataset_specific` as a threshold rule is
  **INVALIDATED**.
- The generator now verifies compatible completion before requiring or
  constructing a pairwise case.
- The paper-consistent inventory changes from 942 to 921 configurations: 22
  printed anchors and 899 interpretive cases.
- The three-seed screen changes from 2,826 to 2,763 attempts.
- No executable paper interpretation was removed. The difference consists of
  combinations that cannot perform the operations they name.
- Existing experimental artifacts remain immutable and retain their original
  fingerprints.
