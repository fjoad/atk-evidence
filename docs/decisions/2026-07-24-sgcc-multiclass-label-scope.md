# SGCC Multiclass-Label Scope

**Date:** 2026-07-24  
**Status:** Accepted before replacement results

## Decision

The `multiclass_labels` ambiguity applies only to the ISET multiclass-SVM
family. ISET has six generated attack identities, so both binary
benign/malicious labels and seven classes (benign plus attacks 1–6) are
source-grounded readings of “multiclass SVM.” SGCC supplies only a
benign/malicious customer label and does not identify six ISET attack
functions. Its multiclass-SVM family therefore retains only the executable
binary-label reading.

The runner still fails loudly if a seven-class request reaches data without
attack IDs. This prevents a future manifest edit from silently inventing SGCC
classes.

## Consequences

- No ISET interpretation is removed.
- The SGCC multiclass-SVM printed-anchor ID changes from
  `sgcc_multiclass_svm-2bf906dd23eb` to
  `sgcc_multiclass_svm-42e55d392009`.
- Total pairwise coverage remains 921 paper-consistent configurations and 22
  corrected controls; only the SGCC family's raw Cartesian space and dimension
  count shrink.
- The committed branch inventories are regenerated. Their hashes also include
  newly embedded dataset/model/table/dimension metadata, so an inventory-hash
  change is not itself an experimental-semantic change.

## Rationale

Testing all defensible interpretations does not authorize creating class labels
that the named dataset and paper never provide. Retaining the impossible SGCC
case would test an externally invented task, not an ambiguity in the text.

