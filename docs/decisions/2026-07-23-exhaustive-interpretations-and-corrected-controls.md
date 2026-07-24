# Exhaustive Paper Interpretations and Corrected Controls

**Date:** 2026-07-23  
**Status:** Approved by user direction  
**Scope:** Paper 1 first; reusable policy for later papers

## Decision

Every audited paper receives three non-conflated experiment tracks:

1. **Printed-paper track:** execute every stated operation in the place and
   order printed, including statistically improper operations. If a printed
   operation is internally impossible, retain it as `NON-EXECUTABLE` and run
   each minimal defensible repair as a named branch.
2. **Paper-interpretation track:** enumerate and execute every materially
   defensible interpretation of contradictory or missing text. No single
   “primary assumption” may stand in for the family.
3. **Corrected-control track:** execute the methodology we would recommend
   scientifically, including leakage-free preprocessing, untouched test data,
   valid model selection, and appropriate uncertainty estimates.

The paper itself is the only authority for whether a branch is
paper-consistent. Precursors, other papers, field practice, or tool defaults
may explain why an implementation is plausible, but they cannot silently add
methods to the paper-consistent family. Such externally motivated cases are
separately labeled author-implementation possibilities or corrected controls.

## Meaning of “all interpretations”

The project cannot test an infinite set of imaginable programs. It will make a
bounded exhaustion claim only after:

- every material normative statement in the paper has a source locator;
- every omission, contradiction, and invalid expression has a branch row;
- every materially defensible reading has an executable branch or an explicit
  non-executable record;
- every excluded reading has a written reason;
- coupled choices are represented by a dependency-aware branch lattice;
- omitted numerical hyperparameters have a predeclared finite envelope;
- all branches and exclusions are frozen before confirmatory outcomes.

This is exhaustive with respect to the documented text and frozen envelope,
not beyond all conceivable undisclosed implementations.

## ADASYN example

- Paper-printed anomaly path: apply ADASYN inside `B2 + M`, which is the test
  set.
- Paper-printed supervised path: concatenate `B + M`, apply ADASYN, and then
  split 2:1.
- Ambiguity branches: cross the all-customer-versus-heldout-customer
  construction of `M` with customer-versus-row split readings where the text
  conflicts.
- Corrected anomaly control: train on benign training customers and evaluate
  on untouched real/synthetic attack examples without test-set ADASYN.
- Corrected supervised control: split customers first, fit scaling on training
  only, apply ADASYN only to the training fold, and leave validation/test
  untouched.

The corrected result cannot count as reproduction of the paper. Conversely, a
statistically improper printed branch that reliably reproduces the table must
be reported as a reproduction under the printed method.

## Consequences

- The former implementation-v1 “primary branch” becomes one historical branch,
  not the experiment definition.
- Branch registration alone does not make a branch paper-exact.
- Screening may reduce compute, but it must preserve every branch result and
  use predeclared promotion rules that cannot discard a potentially matching
  result pattern.
- No best-branch reporting. The report includes all branches, seed
  distributions, failures, and exclusions.
- Paper-level conclusions distinguish:
  - reproduced by at least one paper-consistent branch;
  - not reproduced within the complete frozen paper-consistent family;
  - reproduced only by an externally added or corrected method;
  - non-executable from the available text.

## Machine closure v1

The first frozen enumeration is
`studies/atk-2022-deep-autoencoder/config/branch_lattice.toml`. It contains 921
compatible paper-consistent configurations (22 printed anchors and 899
interpretive cases) plus 22 separately identified corrected controls. Every
one of the 36
registered ambiguity rows has a machine-validated coverage reference. Coverage
means every registered option and every allowed option pair within each
model/data family; the unrelated 52.57-billion higher-order Cartesian product
is explicitly excluded. This closure does not authorize compute until the
corresponding source-derived implementations and structural gates exist.
