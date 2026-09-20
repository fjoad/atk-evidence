# Matched split and resampling check

**Recorded:** 2026-09-20, before preparing or fitting these controls.

## Question

How much of the first forest's strong pilot behavior survives (1) generating
synthetic training examples without access to test observations and (2)
keeping every attack sibling of a held-out source day out of training?

The existing result is reference A. These are controlled alternatives to the
printed preparation, not replacements for the preserved reproduction attempt.

## Inputs and invariants

- Reuse the original 560 days from 20 customers and their six existing attack
  versions. Recover them from the preserved p00 train/test arrays; do not
  regenerate attacks or choose new customers/dates.
- Verify the frozen preparation metadata and all arrays before use. p00
  metadata SHA-256 is
  1c2e5ebfee9584c160fee209851e7f685d8c8d7971922d62a63960bc6308c3de.
- Reuse the original forest predictions. Baseline result SHA-256 values:
  p00 a47c4f85e0e66891b6d2ad2103bf3f5498ba28b34a99eec94c686e288fec125a;
  p30 71708e92acc5c1cdc00681dd93107a436f7c55718b23b363b52de6af0d438803.
  Verify their score/model hashes and source revisions.
- Same 100-tree implementation and settings, seed 20260920, four fit workers,
  one scoring worker, ordinary accuracy, and native probability argmax.
- Same poisoning customers at each level: six of twenty selected using the
  existing role-301 seeded order at nominal 30%. Only original malicious
  training labels are flipped. Report actual fractions in each arm.
- Same ADASYN implementation, five neighbors, clean labels before poisoning,
  and role-202 random seed. Fit scaling using each arm's training inputs.

## The three arms

| Arm | Training preparation | Evaluation |
|---|---|---|
| A, saved reference | ADASYN before the row split, as previously implemented | Reuse saved probabilities on specified original rows only |
| B, training-only synthesis | Preserve A's original training rows; fit ADASYN on those rows alone | A's original test rows, without synthetic test examples |
| C, source-day split | Keep a seeded two-thirds of the 560 source days for training, with all seven versions together; then training-only ADASYN | The shared evaluation described below |

The B intervention preserves original train/test identities, but changes
synthetic values, ancestry, and realized training class counts. ADASYN rounds
its allocations, so we retain and report its output rather than trim rows or
pretend all arms have exactly the same training size. The contrast estimates
the effect of this preparation policy, not leakage independently of every
change in augmentation geometry.

For C, sort the 560 source-day IDs, permute with root seed 20260920 and
independent RNG role 501, and assign floor(2*560/3)=373 days to training and
the remaining 187 to holding out. This keeps training coverage near B's row
budget without purging almost all of B's training rows. All seven original
versions of each training day stay together. This tests new days from the
same customer cohort; it does not test unseen-customer generalization.

## Paired evaluation

The primary common evaluation E is the intersection of A's original test
rows and C's 187 held-out source days, in A's original test order. Select it
using identities alone, before inspecting predictions. Require both classes.
Record E's row, class, customer, and source-day counts plus its identity hash.

Compare A, B, and C on exactly E, with identical true labels. A reuses old
predictions; B's scores on its larger original test set are sliced by the same
mask; C is scored directly on E. No synthetic evaluation rows are included.
Also compare A/B on all 1,288 original test rows as a secondary, larger
evaluation for the synthesis-placement intervention. Do not evaluate C on
original test rows belonging to its training days.

Verify:

- B's original training and test identities exactly match A's corresponding
  original identities.
- Every synthetic B/C training parent is an original row in that arm's
  training set. No parent belongs to that arm's evaluation.
- C shares no source-day identity, including synthetic ancestry, with E.
- Raw E values, order, and true labels are identical across arms and poison
  levels. Standardized E values may differ with the training-fitted scaler.

## Outcomes and interpretation

For each poisoning level, report all seven metrics, original counts,
per-attack detection, AUC, and the same saved-score ROC diagnostics used
before (FA caps 17.6% and 33.3%). Primary comparisons are B-A and C-B on E;
secondary B-A uses all original test rows.

If B falls relative to A, pre-split resampling is a candidate contributor. If
C additionally falls relative to B, source-day overlap is a candidate
contributor. Training membership/counts also change in C; state this limitation.
If performance stays high or improves, report that result and retain the
opening for the method. A difference smaller than five AUC points will not
by itself trigger repeated fits in this pilot; this is an operational triage
marker, not an equivalence margin or significance test.

No confidence interval or population claim is planned for one seed and twenty
dependent customers. Test-chosen cutoffs remain diagnostics, not deployment
calibration. This control cannot identify the authors' code or intent.

## Budget and stopping

One Slurm CPU job: at most 15 minutes, four CPUs, 16 GiB, no GPU. Four new
forest fits total: B/C at 0%/30%. A is never retrained. Constructed fixtures
are separate software tests. All real preparation, fitting, and scoring run
on the compute node. Preserve failures; do not change grouping, seed,
sampling, or parameters in response to performance. Stop after artifact
checks and update the journal with the next question justified by the result.
