# Does the order of poisoning and balancing change the forest result?

**Recorded:** 2026-09-22, before this control's data preparation or fit.
**Question:** E19; controlled (`C`) setup-mechanism diagnostic (`M`), not a
Table III reproduction, proposed-autoencoder mechanism test, or confirmation.

## Source and reason for the check

The complete paper was read on September 20. Pages 2677-2678 were re-read
and visually checked for this intervention. Section III-A.2 (p. 2677) says
to concatenate benign/malicious observations, balance with ADASYN, split
2:1, and scale. Section III-A.3 (p. 2678) describes malicious training data
mislabeled benign, and generalized penetration as a percentage of customers.
It does not explicitly place this corruption relative to ADASYN. Our original
balance-then-poison completion remains reasonable; it is not a discovered bug.

Moving poisoning before *full-pool* synthesis would change synthetic test
examples and could mix truly benign and malicious parents under the observed
benign label. Assigning clean ground truth to those test interpolations would
beg the question. Instead use the already-completed training-only control B
and keep its original evaluation rows fixed. This deliberately departs from
the paper's pre-split synthesis, in both arms equally.

This is a resampling-policy contrast, **not** an isolated class-prior effect.
Order changes class counts, generated feature geometry, training sample count,
bootstrap draws and fitted scaling. A population-prior correction formula is
not automatically valid for customer-dependent label corruption.

## Fixed arms and identities

- **B-p30:** reuse the saved `split-control-20260920-attempt1/B-p30` forest
  and predictions. On the fixed original training rows, clean ADASYN precedes
  poisoning. There are 4,532 training rows, 675 original attack labels flipped,
  and observed counts 2,952 benign / 1,580 attack (34.86320% attack).
- **D-p30:** same 2,632 original training rows, in the same order; first flip
  the same 675 attack labels from the same six customers, then stock ADASYN on
  **observed** labels, training-only. Before synthesis the observed counts
  are 1,052 benign / 1,580 attack. Request minority equality but retain actual
  rounded allocations. Do not trim, retry, or switch sampler if it fails.
- **Zero-poison guard, no extra fit:** regenerate this operation with no
  corruption and require every prepared array to equal saved B-p00 exactly.
  The existing B-p00 fitted result supplies clean context. A failed guard stops
  before the new fit; preserve failure and diagnose code, not model accuracy.
- **Evaluation:** all 1,288 original B test rows, same raw features, identities,
  order and clean labels: 1,105 attacks and 183 benign. No synthetic test rows.
  Final standardized features may differ because scaling is fit on each arm's
  actual training population, as in the original preparation. No source-day
  grouping is added; existing sibling/customer dependence remains.

Bind to B's immutable result records (which in turn bind inputs and models):

```
B-p00 d4e2c8904255435cabd909a8dda814ad5c7358c5779af440b1c64bfa5dbf5522
B-p30 735dd1e71e2ed0b3ea07d8c0344f9e089090af88b2090ea57807180341897aae
```

Use original preparation p00 metadata
`1c2e5ebfee9584c160fee209851e7f685d8c8d7971922d62a63960bc6308c3de`.
Recover raw rows; do not regenerate attacks, reselect customers/days, or refit
the old forests. Preserve all original artifacts and scientific revisions.

## Executable completions and provenance

Reuse the frozen poisoning rule, seed 20260920/role 301, and ADASYN seed
`library_seed(20260920, 202)`, five neighbors, imbalanced-learn 0.14.2.
Stock ADASYN sees only observed labels. Reuse its verified interpolation
recorder through an explicitly labeled view; never overwrite original truth.
Record both original parent UIDs and the interpolation coefficient of every
synthetic example, and count parent pairs containing zero, one, or two
originally malicious observations. Both parents must be observed-benign
training originals; no evaluation row enters training ancestry.

For continuity with B, interpolations of two true-benign parents keep the
synthetic benign label 0. Whenever either parent is truly malicious, set the
synthetic truth and attack-ID fields to -1 (unknown), not automatically to
benign or malicious. Observed label remains 0. These unknown-truth rows are
training-only. Report observed-label training accuracy, but set all-row
true-label training accuracy to null if any synthetic truth is unknown.
Original truth is never supplied to the model or used to choose synthesis.

Use the unchanged stock forest: sklearn 1.9.0, 100 trees, seed 20260920,
four fitting workers, other parameters identical to B. Main environment:
NumPy 2.5.1, SciPy 1.18.0, joblib 1.5.3, threadpoolctl 3.6.0. One thread per
numerical kernel. Fit StandardScaler on actual training raw float64 values;
store transformed float32 inputs. Save model, scores, every prepared array,
hashes, counts, versions, hardware/allocation and timings. Exact save/reload
probabilities and predictions must agree. Audit from saved arrays and scores;
do not silently regenerate research data during a local audit.

## Measurements, decisions and stopping

Report D-p30 minus B-p30 for all seven metrics, raw confusion counts, per-attack
detection, default predictions, AUC, and best saved-score DR at both fixed
FA caps 17.6% and 33.3%. Preserve daily-mean and constant controls. Reuse B-p00
as clean context, not a newly trained baseline. Best test-selected cutoffs are
diagnostic, not validated deployment thresholds.

If restoring roughly equal *observed* counts mainly moves the operating point,
default DR and FA should rise while AUC/common-cap DR change less. If common-cap
DR or AUC also changes substantially, resampling affects discrimination too.
Neither outcome identifies class proportions alone or the authors' code.
Small or adverse changes are retained, not a reason to try more samplers.
Describe all contrasts without a significance, equivalence, or population
claim: this dependent pilot supplies one matched intervention, not independent
repetitions. No post-hoc binary materiality threshold will be introduced.

Exactly **one** new real-data fit, one seed, on a Panther compute node. Budget:
one CPU-only allocation, 4 CPUs / 16 GiB / 15 minutes, zero GPU; software
fixtures and the zero-poison guard run first. The controlled output directory
must be new. Fail closed on incompatible versions, source/input/hash mismatch,
guard failure, invalid ancestry, or missing allocation. Preserve partial files
and error records; no automatic scientific retry or parameter alternative.
Stop after auditing this contrast. Decide whether any remaining order question
justifies another intervention before returning to specified model coverage.

As recorded, Panther access is unavailable and **no control has run**. The
user's continuation covers this bounded follow-up, not additional settings,
seeds, full-population preparation or publication.
