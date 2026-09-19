# Initial data preparation contract

**Recorded:** 2026-09-20, before real-data preparation or model outcomes.

**Scope:** implement and check the shared input pipeline. No model fitting.

## Verified sources

- Target paper: pp. 2676-2678, Table I and Sections III-A.1-3.
- Official dataset: [ISSDA CER electricity trial](https://doi.org/10.7929/ISSDA/BX59EU).
  Its public API was checked on September 20. The six existing consumption
  ZIPs match the official filenames, byte sizes, and MD5 values.
- The allocation CSV is a converted file, not the official archival TAB.
  A fresh comparison against the public workbook at commit
  db791cfeb5d725b28bfbd4b2b1bf33a8cadaa15c matched all five allocation
  columns for all 6,445 rows after documented blank/zero normalization.
  There are 4,225 residential codes. The comparison initially skipped the
  first workbook row under an incorrect header assumption; inspecting the
  workbook showed it has no header. Including that row resolved the mismatch.
- [Official file manifest](https://issda.ucd.ie/api/access/datafile/793):
  day 1 is 2009-01-01; the last two digits specify slots 1-48; values are kWh.
- The cited Jokar work is included in its author's
  [thesis](https://central.bac-lac.gc.ca/.item?app=Library&id=TC-BVAU-56294&oclc_number=1033147699&op=pdf),
  Chapter 5 (identified with DOI 10.1109/TSG.2015.2425222 in the preface),
  pp. 116-118, PDF pp. 134-136. Page 117 specifies uniform reduction factors
  in [0.1,0.8], a four-hour minimum bypass, and a sampled duration up to
  24 hours. That experiment uses hourly profiles; our target uses half-hourly
  readings. The target swaps the mean and scaled-mean attack numbers relative
  to the thesis. Follow the target's Table I ordering.
- Target reference [26], [Nabil et al. 2018, Section V](https://arxiv.org/html/1809.01774),
  independently specifies half-hourly factors in [0.1,0.8] and durations
  [8,48] slots, but its start/end wording also needs an endpoint convention.

## Initial choices and unresolved alternatives

| Item | Initial executable choice | Status / other plausible reading |
|---|---|---|
| Population | Seeded selection of 3,000 sorted residential IDs, without replacement, before looking at readings or results | Paper omits IDs; exact-sample identity remains unknown. Record selected IDs only in ignored preparation outputs. |
| Missing/DST days | Require exactly one reading at every slot 1-48; reject whole days with missing, duplicate, or extra slots; reject invalid numeric values | Not specified by paper. No interpolation or merging duplicated readings. |
| Dates | All available dates for full mode; pilot takes the first 28 complete days per selected customer after the full raw scan | Pilot is partial input coverage, not a paper result. |
| Randomness | Root seed 20260920; separate fixed streams for selection, six attacks, splitting, balancing, and poisoning | Our choice; never change it to improve a result. |
| Attack 1 | One uniform factor per customer, shared across that customer's days | Scope of “all samples” is ambiguous; a global factor remains an explicit alternative. |
| Attacks 2 and 5 | Independent uniform factor per half-hour reading | Hour-paired factors remain an untested alternative. |
| Attack 3 | Uniform integer duration 8-48 slots, then uniform valid start; zero slots [start,start+duration) | Guarantees 4-24 hours within a day. Start-first with clipping is an implemented alternative. |
| Attacks 4 and 6 | Repeat each day's mean; reverse all 48 coordinates | Target Table I formulas. |
| Split | Seeded row permutation, floor(2n/3) training, remaining test; no stratification or customer grouping | Paper says disjoint samples. Corrected customer-disjoint evaluation is separate. |
| ADASYN | imbalanced-learn 0.14.2, five neighbors, requested class equality, raw kWh before scaling | Preserve actual counts: rounding need not yield exact equality. Never substitute SMOTE on failure. |
| Scaling | StandardScaler fitted to final training inputs, then applied to test; float64 statistics, float32 stored inputs | Train-fit scale is explicit; per-day scaling is not silently substituted. |
| Poisoning order | After clean resampling/splitting, before fitting the scaler; only training labels/inputs change | Poison-before-ADASYN remains unresolved. |

## Two-class path

Generate B and all six malicious siblings M. Apply ADASYN to B+M, then split
2:1. Keep true labels and a separate observed training-label vector.

For generalized models, select a nested seeded fraction of selected customers
(floor(p*C)) and flip all their original malicious training labels to benign.
Synthetic rows are originally benign and retain both parent IDs; they are not
arbitrarily assigned to one household. Customer-based p differs from the
fraction of all training rows flipped; report both.

For a customer-specific model, select floor(p*n_train) malicious training rows
without replacement and flip them to benign. This uses all training rows as
the denominator. Report the resulting malicious-class fraction too. The
alternative denominator, malicious training rows only, is exposed explicitly.
Fail if the requested flip count exceeds available malicious rows.

Feature values and true test labels remain unchanged by this label flip.

## Novelty path

Split B into B1/B2 in ratio 2:1. Test on B2 plus all M and apply ADASYN to
that test population. The source does not explain how a benign-only training
set acquires malicious rows. Our initial completion replaces selected B1
profiles with one of their six attack siblings, keeping observed label 0
and retaining the true malicious label and attack ID.

Generalized selection uses the same nested customer selection as above.
Customer-specific selection replaces floor(p*len(B1)) training profiles.
It does not append rows or change training-set size.

Preserving the stated all-M test population means a replaced attack profile
also occurs in the test pool. This direct overlap is reported explicitly;
it is a consequence of this completion, not an assertion about author code.
Do not silently remove those rows or change the test population between levels.
A disjoint-attack control is a later separate preparation.

Fit scaling on the actual contaminated training inputs. The raw test
population stays fixed across levels; its standardized values may change.

## Provenance and software checks

Each row retains original row identity, customer/day where defined, attack ID,
true label, observed label, and synthetic status. Every ADASYN row records its
two original benign parents and interpolation coefficient. Unmodified ADASYN
uses a neighbor estimator that records its two index queries. Replaying the
same seeded interpolation draws recovers the parents without another neighbor
search. Every generated interpolation is checked against the stock output.
Values and labels must match uninstrumented ADASYN exactly in fixtures.

Tests cover attack formulas, parser exclusions across chunk boundaries,
deterministic/nested poisoning, split identity, scaler fitting population,
unchanged test labels, synthetic ancestry, exact library parity, output
round-trip, and refusal outside a Slurm compute job.

## Bounded first execution

After code and this contract are committed, one CPU Slurm job may verify the
raw bytes, scan archives for 20 seeded customers and take their first 28
complete days (avoiding an assumption that everyone started on the same date),
and exercise both paths at 0% and 30%. Include one customer-specific example.
Maximum allocation: 15 minutes, 4 CPUs, 16 GiB RAM, zero GPUs. Use one thread
per BLAS/neighbor kernel. Preserve success or failure and actual counts.
No full-population ADASYN, model training, scoring, or statistical conclusion.
If the allocation is insufficient, preserve the attempt before changing it.
