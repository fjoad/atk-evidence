# Printed-table arithmetic audit contract

**Frozen before execution:** 2026-09-02

**Evidence:** controlled source-only numerical consistency check (`C/N`)

**Inputs:** the visually transcribed Tables III–V under `reported/`, from PDF
SHA-256 `03a372fb5ee129799d43a50e6bec8f86fb108e9f8bdc85350e3ffe3e13d4a4ff`.

## Question

Do the metrics printed for each model and poisoning level satisfy the paper's
own formulas and its stated balanced-data evaluation, allowing every value the
full benefit of rounding to one decimal place?

This audit does not ask how the numbers were produced and cannot identify
author intent.

## Fixed transcription

Each CSV contains `model,metric,p0,p10,p20,p30`. Required coverage is:

- Table III: 7 models × 7 metrics × 4 levels = 196 cells;
- Table IV: 7 models × 7 metrics × 4 levels = 196 cells; and
- Table V: 3 models × 7 metrics × 4 levels = 84 cells.

Total: 476 values. Model names, metric names, poisoning levels, and all values
must be checked against the rendered pages before the script executes.

## Rounding model

A printed value `v` represents the closed interval
`[v - 0.05, v + 0.05]` percentage points, clipped to `[0,100]`. Interval
intersection counts as a pass. This outward convention is favorable to the
paper and avoids choosing hidden unrounded digits.

## Predeclared checks

For every model × poisoning-level row:

1. **Specificity:** the interval for `SP` must intersect `100 - FA`.
2. **F1:** the interval for `F1` must intersect the range of
   `2*DR*PR/(DR+PR)` over the DR and PR intervals.
3. **Balanced accuracy:** because the paper says ADASYN balances the relevant
   benign/malicious population, the ACC interval must intersect
   `(DR+SP)/2`.
4. **Balanced precision:** under that same balance, the PR interval must
   intersect `DR/(DR+FA)`.
5. **Any-prevalence screen:** without assuming balance, calculate the prevalence
   interval compatible with DR/FA/PR and the interval compatible with DR/FA/ACC.
   Disjoint intervals prove that no common class prevalence can reconcile those
   four reported metrics within rounding. Overlap is only “not excluded,” not a
   proof that one integer confusion matrix exists.

AUC is not determined by one operating point and receives no identity check.

## Prose checks

Calculate without selecting favorable rows:

- per-model and across-model DR change from 0% to 10%, 20%, and 30%;
- generalized minus customer-specific DR at each matched cell;
- deep `(feed-forward, GRU, AEA)` versus shallow `(random forest, AdaBoost,
  ARIMA, SVM)` mean DR at 0% and 30%;
- AEA > GRU > feed-forward ordering at every poison level;
- sequential-ensemble DR deterioration from 0% at all levels; and
- sequential versus AEA and ensemble-averaging differences at 30%.

The output will state whether each prose phrase is ambiguous between absolute
percentage-point change and relative-percent change. No claim is called false
merely because “roughly,” “up to,” or an unstated averaging convention admits
more than one reading.

## Outcomes and language

- A passed identity is reported plainly.
- A failed balanced identity is bounded to the paper's stated balance and the
  printed metrics.
- A failed any-prevalence screen is a stronger continuous-rate inconsistency but
  still does not determine provenance.
- Regular numerical patterns may motivate later tests but are not evidence of
  fabrication by themselves.
- No result from the earlier paper is used in this calculation.

## Stop rule

Run the deterministic checker once after this contract and the table CSVs are
committed. Correct only demonstrated transcription or checker defects, preserve
the failed attempt, and rerun under a new recorded revision. Stop after the
source finding; do not prepare data or train a model.
