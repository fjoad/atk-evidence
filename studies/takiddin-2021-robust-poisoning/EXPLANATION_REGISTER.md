# Explanation register

**Created:** 2026-09-02

**Status:** active; first random-forest pilot completed on 20 September

**Boundary:** These are competing explanations for observations, not findings
about author intent. A source inconsistency does not identify how a value was
produced, and a future numerical result cannot by itself establish mechanism
or fabrication.

## E1 — one ordinary confusion matrix generated every metric in each row

**Status:** `CONTRADICTED FOR THREE TABLE V ROWS WITHIN THE DECLARED ROUNDING
MODEL`; otherwise not excluded by the any-prevalence screen.

At 0%, 10%, and 30% poisoning, the sequential ensemble's DR/FA/PR-implied
prevalence range is disjoint from its DR/FA/ACC-implied range. No one
continuous-rate population can yield all four metrics within outward
one-decimal intervals. This does not test integer sample counts, so a passing
row is only not excluded.

## E2 — every evaluated population was exactly balanced as described

**Status:** `WEAKENED FOR 58 OF 68 ROWS`.

Only 10 precision values agree with `DR/(DR+FA)` under exact balance, while all
68 accuracy values agree with `(DR+SP)/2`. Novelty-test ADASYN is explicitly
described as balancing the final test set. Two-class ADASYN occurs before a
later split, so exact balance in each split is not guaranteed. Exact split
counts would discriminate this explanation, but the paper does not report
them.

## E3 — ACC is balanced accuracy rather than the ordinary accuracy defined

**Status:** `SUPPORTED AS A REPORTING-PATTERN EXPLANATION; NOT IDENTIFIED`.

All 68 ACC values agree with balanced accuracy. The paper prints the ordinary
accuracy equation instead. The same numbers can coincide on a balanced
population, so this pattern alone cannot distinguish a balanced-accuracy
calculation from an ordinary calculation on equal class counts.

## E4 — PR used another definition, population, or threshold

**Status:** `SUPPORTED AS A PLAUSIBLE EXPLANATION; NOT IDENTIFIED`.

Precision is the dominant failure under balance. The prose incorrectly
describes precision as the fraction of malicious examples detected, while the
displayed equation is standard precision. A changed definition or evaluation
population could explain some values, but would contradict the presentation of
each row as one metric tuple. The three any-prevalence failures require more
than simply changing class balance.

## E5 — isolated reporting or transcription error

**Status:** `SUPPORTED FOR THE SINGLE F1 CELL AS A SIMPLE POSSIBILITY`; too
narrow by itself for the broader precision pattern.

Table III random-forest F1 at 20% poisoning fails the harmonic identity by
about 0.45 percentage points beyond the most favorable rounding boundary.
The PDF and transcription were visually checked, so any typo would be in the
source table rather than the CSV. A later independent transcription can further
test local copying risk.

## E6 — metrics were combined across runs, customers, thresholds, or folds

**Status:** `OPEN`.

Macro-averaging customer metrics, taking separate best runs, or calculating
columns at different thresholds can break single-confusion-matrix identities.
The paper does not report run counts, seeds, customer-level dispersion, or
aggregation details. Such a procedure could explain inconsistencies but would
not reproduce the row semantics as written.

## E7 — hidden unrounded digits reconcile the failed rows

**Status:** `CLOSED WITHIN ONE-DECIMAL ROUNDING`.

Every printed input received the complete outward half-unit-in-last-place
interval. The failed ranges remain disjoint. This closure is conditional on
ordinary rounding to one decimal place, not arbitrary reporting tolerance.

## E8 — the qualitative model and poisoning trends reproduce

**Status:** `UNTESTED`.

The table arithmetic broadly supports the prose ordering and degradation
descriptions, but no data were prepared and no model was run. Later testing
must preserve the exact-data, poison-construction, seed, and paper-time bounds.

**September 20 update:** one paired 100-tree forest pilot on 20 customers
learned strong discrimination. At p00, DR/FA/AUC are 92.31/3.02/98.55;
at p30, 61.18/0.44/94.36. This supports baseline viability in that
construction and weakens an expectation of universally poor baseline behavior
there. It does not establish the full-population table or its cross-model
ordering. The source sample and original/synthetic dependence remain material.

## E9 — the sequential architecture causes robustness through its staged
components

**Status:** `UNTESTED`.

The source tables do not separate staged feature use from architecture size,
data volume, preprocessing, hyperparameter selection, or thresholding. A
mechanism conclusion requires the `M5` and `M6` witnesses in
[`METHOD.md`](METHOD.md), not a headline metric match.

## E10 — the source inconsistency identifies fabrication or author intent

**Status:** `NOT IDENTIFIED`.

The present evidence cannot distinguish reporting error, undocumented
procedure, incompatible aggregation, provenance failure, or deliberate action.
No intent conclusion follows by eliminating only the explanations visible in
the paper.

## E11 — the reported computation fits the named hardware and time

**Status:** `UNTESTED; SOURCE BOUNDARY VERIFIED`.

The paper explicitly names one RTX 2070, 50 epochs, batch 100, and model-family
training times of roughly one to four hours. Those constraints are frozen for
a later prospective check. No throughput measurement has occurred in this
study.

**September 20 update:** the two small CPU forest fits took 0.649 and 0.566
seconds; the complete job took 16 seconds. This measures pilot cost only,
not the one-hour full-data claim, neural throughput, or an RTX-2070 workload.

## E12 — poisoning changes the decision cutoff's behavior while ranking survives

**Status:** supported within the paired RF pilot.

Default detection drops 31.13 points, but at the same 17.6% false-alarm cap,
the best detection on the saved scores drops only 5.25 points. AUC decreases
4.19 points. Threshold/calibration behavior can explain a substantial part
of the default-decision decline; some ranking deterioration remains.
Cutoffs were chosen using test labels as diagnostics, not independently
validated calibration. No claim about every model or the authors' choices.

## E13 — resampling and dependent splits contribute to the strong pilot result

**Status:** open; dependence and stratum differences observed.

At p00, original versus synthetic benign FA is 6.56% versus 2.33%;
at p30, 2.19% versus 0.11%. Training/test synthetic-parent crossings were
already recorded. Original-row AUC remains high, but removing synthetic test
rows does not remove training contamination or shared source days. A matched
preparation control, with the same model and an explicit comparison population,
would test this explanation. More seeds of the existing setup do not isolate it.
