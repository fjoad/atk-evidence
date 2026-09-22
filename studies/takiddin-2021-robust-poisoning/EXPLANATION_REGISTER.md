# Explanation register

**Created:** 2026-09-02

**Status:** active; four model pairs, forest controls, and read-only SVM follow-up complete

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

**Status:** resampling-policy contribution supported in this pilot; an
additional collapse under source-day grouping was not observed.

At p00, original versus synthetic benign FA is 6.56% versus 2.33%;
at p30, 2.19% versus 0.11%. Training/test synthetic-parent crossings were
already recorded. Original-row AUC remains high, but removing synthetic test
rows does not remove training contamination or shared source days. A matched
preparation control, with the same model and an explicit comparison population,
would test this explanation. More seeds of the existing setup do not isolate it.

**September 20 controlled result:** on the same 445 original test rows,
moving synthesis into training lowers AUC from 98.31606 to 91.84157 at p00
and from 90.98094 to 81.98823 at p30. The larger 1,288-row A/B comparison
gives the same direction (-5.78765/-8.00682 points). Synthetic values and
training counts also change; this identifies a policy effect, not leakage
alone. Grouping whole source days raises AUC by 1.33925/0.98358 relative to
training-only synthesis on the original row split. C retains AUC
93.18082/82.97181, versus the simple daily-mean reference's 67.62975.
This weakens the explanation that these two dependence paths account for all
useful discrimination. It does not establish equivalence, unseen-customer
generalization, or full-data reproduction. See the
[matched-control record](results/split_control_20260920/README.md).

The threshold explanation also remains material but incomplete in C: default
DR drops 31.86528 points with poisoning, versus 21.24352 at the common
17.6% FA cap; AUC falls 10.20901 points. No automatic repetition is justified
merely to reverse the observed result. Continue to a separately specified model
question rather than turn this pilot into an undeclared search.

## E14 — the forest's useful ranking and cutoff effect are unique to that model

**Status:** weakened for the original pilot by a second baseline, not generally excluded.

The stock historical-algorithm AdaBoost pair (d47a6de, job 398348) gives
DR/FA/AUC 81.09/15.17/90.71 at p00 and 46.43/5.06/83.74 at p30.
At the common 14.1% FA cap, saved-score DR is 80.72/67.33; at 29.9%,
88.78/80.45. Ranking deteriorates but remains useful; the 34.66-point
default-DR decline is not a complete loss of discrimination. The poisoned
scores can exceed the printed 70.1% DR within its 29.9% FA allowance, while
the unpoisoned scores cannot reach 85.7% DR within 14.1% FA. Neither partial
comparison is full-population reproduction or an all-model limit.

The forest exceeds AdaBoost AUC on these identical pilot rows, opposite their
printed ordering. Different runtime versions, unspecified model choices,
twenty dependent customers, pre-split synthesis, and one seed limit the
inference. No extra seed or alternate AdaBoost setting was run to change the
outcome. See the [AdaBoost record](results/adaboost_pilot_20260920/README.md).

## E15 — a cutoff or reversed score rescues the first sigmoid SVM

**Status:** excluded for these two fitted models and sampled test rows at
the corresponding printed corners; not excluded for other SVM configurations.

Frozen e698173, job 398709: AUC 65.64194/63.38126 at p00/p30; default
DR 62.08145/39.45701 and FA 39.39663/31.14463. All 2,224 boundaries in
both directions were inspected. At p00 FA<=10.2%, best DR is 19.36652,
versus 89.2 printed; at p30 FA<=25.7%, best DR is 33.48416 versus 73.7.
Reversal reaches only 1.90045/5.70136 at those respective caps.

Both solvers report success, saved-score orientation and reload agree, and
positive/corrupted-label fixtures pass. But observed-label training accuracy
is only 63.32885/59.72222. Numeric gamma is nearly 1/48 on these standardized
inputs; that observation does not prove all parameter completions equivalent.
The simple daily-mean score has AUC 66.19624 on the same test rows, while
forest/AdaBoost rank substantially better. A universal shared-task failure
therefore does not follow from the weak SVM.

The next proposed explanation check is read-only replay of support-vector
margins plus inspection of the declared sigmoid kernel on a fixed small
subset. Kernel geometry is not yet measured and no cause is established.
No extra fit, parameter, calibration, or seed has been tried. See the
[SVM record](results/svm_pilot_20260921/README.md).

## E16 — the weak SVM result comes from an incorrect score reconstruction

**Status:** excluded to the frozen numerical tolerance for the saved artifacts.

Read-only job 398978, frozen 4665e07, independently reconstructed13,392
train/test row-model scores. Maximum error is 1.67688e-12; all labels agree
with no near-zero ambiguous margins. Native replay exactly matches saved
test scores/labels. Support-row bindings, coefficient signs/bounds/equality,
training accuracy, and original artifact hashes pass. This closes the stated
formula/sign/persistence explanation here, not every possible implementation
or source-reading error. See the [diagnostic record](results/svm_replay_20260921/README.md).

## E17 — the declared sigmoid kernel satisfies the usual convex-dual conditions

**Status:** contradicted numerically on the fixed training kernel; performance
causation and fitted-solution suboptimality remain unestablished.

The identity-selected512-row subset has 366 resolved negative eigenvalues,
minimum -23.45098 with tolerance 1.48403e-8. After centering, minimum -19.96835
persists, with a zero-sum witness satisfying both label-mapped dual equalities
(residual about 2.05e-15). This negative direction embeds in the full fixed
training matrix, so the usual concave-dual argument is unavailable here.
No KKT test, feasible improvement at the fitted box boundary, global-optimum
comparison, or alternate parameters were evaluated. Do not turn the property
into an identified cause of the observed accuracy or an all-SVM exclusion.

The diagnostic is complete; retain parameter sensitivity as open. Proposed
next coverage step is a separately specified feed-forward pilot, not an
unbounded SVM search. No new model has been fitted in this follow-up.

## E18 — the low default feed-forward detection excludes the reported corner

**Status:** contradicted for the saved repaired-BCE pilot scores; no complete
row or full-population reproduction is established.

Frozen b5da23a/job 400825: 50 epochs each, identical initial weights and the
original inputs. DR/FA/AUC 89.14/7.45/96.35 at p00 and 51.58/0.53/90.89
at p30. At FA<=9.3%, unpoisoned DR 90.76923 rounds to 90.8, matching the
printed detection with lower FA. At FA<=24.4%, poisoned DR 88.86878 exceeds
76.0. The whole metric pattern differs: p30 accuracy rounds to 75.8 as printed,
but detection, FA, precision, F1 and AUC do not. Test-selected cutoffs are not
validated calibration. AUC drops 5.46415 points; useful ranking survives.
See the [feed-forward record](results/feed_forward_pilot_20260922/README.md).

## E19 — poisoning/balancing order contributes to the repeated cutoff pattern

**Status:** open, not causally measured.

Forest, AdaBoost and repaired feed-forward show lower default detection and
lower FA under the declared one-sided label corruption, with substantial
remaining ranking. In this preparation, balancing precedes the 675 label flips,
so observed training class proportions change. The paper leaves this relative
order incomplete. This is a named shared-setup question worth specifying before
further costly model coverage, not proof of which order the authors used.
Any controlled run needs an explicit intervention, unchanged evaluation,
budget and stopping rule first. No such comparison was silently launched.

September 22 source/design update: III-A.2-3 does not explicitly settle the
relative order, so the original completion is not an identified coding error.
`POISON_BALANCE_CHECK.md` now specifies D-p30 versus saved B-p30, using the
same original training/evaluation rows and training-only synthesis in both
arms. This changes synthesis geometry/counts as well as observed proportions;
it will not identify a class-prior-only effect. Synthetic rows with any truly
malicious parent have unknown truth and remain training-only. Zero-poison
array parity gates the single fit. Code passes ten constructed fixtures;
Panther is inaccessible and no real-data control has run. E19 remains open.
