# Explanation register

**Created:** 2026-09-02

**Status:** active; six model pairs, forest controls and read-only diagnostics complete; all prior failures preserved

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

**Status:** one order-policy contrast measured September 23; class-prior-only
causation remains open. The proposed simple rescue is not observed here.

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

September 23 outcome: after a preserved startup-only failure and NumPy-alias
repair, frozen 579a3f5/job 402291 completed one D-p30 forest. Observed attack
proportion changes 34.86320→48.72032%, but DR changes 63.89140→63.16742,
FA 8.74317→12.56831, AUC 84.37752→79.80343 on the same 1,288 original rows.
Best DR at FA<=17.6/33.3% falls 7.33032/9.86425 points. Thus the policy does
not merely restore the default decision point; measured ranking worsens too.
Counts, generated geometry, bootstrap draws and scale also change, so the
prior's separate contribution is not identified. Unknown synthetic truths
stay training-only. All guards/audits pass; old models and inputs unchanged.
Stop this contrast; no extra setting/seed to seek a desired result. See the
[order-control record](results/poison_balance_20260923/README.md).

## E20 — the first GRU gap demonstrates failure of the prescribed training

**Status:** not established; the run stopped before completing that training.

Job 402378 hit the 900s fit guard after 32 full epochs plus seven batches of
epoch 33, not 50. P30 never started. The preserved p00 model has AUC 80.82379,
DR 42.98643/FA 8.78438; at FA<=6.8 best DR is 35.74661 versus 92.4 printed. Neither
cutoff nor favorable reversal rescues these saved partial scores. This does
not exclude better scores from completed training. Full-epoch loss declined
from .67070 to .40140; no converged plateau is established. Ranking still
exceeds the daily-mean AUC 66.19624 on these rows.

The constructed runtime projection underestimated the actual research fit;
cause remains open. Separate partial-artifact checks verify inputs, labels,
scores, metrics, weights, optimizer updates and history without relabeling it
complete or rerunning inference locally. Stop this attempt; specify the
timing/budget question before approved completion of the unchanged schedule.
No extra seeds or automatic p30. See [record](results/gru_pilot_20260923/README.md).

**Completed-schedule update:** the separately approved same-seed pair now
finishes all 50 epochs. The first 32 p00 epochs match exactly; full p00 AUC
improves to89.07390 and default DR to62.35294. Thus the partial score was not
the final scheduled result. Its old partial status remains unchanged. See E21.

## E21 — completing the prescribed schedule or changing the cutoff closes the GRU gap

**Status:** not observed for the declared completed pilot; other readings,
optimization trajectories and full-population outcomes remain open.

Frozen1f84703/job402625 completed both50-epoch/2250-update fits in56:11.
DR/FA/AUC is62.35294/9.93789/89.07390 unpoisoned and
39.72851/6.12245/79.88605 poisoned. At corresponding FA caps6.8/20.6%, best
saved-score DR51.40271/67.14932 misses92.4/78.5; reversal gives only
.27149/6.24434. This excludes cutoff/sign rescue for these scores/rows,
not every GRU configuration. The poisoned AUC is close to79.4 reported;
one close metric does not reproduce the complete operating pattern.

Useful ranking remains above daily-mean AUC66.19624. Default DR declines
22.62443 points; common-cap declines are10.04525–16.56109, with AUC down
9.18785. Threshold effects are material but not the entire change. Identical
initialization, inputs and repeated32-epoch prefix pass; all local audits
match cluster bytes. The original partial attempt remains preserved.

P00 loss rises sharply around epochs39–40 then partly recovers. The completed
schedule does not establish convergence, a stable attainable ceiling or the
cause of underperformance. Native-ReLU/table versus generic recurrence
interpretations remain untested alternatives; the temporal and ensemble
reconstruction mechanisms are not identified. No further seeds or training
were run after inspecting this result. Proposed next coverage is standalone
AEA source/reconstruction specification. See [completed record](results/gru_completion_20260924/README.md).

## E22 — the selected AEA can exactly reconstruct the standardized inputs

**Status:** false for negative entries if Sigmoid is the reconstruction head;
the subsequent zero-fit pilot check establishes the conditional FA limit in E24.

III-A.1 standardizes training values; TableII selects a Sigmoid output. Under
the natural coordinate-reconstruction reading and declared MSE score, every
weight choice obeys the pointwise output-box bounds in AEA_SPECIFICATION.md.
Zero error on negative entries is impossible; good anomaly detection is not.
Constructed witnesses verify both statements, threshold-boundary behavior,
rounding and error-unit distinctions. No CER arrays were scored for this
finding, no trained score or full-population bound is available, and MAE/
linear-output/alternative preprocessing conclusions do not follow.

The proposed next check is the separately approved zero-fit novelty-input
and score-geometry experiment. Metadata alone reports108 exact train/test
overlaps at p30 and a different test population from the classifier pilot;
these cannot be hidden when comparing models. See
[source specification](AEA_SPECIFICATION.md) and [proposed check](AEA_GEOMETRY_CHECK.md).

## E23 — classification-only training establishes that the intermediate block reconstructs

**Status:** not established by the stated objective; ensemble behavior untested.

IV-B gives a classification loss without an explicit reconstruction term or
benign pretraining. A constructed complement-and-compensate example gives
identical small classification loss with either the input itself or its
complement as the intermediate representation, while reconstruction errors
differ. This is a logical witness that classification loss alone does not
identify reconstruction, not a trained replication or proof that the actual
ensemble cannot learn useful reconstruction incidentally. Standalone AEA
reconstruction, joint classification, pretraining and combined losses remain
separate experiments. No ensemble was implemented or run.

## E24 — some weights rescue the printed AEA false-alarm point under the frozen reading

**Status:** excluded on the frozen novelty pilot for [0,1] reconstruction,
MSE and the printed0.51 cutoff, including declared rounding allowances.

Job402811, freeze5d5d850, performed zero fits. Even the nearest permitted
reconstruction has MSE above0.515+1e-6 for741/3344 benign p00 rows and
748/3344 benign p30 rows. Every bounded reconstruction therefore has FA at
least22.15909%/22.36842%, above the favorable printed caps5.25%/18.45%.
This applies across all weight choices under these fixed conditions, not
only a fitted model's scores. Predeclared RMSE/SSE versions also miss the
FA caps at the same printed cutoff. The DR upper bound remains100%.

The obstruction persists on original benign rows (minimum28.34225%/33.15508%
at favorable rounding). All56 input arrays, scaler transforms, identities,
synthetic ancestry and output calculations verify; local audit matches
cluster bytes. Original inputs are unchanged. The108 exact p30 train/test
overlaps remain a limitation of the retained preparation, not a new holdout.

The lower-bound score's AUC59.44/56.66 is not an AUC ceiling on a learned
model. Other populations, normalization axes/scales, outputs, score functions
or thresholds remain open. In particular, this is not a statement about the
sequential ensemble's classification output or author intent. No further
training can fix this particular point; propose a bounded score/scale
clarification or repair comparison instead of another seed. See
[geometry result](results/aea_geometry_20260924/README.md).

## E25 — score/scale alternatives may pass a free-cutoff relaxation

**Status:** mixed; original feature-z MSE/global-z MSE p00 remain excluded by
optimistic oracle bounds, while MAE/raw-unit MSE/min-max are not excluded
when the cutoff is free. That does not imply a pass at the printed 0.51.

Corrected job403409, after prelaunch failure403408, evaluated five declared
zero-fit branches using true labels only to maximize possible separation.
Feature-z MSE p00 upper DR at favorable FA cap is88.24405 versus94.05;
global-z MSE is92.26190. Their common p00 cutoff intervals are empty.
Feature-z MAE, raw-unit MSE and training-only min-max MSE have optimistic
envelopes that pass; this is only failure to exclude, not evidence a shared
network can attain the oracle. MAE still fails at 0.51 with minimum
FA36.75239%/33.79187%; raw-unit MSE p00 fixed-cutoff maximum DR90.02976%
also misses. Three p30 DR values in the original README used the wrong cap;
the report is corrected but raw JSON remains unchanged. No branch search or
neural fit followed. See E27.

All56 inputs/ten archives/prior geometry artifacts pass local/cluster audit.
The108 p30 train/test overlaps and training-only statistics remain explicit.
The first job failed before data load from an output-path typo and is preserved.
Next question is textual justification of a surviving score/scale branch,
not automatic AEA implementation, ensemble training or extra seeds. See
[repair record](results/aea_repair_20260925/README.md).

## E26 — a constructed AEA fixture does not authenticate the MAE branch

**Status:** MAE branch promotion withdrawn; experimental software only.

The previous “least invasive surviving” claim conflated free-cutoff and
fixed-cutoff results. The prototype remains separate from direct reproduction
files and CER data. It uses mirrored Sigmoid LSTMs, decoder-query attention,
free-running reconstruction and SGD; loss now requires an explicit choice.
Its old four-test assurance was overstated: the query test passed with that
path disabled, and saved-model loading failed. Six revised tests address the
software defects without establishing useful learning. No branch is selected
for training. See E27 and [implementation scope](AEA_IMPLEMENTATION_ENVELOPE.md).

## E27 — independent recheck distinguishes valid bounds from our reporting errors

**Status:** verified corrections; original evidence preserved.

The user requested a fresh check of the recent steps. Source reinspection
confirms the normalization axis and standalone training loss are omitted.
Independent piecewise endpoint, direct pair-count AUC and order-statistic
calculations reproduce the geometry and repair archives. Original auditors
also reproduce preserved outputs exactly. Thus E24's fixed-cutoff exclusion
and E25's original/global MSE p00 all-cutoff exclusions remain intact within
their pilot scope. Neither establishes author intent or ensemble failure.

MAE's necessary favorable cutoff intervals [0.78932,0.97132) and
[0.60363,1.25402) exclude 0.51. The repair report's p30 DR bounds at the proper
18.45% FA cap are 100/100/96.81548% for feature-z/raw-unit/global-z MSE.
The prototype now has a causal first-query witness with a disabled-query
negative control and exact fresh-process reload. All24 small-fixture weight
arrays/output/attention remain identical after the serialization refactor.
These are software checks, not useful reconstruction evidence. No new
research experiment, selected repair branch or website change. See the
[recheck record](results/aea_recheck_20260925/README.md).

## E28 — standalone AEA repairs determine how to test the sequential ensemble

**Status:** not required by the source; separate experiment specified.

The complete target reread on September26 confirms IV-B, p2682 trains all
ensemble components through classification loss; IV-C independently chooses
ReLU hidden activations, Adam, no dropout and constraint1. Standalone AEA's
Sigmoid/SGD, reconstruction-error score and0.51 cutoff cannot silently be
carried over. The standalone mathematical exclusions remain valid on their
own prepared inputs; no downstream classification bound follows from them.

The [proposed ensemble pilot](SEQUENTIAL_ENSEMBLE_PILOT.md) retains the
original two-class inputs and declares corrected BCE, native cells,
intermediate scalar ReLU projection, final Sigmoid probability and0.5 rule.
Intermediate activation and other omissions remain interpretations, not
newly discovered author settings. No pretraining/reconstruction objective is
added. A standalone AEA fit remains unfinished coverage, not a prerequisite
to this joint-training interpretation. Existing GRU comparisons are not
matched ablations because their downstream settings differ.

The next empirical gates are constructed learning, causal query/feedback,
serialization and full-model variable-batch timing. None has been completed
for the ensemble. The contract specifies the bounded pair and competing
outcomes; it reports no fit, timing, causal mechanism or attainability result.

## E29 — connected gradients and passing software checks establish usable ensemble training

**Status:** insufficient; the declared constructed learning gate failed.

The September27 direct ensemble implementation has the specified9,240,802
parameters and passes structure, causal query/feedback, update/constraint and
fresh-process persistence checks. In the predeclared5,082-parameter learning
fixture, all groups initially have finite nonzero BCE gradients and change
weights. Yet300updates achieve100%/BCE.2253 for normal labels and50%/BCElog(2)
for reversed labels, from identical initial weights. Both fits completed.

The saved reversed model's last GRU and classifier hidden outputs are zero
on constructed test inputs; AEA intermediate outputs remain positive. Final
upstream gradients are zero on its constructed training inputs. This locates
a collapsed final state, without identifying which training step caused it.
Complementing normal-model probabilities solves reversed labels with no
optimization, so the architecture can represent the reversed answer.

The small fixed training configuration fails its learning gate; the full-width
model, CER performance, other source interpretations and broad attainability
remain untested. No GPU timing follows. A writer failure was repaired by
reloading the completed normal fit, not repeating it; the reversed fit was run
once. See [record](results/sequential_constructed_20260927/README.md).

## E30 — the reversed fixture loses its training signal only late in optimization

**Status:** contradicted for the fixed small trajectory; collapse after update2.

The50-update diagnostic replay matches the prior history exactly from the
same initial weights. GRU8 and classifierhidden become all zero after the
second Adam update on constructed train/test inputs; earlier stages stay
active. Feature gradients are zero from that point through update50. Both
stages change in the same update; finer ordering is not identified.

Fixed old/new input/parameter comparisons show that each block's updated
parameters can zero its outputs even on the old inputs. The head's bias
change dominates its local affine change; reverting only the GRU8 candidate
bias restores some activity, and injecting only its new candidate bias into
the old GRU state kills the output. Native/independent recurrence agrees
within8.93e-12. These saved-state interventions use no optimizer updates and
do not establish that a modified model would train successfully.

The tiny-width learner's early dead-ReLU behavior is now localized; the
full-width model remains untested. Do not infer universal failure, select a
new optimizer or waive the learning gate. A bounded published-width
constructed check is the next proposed discriminator. See
[trace record](results/sequential_trace_20260927/README.md).

## E31 — using the published widths rescues the constructed learning gate

**Status:** not observed in the fixed pair; both label orientations fail.

The September28 width-only comparison retains the old8-step constructed
arrays, initializer seed, Adam and all other settings, changing widths to
the declared9,240,802-parameter model. Both300-update fits stay at50%,
BCElog(2), with probability.5. Initial/final gradients are already zero.
Active decoder features feed a scalar projection whose preactivations are
entirely negative; its ReLU output and every downstream stage are zero.
Both label orientations have identical initial and final weights.

This does not establish that width alone caused either failure: the narrow
reversed run shut off later stages after two updates, while the wide pair
starts with an inactive intermediate bridge. The bridge shape/activation is
a recorded interpretation, not an explicit paper instruction. Neither
failure transfers to a different interface,48-step learning or paper data.

The failed gate holds GPU timing and the real-data pair despite their
conditional authorization. Next review the source interface and separately
declare any alternative/control; no automatic activation/seed search.
See [record](results/sequential_full_width_20260928/README.md).

## E32 — the old ReLU bridge failure rules out other declared interfaces

**Status:** contradicted on the fixed eight-step constructed task; research
learning remains a separate question.

Source recheck leaves the readout equation/activation ambiguous. The primary
scalar-Sigmoid interpretation and separate scalar-linear control were declared
before results, with Sigmoid selected for numerical continuation in advance.
Both normal/reversed learning cases at both activations reached100% held-out
accuracy and clipped BCE about1e-7 after300updates, with the same initial
weights/data/settings as the failed full-width ReLU fixture. Old ReLU defaults
and reload behavior remain exact. This closes the claim that the previous
failure necessarily transfers to those two interfaces on that constructed task.

The activation affects decoder feedback as well as GRU input; the comparison
does not isolate one of those paths. Full48-step constructed GPU timing also
passes (job407255,0:0,8:01;1845.84s projected per50 epochs). Neither result
establishes CER discrimination, unique fidelity to the underspecified source,
or reconstruction-mediated robustness. The fixed research pair407294 is a
separate numerical test. See the [interface record](results/sequential_interface_20261001/README.md)
and [preflight](results/sequential_preflight_20261001/README.md).

## E33 — cutoff choice accounts for the completed sequential pilot's miss

**Status:** excluded for the two saved constant scores; failure cause and
other weights/source interpretations remain open.

The scalar-Sigmoid interpretation completed both50-epoch/2,250-update
research fits at3705bcca (Panther407294,0:0,1:21:28). Test probabilities
are constant0.504759/0.353959, withAUC50/50. The0.5 cutoff flags all/none,
but any cutoff or reversal gives bestDR0 under the2.9/5.8 FA caps, including
favorable rounding allowances. Both defaults equal the constant-prior
decision reference. No stable-robustness claim follows from unchanged50%AUC.
Initial test probabilities were also constant at float32 precision.

P00 intermediate profiles still differ, whereas p30 intermediate outputs
are almostzero. P30's improved MSE merely matches the zero-output baseline;
it is not evidence of useful reconstruction. The same-row forest, FF and
earlier GRU show useful ranking, but differ in architecture/settings, so
this does not isolate a causal effect of adding the AEA.

All source/input/artifact/metric/initialization/history/weight/config/optimizer/
norm and compute-node reload checks pass; local pair/comparison audits match
cluster bytes. The eight-step learning pass and full48-step timing check did
not establish useful48-step research learning. Stop the pair. Next propose a
bounded zero-fit inspection locating where profile differences cease to
affect the saved initial/final classifier, before another fit or seed.
No such job is launched. Source interface/native-cell ambiguity and the
full-population/mechanism/attainability questions remain open. See
[the complete record](results/sequential_pilot_20261001/README.md).

## E34 — constant probabilities mean the whole network is disconnected or zero

**Status:** contradicted as a common explanation for the three observed
states; different precision/attenuation/saturation effects are visible.

Fixed zero-fit inspection (3127b19, Panther408550,0:0,2:11) observes shared
initialization and finalp00/finalp30 on the first100 train/test rows. Initial
native logits retain1.81e-8 test-profile variation, hidden by float32 Sigmoid
resolution; widened readout probability range is4.54e-9. Native stable BCE
gradients remain connected. This arithmetic does not establish useful ranking
or a viable full float64 network.

Finalp00 GRU8 maximum is5.92e-18 with profile range5.05e-23; classifier
hidden output is profile-identical but not allzero. Finalp30 encoder/decoder
maxima reach3.12e9/1.39e15, and its Sigmoid bridge is exactlyzero fromstep2
through48 on both batches. GRU2 onward is profile-identical. No whole native
stage is entirelyzero; input-gradient L2 is8.05e-28/1.69e-28 at the final
states. Tiny nonzero derivatives differ from a broken graph or exact zero.

Native and widened final readouts are constant for both trained states.
All source/input/output/model/optimizer hashes and saved-test parity pass;
the decoder observer is exact, and the local audit matches clusterbytes.
These endpoint observations do not determine onset, identify a single gate/
bias cause or transfer to other populations/source choices. Stop this check;
next proposed question is saved recurrent gate/cell arithmetic before any
repair or fit. No new job or trained branch follows automatically. See
[the saved-state record](results/sequential_saved_state_20261002/README.md).

## E35 — gate equations distinguish candidate shutoff from recurrent amplification

**Status:** supported for the saved fixed-weight trajectories; training onset
and a successful repair remain unidentified.

The frozen gate check (ee1a420, Panther408551,0:0,3:05) repeats the same100
test profiles at shared initialization/finalp00/finalp30. All21 cell traces
match native h/c arithmetic and saved outputs exactly, with no updates.

Finalp00 GRU8 has500/400/200 positive candidate coordinates in steps1/2/3,
then zero of30,000 throughout4–48. Update gates remain near0.5, so the
retention-product identity explains the final5.92e-18 maximum state. The
maximum-final-state witness haspositive bias; increasingly negative input
projection overtakes it. Negative bias is not a universal cause. Locally
omitting bias activates13,000 candidates atstep48 but does not test a
changed trajectory or establish that bias removal repairs learning.

Finalp30 LSTM cell growth contains admitted positive recurrent candidate
injection, not amplification by the forget factor alone. The first encoder's
maximum-final-hidden witness hasi=f=o=1 atstep48; its recurrent contribution
26951.291 dwarfs input0.281. Retained72659.45 plus new26951.60 gives
cell/hidden99611.05 to displayed rounding. Encoder/decoder stages amplify
further, to about1.39e15 in decoder3. These selected extrema follow a
predeclared witness rule and are not population prevalence claims.

All data/state/source/artifact checks pass, and the local audit matches
clusterbytes. These endpoints do not identify when or which training update
created the conditions. The native ReLU completion is still one reading of
conflicting source descriptions. Next define one explicit recurrent-cell
alternative/control with full48-step constructed validation before further
research fitting; no alternative or newjob is selected/launched here. See
[gate record](results/sequential_gates_20261002/README.md).

## E36 — one source-explicit recurrent alternative can learn the full48-step fixture

**Status:** supported for the two fixed tanh cases; CER performance remains
untested for this alternative and the reference matrix is incomplete.

Source review preserves the conflict between Algorithm1's explicit tanh
formulas and IV-C's ReLU selection. I-SEQ-tanh-cells changes only recurrent
cell activation, retaining Sigmoid gates/bridge, ReLU Dense, all dimensions,
initial weights, optimizer, constraints and objective. The existing ReLU
default and historical archives remain valid.

Both tanh label orientations complete300updates on the same32train/32test
48-step constructed profiles at100% held-out accuracy/clippedBCE~1e-7.
Finite gradients, constraints and fresh-process GPU reload pass. ReLU-normal
becomes nonfinite at245; reversedReLU is held by the stop rule. Its brief
loss improvement to.48902 at239 precludes an unqualified plateau claim.
The failed reference has invalid final predictions, not a valid50% score.

The task is solved by a zero-parameter mean rule; reversed labels flip both
train and test labels and are not a poisoning test. This validates basic
learning in one finite setting, not robust theft detection, reconstruction
mechanism or a uniquely faithful source implementation. The LSTM and GRU
activation changes are coupled, not separate causal ablations.

Preserve the raw incomplete comparison and all attempts. A pre-fit device
placement error was corrected with exact initial-array parity, no scientific
setting change or trained retry. Three allocations total21:44, within35min;
the last was verification only. No CER fit or unrun reference continuation.
Next specify/wire/timing-gate a fresh-weight CER pilot for the fixed tanh
alternative before launch. See [record](results/sequential_cells_20261002/README.md).
