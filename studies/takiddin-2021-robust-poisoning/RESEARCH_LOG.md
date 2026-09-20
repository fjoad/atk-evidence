# Can this detector learn from corrupted labels?

Updated: 2026-09-20

This is our working journal for *Robust Electricity Theft Detection Against
Data Poisoning Attacks in Smart Grids*, by Takiddin and colleagues (2021).
We are rebuilding the experiments to find out whether the reported results
can be recovered. We record what we notice, why it matters, what we decide to
test, and how the evidence changes our view.

**Where we are, 20 September 2026:** the data are verified, the preparation
choices are recorded, and the first random-forest pilot and matched preparation
controls have completed. Generating synthetic data before splitting materially
improves the apparent performance in this pilot. Removing that access and
holding out whole source days makes the evaluation stricter, but useful
discrimination survives. The full paper and its other models remain untested.

The paper's starting point is straightforward. An electricity meter reports a
customer's usage. The study takes real consumption readings and alters them
to simulate theft. It then deliberately gives some theft examples the wrong
training label: “normal.” A detector that learns from these corrupted labels
may have more trouble recognizing theft later. The paper calls this a data
poisoning attack.

Its proposed solution combines an autoencoder intended to reconstruct the
readings, attention that weights parts of the sequence, a recurrent network
called a GRU, and a final classifier. Table V says that increasing poisoning
from 0% to 30% reduces detection from 95.2% to 92.2%, while false alarms rise
from 2.9% to 5.8%. We want to establish whether that behavior follows from the
published procedure, and whether the baseline models behave as reported too.

[Read the paper's publication record](https://doi.org/10.1109/TSG.2020.3047864).
The page numbers below refer to the journal's printed pages.

## 2 September — An earlier arithmetic check

*This entry was written retrospectively on 20 September from the saved record.*

An earlier pass transcribed the result tables and checked relationships between
their metrics. Those calculations and the original interpretation remain in
the [source-audit record](SOURCE_AUDIT_FINDING.md). No model was trained in
that work.

When we restarted the investigation, we decided to revisit the assumptions
behind those checks. For example, the paper says Table IV averages results
over customers. Averaging several customers' precision values need not give
the precision calculated from their combined predictions. A relationship that
holds for one confusion matrix can therefore fail after averaging. We need to
respect the table's aggregation before interpreting an apparent inconsistency.

This is one reason to keep a journal: a concern can remain worth examining
while the explanation for it changes.

## 20 September — Starting with the setup, and including every model

Our working hypothesis is that the published performance will be difficult to
recover even after plausible missing details are filled in. The investigation
has to test that expectation.

We reread all ten pages, including the equations, model diagram, training
algorithm, and tables. We decided to cover all seven baseline models—random
forest, AdaBoost, SVM, ARIMA, feed-forward, GRU, and attention autoencoder—and
both proposed ensembles. Simple baselines are useful checks because their
implementations leave fewer architectural decisions to resolve.

The fresh reading produced four questions that affect what we would actually
run. They are recorded below before any model results are available.

## 20 September — What exactly gets poisoned?

Section III-A.3, page 2678, gives two definitions. For a generalized detector,
the percentage refers to **customers** presenting poisoned data. For a
customer-specific detector, it refers to **that customer's samples**. Later
discussion describes percentages of training data more loosely.

These definitions can produce different experiments. Selecting ten of a
hundred customers does not necessarily select a tenth of the training
examples: customers may contribute different numbers of usable days. We also
need to know which of the six theft transformations supplies the mislabeled
examples, whether examples are replaced or added, and whether labels are
changed before or after synthetic balancing.

**Our decision:** write the poisoning operation precisely, retaining customer
and day identities and both the true and corrupted labels. Any unresolved
interpretation will be named before its performance is measured.

**Next check:** construct a small, hand-checkable example and verify exactly
which training rows and labels change. Keep the test labels correct. This
check has not run yet.

## 20 September — The printed loss loses the label

Equation (1), page 2679, is described as cross-entropy. A loss is the penalty
the model tries to reduce during training. It should reward a prediction when
it agrees with the example's label and penalize it when it disagrees.

Let `y` be 1 for theft and 0 for normal consumption, and let `p` be the model's
predicted probability of theft. The printed binary expression uses `log(p)`
in both terms. For each example it simplifies to:

```text
−[y log(p) + (1 − y) log(p)] = −log(p)
```

The label `y` disappears. Predicting a high probability of theft receives the
same small penalty for a normal example as for a theft example. Minimizing
this expression encourages high theft probabilities for every example.

The usual binary cross-entropy expression instead contains `log(1 − p)` in
the second term:

```text
−[y log(p) + (1 − y) log(1 − p)]
```

That change makes the penalty depend on whether the prediction agrees with
the label. It is an obvious possible correction to the printed equation.

**What we established:** the equation as printed has this algebraic defect.
**What remains unknown:** which loss the authors' code used.

**Our decision:** preserve the printed error and use standard binary
cross-entropy as a declared plausible repair when testing the intended
classifier. Failure of the printed formula alone would not settle whether
the intended model can reproduce the results. The repaired model has not
been trained yet.

## 20 September — What makes the intermediate output a reconstruction?

The proposed network first passes a day's readings through the attention
autoencoder, then through GRU layers, then through a classifier. Section IV-B,
page 2682, calls the intermediate output reconstructed data and says the
combined network's weights are trained to minimize the classification loss.

A classification loss measures whether the final theft prediction is right.
A reconstruction loss measures how closely an intermediate output copies the
input readings. These are different training goals. The section specifies no
additional reconstruction-loss term or benign-only pretraining step for the
combined network.

This leaves a question about the proposed explanation. The intermediate
representation might be useful for classification without closely
reconstructing consumption. Its name and position in the diagram do not
establish what it has learned.

**Our decision:** implement the stated joint classification training and
identify plausible alternatives separately: reconstruction pretraining and a
combined classification/reconstruction objective. The standalone autoencoder's
training procedure must not silently become the combined model's procedure.

**Next check:** inspect what the intermediate output actually contains and
whether reconstruction training changes detection or poisoning robustness.
We have no measurements answering this yet.

## 20 September — The evaluation itself can change the problem

Section III-A, page 2677, uses ADASYN, a procedure that generates additional
examples of the less common class. For novelty detectors, the paper adds
synthetic normal examples to the **test set**. For two-class detectors, it
balances the data **before the training/test split**.

This changes which examples the detector is judged on, and the second
procedure can put related examples on both sides of a split. Consequently,
data preparation could contribute to the reported performance. Its direction
and size need measurement.

We also noticed that Table V's headline of a three-percentage-point detection
drop accompanies a doubling of the false-alarm rate:

| Quantity | No poisoning | 30% poisoning |
|---|---:|---:|
| Theft examples detected | 95.2% | 92.2% |
| Normal examples incorrectly flagged | 2.9% | 5.8% |

A detector's cutoff controls how readily it raises an alarm. Lowering that
cutoff can catch more theft while flagging more normal examples. We therefore
need to compare both the printed operating points and detection at the same
allowed false-alarm rate.

**Our decision:** reproduce the stated preparation, preserve the original and
synthetic examples' identities, and keep a separate conventional evaluation
with untouched test examples. Examine the full detection/false-alarm tradeoff
for each saved score vector. A cutoff check concerns those particular scores;
new training can produce a different ordering.

**Next check:** verify the preparation sequence and evaluation populations
before comparing any models. Neither preparation nor scoring has run yet.

## 20 September — Looking for code before writing our implementation

We checked the paper for a code link, the author's publication listing,
publisher metadata, GitHub using the exact title and DOI, and Zenodo using
the DOI. We also searched for a corresponding code archive through the web.
We found the publication and an author-hosted PDF, but no matching public
implementation in the sources checked.

The IEEE article page presented a verification barrier, so its supplementary
material area could not be inspected. The accessible Crossref record listed
the article and PDF but no related software record. These limits are part of
the result: **we have not located public code**. This search does not establish
that code was never released or explain why it was unavailable to us.

The [dated search record](CODE_AVAILABILITY.md) preserves the queries, links,
results, and access limits. We can now proceed with a reconstruction from the
paper while remaining able to inspect any implementation found later.

## 20 September — A large gap should change the next question

We also agreed on a rule for avoiding uninformative reruns. Suppose a model
achieved 30% against a reported 90%. These are hypothetical numbers, not a
result from this study. Our first response would be to investigate the gap
before launching three more training seeds.

We would first inspect the saved data, labels, scores, metric calculations,
and training history. Did the model receive the intended examples? Did its
weights update? Are we measuring the same quantity? Does a simple control
reveal a shared preparation problem? These checks can identify a mistake that
repeated training would simply reproduce.

The next experiment then needs a specific purpose. If the proposed explanation
is a poor cutoff, inspecting the saved scores may answer it without retraining.
If an unstable fit could explain the discrepancy, a bounded repetition can
test that. If learning is still improving, a measured extension may answer a
different question from changing the seed.

**Our decision:** before an extra run, write down the explanation it tests,
the outcomes that would change our next decision, and its maximum cost and
stopping rule. If neither outcome would change what we do or can conclude, we
should not spend the compute.

Repetitions still matter when we estimate variability or test a statistical
claim. A large gap does not tell us how variable the model is. The point is to
use repetitions to answer that question deliberately, after establishing a
sound setup. Testing all reported models remains part of the plan.

## 20 September — Finding the data and checking the attack definitions

The consumption archives were already present locally and on the cluster.
We independently checked them against the current official ISSDA metadata:
all six filenames, file sizes, and checksums match, and the compressed files
pass their integrity checks. The customer-type CSV contains 4,225 residential
meters. We compared its allocation information with a separate public workbook;
all 6,445 rows match. Our initial workbook reader mistakenly assumed a header
row. Checking the first row exposed that mistake, and including it resolved
the apparent mismatch.

The paper never identifies its 3,000 customers. We will select them using a
fixed random seed before looking at performance, and identify that as our
choice. Matching the source files does not recover the authors' exact sample.
The [data-source record](DATA_SOURCES.md) preserves both facts.

We also found the cited attack study in its author's thesis. Chapter 5,
page 117, confirms uniform reduction factors between 0.1 and 0.8 and a bypass
lasting four to 24 hours. That study used hourly profiles. Our target uses
half-hourly readings and numbers the mean-based attacks differently. We follow
the target's equations and record how we translate durations and endpoints.

These checks let us write the [initial preparation contract](PREPARATION.md).
It states the split, scaling, balancing, poisoning operation, and remaining
alternatives before any model outcome is available.

## 20 September — Checking the preparation code before using it

We implemented the novelty and two-class preparation paths and checked them
on constructed software fixtures. Thirteen tests passed. They cover the six
attacks, incomplete and duplicated days, records crossing file boundaries,
poisoning, training-only scaling, and saving/reloading the prepared arrays.

We use the standard ADASYN implementation for its generated values. Recording
its neighbor choices and replaying its seeded interpolation draws lets us
retain both parent examples for every synthetic row. The instrumented output
matches the uninstrumented library exactly in the test, and every generated
row is checked against its recorded parents.

The implementation also makes a source ambiguity visible. A novelty detector
starts with only benign training examples. To interpret poisoning, we replace
selected training examples with attack versions while leaving their training
labels as benign. If we also retain all six attack populations in testing,
some training attack profiles occur in the test pool. The code measures that
overlap. This is a consequence of our declared completion; the paper does not
explain how its own implementation resolved it.

The next check is a small preparation-only cluster job: 20 seeded customers,
their first 28 complete days, both paths at 0% and 30% poisoning, and one
customer-specific example. Its budget is 15 minutes on four CPU cores with
16 GiB of memory. No detector will be trained. This will check that the
pipeline works on the actual files and expose its real counts and overlap.

## 20 September — The first preparation check passed, and exposed two choices

The cluster job finished successfully in 2 minutes 16 seconds using CPU
resources only. It read all six source archives, selected 20 customers, and
kept their first 28 complete days: 560 original daily profiles. Both
preparation paths completed at 0% and 30% poisoning, together with a
single-customer example for each path.

After copying the outputs back, we checked all 224 saved array files. Their
hashes, labels, class counts, scaling, and synthetic-parent calculations agree
with the saved records. These checks establish that the preparation did what
we specified. They do not supply a model-performance result.

The outputs make two consequences concrete. First, selecting 30% of customers
in the generalized two-class setup changed 675 of 4,464 training labels:
**15.12% of training rows**. Selecting 30% of all training rows in the
customer-specific interpretation changed 67 of 224 labels: **29.91%**. These
are different corruption strengths, despite both being labeled “30%.”
The paper's wording leaves this distinction unresolved, so any comparison of
the two detector types will have to acknowledge it.

Second, synthetic examples can have parents across the two-class train/test
split. We recorded 1,306 links from synthetic test examples to original
training examples in the generalized setup. Separately, our novelty-poisoning
completion placed 108 identical attack profiles in both training and testing.
These are observations about this implementation of the recorded choices.
They do not show which choices the authors used or how much the overlap changes
detection performance.

**What changed:** we now have verified data and an executable preparation
pipeline, plus concrete checks for two ways the setup could affect the claimed
comparison. We can begin model implementation without hiding those choices.
The [complete preparation record](results/preparation_20260920/README.md)
contains the counts, timings, verification, and original program output.

## 20 September — Giving the first baseline a specific question

The next step is the paper's random forest with 100 trees. We have recorded
its complete settings and the exact prepared inputs before fitting it. The
first pair will use the existing 0% and 30% generalized two-class samples.
That is enough to check the model and inspect its behavior, but it is a small
pilot rather than a reproduction of the paper's full population.

We checked the metric definitions again. This paper's accuracy counts all
correct predictions divided by all examples. Balanced accuracy averages the
two classes' success rates. They can differ when the classes are unequal, so
we report the paper's ordinary accuracy and keep balanced accuracy separately.

Before using real data, constructed software examples check that the forest
learns an obvious class difference and that flipping training labels changes
what it learns. This checks that our training call actually uses the corrupted
labels. Saved models must give identical predictions after reloading.

We will record all seven metrics, performance by attack type, and false
alarms on original versus synthetic benign examples. A constant prediction
and a simple daily-consumption score provide reference points. Inspecting
every score cutoff will help distinguish poor ranking from a poor choice of
decision threshold, without retraining.

The [first-baseline contract](FIRST_BASELINE.md) fixes the two fits, one seed,
and a 15-minute CPU allocation. If the forest works well, we will report it.
If it works poorly, these saved checks should tell us which explanation to
investigate before spending more compute. No fitted result is available at
the time of this entry.

## 20 September — The first baseline works, and the cutoff changes the story

The first two random-forest fits finished successfully. Each used 100 trees,
4,464 training rows, and the same 2,232 test rows. The entire CPU job took
16 seconds; the individual fits each took less than a second. Both saved
models produced identical probabilities and predictions after reloading.

Here is what happened on this small, 20-customer pilot:

| Training condition | Attacks detected | False alarms | Ranking AUC |
|---|---:|---:|---:|
| No poisoning | 92.31% | 3.02% | 98.55% |
| 30% of customers selected for poisoning | 61.18% | 0.44% | 94.36% |

Detection is the fraction of theft examples flagged. False alarms are the
fraction of normal examples flagged. AUC describes how well scores rank theft
above normal examples across possible cutoffs; 50% is chance-level ranking.

**This changes our working expectation.** The unpoisoned forest performs well,
including against the two simple controls we tried. We cannot carry forward
an assumption that every baseline will perform badly. These measurements are
on a small construction, so they also cannot establish that the paper's
full-data results have been reproduced.

The poisoning result initially looks much worse if we read only detection:
it falls by 31.13 percentage points. But false alarms also fall, and AUC
remains high. That led us to the cutoff checks recorded before this run.

Using the already saved scores, and allowing at most 17.6% false alarms,
detection can reach 97.29% without poisoning and 92.04% with poisoning—a
5.25-point difference. At the 33.3% false-alarm cap, the corresponding
detection rates are 99.19% and 95.02%. No model was retrained for this check.
It shows that much of the default-decision decline can be changed by moving
the cutoff, while some ranking deterioration remains.

We chose these favorable cutoffs using the test answers. That makes them
diagnostic limits for these scores, rather than thresholds we have validated
for future data. Still, they rule out an interpretation that the poisoned
forest has lost all useful discrimination in this pilot.

The original/synthetic breakdown raises the next question. Without poisoning,
false alarms are 6.56% on original normal examples and 2.33% on generated
normal examples. With poisoning, they are 2.19% and 0.11%. The original-row
AUC remains high too, but excluding generated test rows from a calculation
does not undo their use during training or their relationships across the
split.

**Our next question:** how much of the strong performance depends on that
split and resampling procedure? A matched preparation control would address
it directly. Repeating training seeds on the same construction would address
a different uncertainty. The remaining baseline and proposed models stay in
scope.

The [first-baseline record](results/rf_pilot_20260920/README.md) preserves all
seven metrics, the paper's full-data values for context, per-attack results,
simple controls, timings, and artifact checks. All input/output hashes passed;
the locally recomputed comparison matches the cluster's file byte for byte.

## 20 September — Separating two possible effects of preparation

We decided to follow the resampling question with two controlled changes,
using the same 20 customers, original days, existing attacks, forest settings,
and seed. The previous forest predictions remain the reference; we will not
retrain them.

First, we will keep the original training and test examples in their existing
places, but generate additional training examples using training data only.
This removes access to test examples during synthesis. ADASYN's rounding can
change the final training count slightly, so we will record the actual counts.

Second, we will assign whole source days to training or testing. A normal day
and its six altered versions will stay together. This tests whether training
on related versions of a test day contributed to the earlier performance.
It still uses the same customer cohort; it does not test new customers.

Changing the split also changes which rows can be evaluated safely. We will
therefore compare all three versions on the intersection of the earlier
original test set and the newly held-out days. The rule for selecting that
intersection is fixed before inspecting predictions. Every comparison will
use identical test rows and true labels. We will also compare the first two
versions on the larger original test set they share.

Eight constructed-input tests passed. They verify matching identities and
ordering, intact attack families, training-only synthetic parents, unchanged
test labels, and removal of source-day overlap where required. The
[control contract](SPLIT_RESAMPLING_CHECK.md) fixes four new fits—two
preparations at two poisoning levels—inside one 15-minute CPU allocation.
If performance remains high, that outcome will be recorded too. At the time
of this entry, these controls have not been fitted on the real observations.

## 20 September — Preparation matters, but the baseline does not collapse

The four controlled fits completed on Panther in one CPU job. We reused the
original forest's predictions and compared all three preparations on the same
445 original test examples: 386 attacks and 59 normal examples. The shared
test set was chosen from identities before inspecting scores. All customers
still come from the same 20-customer pilot.

Here are the ranking results on those identical examples:

| Preparation | AUC without poisoning | AUC with 30% of customers selected |
|---|---:|---:|
| A: generate synthetic examples before splitting | 98.32% | 90.98% |
| B: generate them from training examples only | 91.84% | 81.99% |
| C: also keep whole source days out of training | 93.18% | 82.97% |

The first change matters. AUC drops by 6.47 and 8.99 percentage points.
False alarms rise from 3.39% to 33.90% without poisoning, and from 1.69%
to 15.25% with it. On the larger 1,288-row original test set shared by A/B,
the same comparison also lowers AUC and raises false alarms. The effect is
not merely that synthetic test examples were easier: neither side of these
comparisons includes synthetic test rows.

We should not label the whole difference a precise measurement of leakage.
Moving synthesis into training also changes the generated values and slightly
changes the final training count. A and B contain 4,464 and 4,532 training
rows. The result identifies a consequential preparation policy, with those
changes included.

The second change did **not** cause a further collapse. C's AUC is slightly
higher than B's in both conditions. This is one split, and grouping changes
training membership too; it does not prove related days never matter. It does
show useful learning after these two sources of dependence are removed.
The simple daily-consumption reference has AUC 67.63% here, compared with
C's 93.18% and 82.97%.

The detection/false-alarm trade-off still matters. In C, default detection is
94.30% without poisoning and 62.44% with it, but false alarms also change
from 32.20% to 11.86%. At a common maximum of 17.6% false alarms, the best
cutoffs on the saved scores give detection of 91.19% and 69.95%. These are
test-label-chosen diagnostic cutoffs, not thresholds validated on fresh data.
Poisoning still damages ranking: C's AUC falls by 10.21 points.

**What changed in our thinking:** the original preparation contributes to
the unusually favorable first result, but it does not explain all useful
discrimination. We have evidence for a narrower setup criticism, not evidence
that this baseline cannot work. We will not repeat seeds simply to look for
a worse result or assume this finding transfers to another model.

The common evaluation has only 59 normal examples, so each extra false alarm
moves its rate by about 1.69 percentage points. Customers and related days
are dependent; no population confidence interval is claimed. C tests held-out
days from the same customer cohort, not new customers or a later time period.
This is still a controlled pilot, not a reproduction of the full paper.

All four models passed save/reload checks. The cluster verified all 112 new
input arrays, training-only synthetic parents, matching test values and labels,
and zero shared source days in C. All matched results were recomputed from
saved predictions. The [complete record](results/split_control_20260920/README.md)
includes every metric, the actual poisoning fractions, timing, and audit files.
The job took 1 minute 48 seconds within its 15-minute limit; no additional
model or seed was run after seeing the outcomes.

## 20 September — Pinning down the next baseline before fitting it

The next approved model is AdaBoost. The paper specifies decision-tree weak
learners (p.2678) and 50 estimators (p.2679), but not tree depth, learning rate,
algorithm variant, or software version. We will reuse the original two-class
pilot exactly, at 0% and 30% customer poisoning. We will not replace its
preparation with the stricter forest controls and call that the printed method.

This rereading also corrects our earlier source note: the paper does identify
decision trees as the weak-learner family. It is their depth and other settings
that remain unspecified, not the family itself.

There is a consequential software choice here. The
[historical scikit-learn default](https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
was SAMME.R with depth-one trees and learning rate 1. Current versions no
longer provide that variant. We chose stock scikit-learn 1.5.2 in a separate
pinned environment because it still provides SAMME.R. That is an explicit
completion, not a claim to know the authors' software or reproduce every
detail of an older library. The alternative SAMME algorithm remains untested.

The [AdaBoost contract](ADABOOST_PILOT.md) fixes two fits, one seed, and a
15-minute CPU budget. We will record all seven metrics, original/synthetic
breakdowns, simple controls, and cutoff diagnostics at both AdaBoost's printed
false-alarm rates and the previous forest's rates. The saved weak learners
must show actual training and survive exact save/reload checks. A perfect
fit may legitimately stop before 50 trees; we will report the actual count.
No performance on the research observations is known at this point.

Adding another model extends the same direct implementation files. Earlier
experiments remain bound to their immutable Git revisions and saved hashes,
not retroactively rewritten to match today's code. Historical audits now
check those revision bytes when a file has since been extended.

All seven AdaBoost fixture tests passed in the pinned environment. The main
repository suite passed 307 tests, with four AdaBoost fit tests skipped there
because they require that separate environment; those four passed in it.
The earlier forest-control audit still matches its saved output exactly.
Panther's login node killed the environment bootstrap before any fit; setup
then completed in a separate 67-second, one-CPU allocation. No research-data
experiment was part of that setup job.

## What we will do next

The random forest and this matched preparation check are complete. The next
distinct question is whether another reported shallow baseline shows similar
poisoning and cutoff behavior on the original verified pipeline. A bounded
AdaBoost pair is the proposed next step, after recording its library/version
choices. We have not launched it. Continue through the remaining models with
their reported settings; the objective remains coverage of the complete paper.

The paper specifies 50 epochs, batch size 100, and an RTX 2070, with roughly
one to four hours of training depending on the model (pages 2680 and 2682).
We will measure the actual cost on the cluster before scheduling the tables.
Generalized results come first; customer-specific results remain in scope and
will be costed separately.

If measured results fall short, the next entries will explain which
alternative could close the gap, what test we chose, and what happened.
Repeated runs and uncertainty estimates will follow a declared budget and
statistical procedure. An apparent plateau will need evidence across measured
training or model sizes; a flat-looking curve alone will not establish a
universal ceiling.

Future entries will report successful matches, implementation mistakes, and
changes of mind alongside negative results. We will append dated corrections
and link back to the affected entry, so readers can follow how a conclusion
changed.

## Inspect the record

- [Current all-model plan](../../docs/plans/2026-09-20-robust-all-model-reproduction.md)
- [Code-availability search](CODE_AVAILABILITY.md)
- [Earlier source specification](METHOD.md), preserved with its original audit inputs; the fresh issues above supplement it
- [Historical arithmetic finding](SOURCE_AUDIT_FINDING.md)
- [This journal's editable source](RESEARCH_LOG.md)

This journal contains source observations, an algebraic check, verified
preparation, a fitted random-forest pilot, and controlled follow-up results.
It does not yet contain a full-population reproduction of this paper.
