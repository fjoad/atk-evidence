# Can this detector learn from corrupted labels?

Updated: 2026-09-20

This is our working journal for *Robust Electricity Theft Detection Against
Data Poisoning Attacks in Smart Grids*, by Takiddin and colleagues (2021).
We are rebuilding the experiments to find out whether the reported results
can be recovered. We record what we notice, why it matters, what we decide to
test, and how the evidence changes our view.

**Where we are, 20 September 2026:** we have read the complete paper, checked
for released code, and identified the choices needed to implement it. Model
experiments for this study have not started. The questions below are open.

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

## What we will do next

First, settle the dataset, poisoning operation, and the remaining source
choices. Then build the models with ordinary libraries and the paper's
selected settings. Small checks must show that the implementations receive
the intended data, update correctly, and calculate the metrics correctly.

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

This journal contains source observations, an algebraic check, and planned
experiments. It does not yet contain a trained reproduction result for this
paper.
