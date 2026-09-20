# First random-forest check

**Recorded:** 2026-09-20, before any detector is fitted for this study.

## Question and scope

Can the paper's 100-tree random forest learn and produce consistent,
interpretable predictions through the verified preparation pipeline?
Does the declared 30% customer-poisoning operation change the resulting
scores or mostly the decision threshold?

This first pair uses only the existing 20-customer, 28-day pilot. It is an
exploratory instrument and behavior check, not full Table III reproduction
or a statistical rejection of the paper. The reported full-data rows provide
context, not a pilot pass/fail criterion.

## Frozen inputs

Prepared by commit 30ce6c4 in job 397206. Reuse the generalized two-class
outputs without regenerating, resampling, or changing any labels:

- p00 metadata SHA-256:
  1c2e5ebfee9584c160fee209851e7f685d8c8d7971922d62a63960bc6308c3de
- p30 metadata SHA-256:
  9ac0faeea220e62ed5af13e39deb174a190cc69749786c38ea58240e5031f21a

Each has 4,464 training and 2,232 test rows. Test values and labels are paired.
At nominal 30%, 675 original malicious training labels (15.12% of all training
rows) were flipped. The ADASYN-before-split parent overlap is preserved and
limits any generalization claim.

## Model and predictions

Section III-C, p.2679, gives 100 random-forest estimators. Other settings are
declared completions, using scikit-learn 1.9.0:

- 100 trees; Gini splitting; unlimited depth; two examples to split;
  one example per leaf; sqrt(number of features) considered per split;
- bootstrap sampling; no class weights, pruning, feature monotonicity
  constraints, out-of-bag estimation, or warm starts;
- seed 20260920, unchanged across the paired fits;
- four workers while fitting; one while scoring and serializing, avoiding
  parallel floating-point accumulation order in reload checks;
- train only against observed labels; retain true labels for diagnostics;
- use the library's predicted class: argmax of class probabilities, with
  exact ties assigned to class 0; use class-1 probabilities for AUC.

These settings match the locally inspected default estimator apart from the
explicit seed and worker counts. Historical max_features='auto' for
classification also meant sqrt. The paper does not identify its library
version; do not claim to have recovered the original environment.

## Measurements and cheap diagnostics

Compute all seven metrics from one common evaluation population. ACC is
ordinary accuracy, as this paper defines it; balanced accuracy is additional.
Undefined precision/F1/AUC fields must be explicit rather than invented.

Preserve probabilities, predictions, labels, row identities, synthetic status,
attack type, fitted model, actual parameters, tree counts/depths, input/output
hashes, software versions, fit/score/reload times, and failures.

Before interpreting a gap, check:

1. native predictions agree with argmax of saved probabilities;
2. saved/reloaded model probabilities and labels agree;
3. training accuracy against observed and true labels, and training score
   variability, establish whether the fit changed its behavior;
4. a constant training-class-prior control and the no-training negative
   daily-mean-consumption score provide simple comparisons;
5. per-attack detection and original/synthetic benign false alarms expose
   aggregate effects;
6. the complete ROC threshold list, including ties and no alarms, measures
   best detection at 17.6% and 33.3% false alarms and best balanced accuracy;
   diagnostic score reversal is retained too.

The paper's random-forest DR/FA/AUC targets are 82.2/17.6/81.4 at p00 and
66.8/33.3/66.7 at p30 (Table III, p.2681). Threshold sweeps use test labels
only as fixed-score diagnostics and never select a deployable cutoff or model.
Report primary all-row and original-row-only results separately. No confidence
interval from this one seed or from treating dependent rows as independent.

## Competing outcomes and next decision

- If the instrument or reload checks fail, preserve the attempt and repair
  the identified defect before more experimental fits.
- If the forest learns and performs well, report that opening plainly.
  A good pilot supports pipeline viability; its small population and synthetic
  overlap do not establish full-data reproduction.
- If scores rank poorly, inspect labels, per-attack behavior, and the simple
  controls before changing settings or repeating seeds.
- If ranking remains useful but the default labels deteriorate under poison,
  threshold/calibration is a live explanation for that pilot, not evidence
  that discrimination has disappeared.

Stop after this pair and its artifact audit to choose the next informative
step. No automatic seed repetition, full-data promotion, or model-family sweep.

## Compute

One CPU Slurm job, at most 15 minutes, four CPUs, 16 GiB RAM, no GPU.
Exactly two experimental forest fits, one per frozen input. Software fixtures
may fit only constructed test arrays. All real-data fitting/scoring and
fresh-model reload checks run on the compute node. Preserve partial results
if time or memory is exhausted. The paper's roughly one-hour full-data time
cannot be tested by this small pilot.

References: [RandomForestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
and [ROC threshold conventions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html).
The executable installed version and complete actual parameters are recorded.
