# First SVM pair

Recorded 2026-09-21 before fitting research data; user approved the next model.

## Question

Does the reported sigmoid-kernel SVM learn useful ranking on the original
verified pilot, and does poisoning affect its default decision, ranking, or
both? This is an exploratory numerical question with an explicit completion
of missing details (N/I), not full Table III reproduction, mechanism proof,
or an all-configuration attainability bound.

The complete paper was read during source reconstruction. We rechecked
Section III-C, p.2679, and Table III, p.2681, visually for this step.
The paper specifies C=1 and a sigmoid kernel; III-B.2(c), p.2678, specifies
supervised benign/malicious labels. Gamma, coef0, tolerance, library/version,
class weighting, and score/probability treatment are not given. Do not replace
the printed kernel with an RBF or linear kernel to improve the result.

## Initial executable completion

Use stock sklearn.svm.SVC from scikit-learn 1.9.0 in the existing forest
environment. Declare C=1.0, kernel='sigmoid', gamma='scale', coef0=0.0,
shrinking=True, probability=False, tol=0.001, cache_size=200 MB,
class_weight=None, max_iter=-1, decision_function_shape='ovr',
break_ties=False, verbose=False, random_state=20260920. Degree=3 is recorded
but ignored by the sigmoid kernel. One numerical thread; no extra calibration.
The external 15-minute job limit remains finite even though no iteration cap
is passed to libsvm. Record fit_status_ and n_iter_; an interrupted or
nonconverged fit is not a successful reproduction attempt.

The existing local/Panther environments agree on NumPy 2.5.1, SciPy 1.18.0,
joblib 1.5.3 and threadpoolctl 3.6.0; requirements-svm.txt pins them. Explicit
probability=False emits a parameter-deprecation warning in sklearn 1.9.0;
keep that expected warning without mistaking it for nonconvergence.

[Historical SVC documentation](https://scikit-learn.org/0.24/modules/generated/sklearn.svm.SVC.html)
supports gamma='scale', coef0=0, and uncalibrated decisions as ordinary defaults.
Scale uses 1/(number of features * variance of training inputs); record the
actual numeric gamma and variance. The older auto choice is 1/48 here and
remains an untested alternative. Native predict() supplies primary labels;
decision_function() supplies the finite, signed, higher-is-malicious ROC score.
The constructed binary tie check gives class 1 at margin zero. Validate that
relationship, including ties, without inventing probability values.

Probability calibration is omitted: it would add internal five-fold fitting
and can disagree with native class predictions. Gamma/coef0 alternatives,
probability calibration, and other kernels remain distinct unrun choices.
The authors' exact software stack is unidentified. A solver's success status
is not a proof of a globally optimal sigmoid-kernel solution.

## Fixed inputs and target context

Reuse the original pre-split-ADASYN pilot unchanged: 20 customers, 28 days,
48 features, 4,464 training rows and 2,232 test rows. At p30 the same six
customers contribute 675 flipped training labels (15.12097% of all rows).
No new attacks, scaling, sampling, balancing, or split. Metadata SHA-256:

- p00: 1c2e5ebfee9584c160fee209851e7f685d8c8d7971922d62a63960bc6308c3de
- p30: 9ac0faeea220e62ed5af13e39deb174a190cc69749786c38ea58240e5031f21a

Verify every consumed array hash and pair identity. Known pre-split synthetic
and source-day dependence remain; the corrected forest controls do not
silently become the SVM's input protocol.

Table III p00/p30 context: DR 89.2/73.7, FA 10.2/25.7, SP 89.8/74.3,
PR 89.0/74.0, ACC 89.5/74.0, F1 89.1/73.8, AUC 89.5/74.0 (%).
These concern a different, larger population and cannot turn a pilot match
or miss into a full-data reproduction finding.

## Saved observations and decision rules

Record all seven metrics with ordinary accuracy, confusion counts, original-only
metrics, per-attack detection, original/synthetic benign false alarms, and
constant-prior/negative-raw-daily-mean controls. Enumerate all saved-score cutoffs
in both directions at FA caps 10.2/25.7% (SVM's printed corners), plus
17.6/33.3% for comparison with the existing forest. These test-label-chosen
cutoffs are diagnostics, not independently calibrated deployment decisions.

Record training accuracy against observed and true labels, support counts by
class, support indices/dual coefficients in the model, actual gamma, numeric
training variance, intercept, solver iterations/status, margin range/ties,
timing, peak RSS, dependencies and warnings. Save/reload must preserve every
test margin and prediction exactly. Keep raw margins; do not sigmoid-transform
them into fake probabilities or lose distinctions through saturation.

If ranking remains high but default detection drops, inspect the predeclared
cutoff contrasts. If ranking is poor, first check convergence, label/score
orientation, and positive controls. Name the next source-related uncertainty;
do not silently try another seed, kernel, gamma, calibration, or preparation.
Report favorable results as plainly as failures. The limit from fixed scores
does not extend to other model parameters or all feasible implementations.

## Budget and verification

One CPU job: at most 15 minutes, four CPUs, 16 GiB, no GPU. Two experimental
fits, p00/p30, seed 20260920 (not an active randomness source with probability
disabled). Constructed fixtures are separate; real fitting and scoring remain
on a compute node. Record partial/failed attempts; do not retry from results.

Before freezing, test exact parameter/stock-library parity, a positive example,
corrupted-label use, raw-score ranking and reversal, zero-margin ties, fit-status
capture, and save/reload. Preserve forest/AdaBoost behavior and audit them at
their recorded source commits. After transfer, recompute metrics and compare
input/output hashes and paired labels/identities. Stop after updating the
journal and next justified question. No confidence interval, unseen-customer
claim, full-data runtime inference, or author-intent conclusion is earned.
