# First AdaBoost pair

Recorded 2026-09-20 before fitting research data. User approved this next step.

## Question and boundary

Does the paper's next shallow baseline learn on the original verified pilot,
and does 30%-customer poisoning damage ranking, change its default decision,
or both? This is an exploratory N question using an I completion of missing
model details, not a full Table III reproduction, mechanism test, or universal
attainability bound. AUC near the simple controls would motivate an instrument
and model-capacity diagnosis; high AUC with lower default detection would
motivate a cutoff explanation. Report either result. No automatic retry follows.

## Source and executable completion

The complete ten-page paper was read in the preceding source phase. Sections
III-B.2(b), p.2678, and III-C, p.2679, were re-read and visually checked now:
AdaBoost uses decision-tree weak learners and selects 50 estimators. Table III,
p.2681, supplies the generalized target and was visually checked as well.
The paper does not specify the tree
depth, algorithm variant, learning rate, package/version, or seed. Its generic
Keras statement does not define a standard AdaBoost implementation.

Use stock scikit-learn AdaBoostClassifier with estimator=None (the library's
depth-one DecisionTreeClassifier), n_estimators=50, learning_rate=1.0,
algorithm=SAMME.R, random_state=20260920. Initial sample weights are uniform;
fit only the observed, possibly corrupted labels. Permit and record the
library's legitimate early stopping; 50 is a maximum, not a reason to invent
extra trees after a perfect fit. Native predict() defines the primary labels;
predict_proba()[:,1] defines the higher-is-malicious score.

[scikit-learn 0.24 documentation](https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
records SAMME.R, learning rate 1 and a depth-one default tree.
[Version 1.5 documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
still provides this algorithm and records its later removal. Use an isolated
1.5.2 environment, not a handwritten substitute or the existing 1.9.0 forest
environment. Pin NumPy 1.26.4, SciPy 1.13.1, joblib 1.4.2, threadpoolctl 3.5.0.
This preserves a historical algorithm choice, NOT the authors' unknown exact
software stack or bitwise equivalence to 2020 libraries. SAMME and other tree
depths remain untested reasonable alternatives; do not run them in this pair.

## Inputs and evaluation

Reuse the original p00/p30 preparations unchanged: 4,464 training rows,
2,232 test rows, 48 features, twenty customers, first 28 complete days,
ADASYN before splitting. Metadata SHA-256:

- p00: 1c2e5ebfee9584c160fee209851e7f685d8c8d7971922d62a63960bc6308c3de
- p30: 9ac0faeea220e62ed5af13e39deb174a190cc69749786c38ea58240e5031f21a

The same six customers have 675 malicious training labels flipped at p30
(15.12097% of all training rows). No new attacks, scaling, resampling, or
customer selection. Verify consumed array hashes and the unchanged paired
features/test identities. Preserve the known source-day/synthetic dependence;
do not confuse this with the separately completed corrected forest controls.

Record all seven printed metrics, ordinary accuracy, confusion counts, AUC,
per-attack detection, original/synthetic benign false alarms, original-only
metrics, constant-prior and negative-raw-daily-mean controls. Predeclare
fixed-score ROC caps 14.1%/29.9% (AdaBoost's printed p00/p30 FA), plus
17.6%/33.3% for common comparison with the forest. Consider reversal as a
diagnostic. Test-selected cutoffs are not validated deployment calibration.
Preserve every boosting error/weight, fitted tree count/depth, training accuracy
against observed and true labels, fit/score/reload time, warnings, versions,
and hashes. Save/reload must exactly preserve probabilities and predictions.

Printed p00/p30 context (not comparable-population reproduction targets yet):
DR 85.7/70.1, FA 14.1/29.9, SP 85.9/70.1, PR 85.3/70.0,
ACC 85.8/70.1, F1 85.5/70.0, AUC 85.0/70.0.

## Verification, budget, and stopping

Use constructed fixtures locally; real fitting/scoring stays on a Slurm compute
node. Verify stock-estimator parity, a perfect-fit/early-stop example, genuine
multi-round fitting, corrupted-label use, source revision checking, correct
AdaBoost target selection, metrics/cutoffs, and exact persistence/reload.

One CPU job: maximum 15 minutes, four CPUs, 16 GiB, no GPU. Two experimental
fits total, seed 20260920. Numerical libraries use one thread; no extra seeds,
grid search, resampling controls, full-data runs, or next model in this job.
Dependency installation is environment setup, not a research-data experiment.
Freeze code before submission. Preserve failures and diagnose software errors
without turning them into negative paper evidence or silently retrying a fit.

Afterward audit saved inputs/outputs and independently recompute all metrics
in the same pinned environment; compare the result with the original forest
only on identical test rows and disclose the different library versions.
Stop after recording the outcome and next justified question. Twenty dependent
customers and one seed do not establish a population CI, stability, general
failure of AdaBoost, or author intent. Website edits remain unpublished drafts.
