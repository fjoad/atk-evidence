# Read-only SVM replay and kernel finding

**Date:** 2026-09-21. **Scope:** the two existing SVMs and one predeclared
512-row training subset. Zero experimental fits; no parameter alternatives.

The saved SVM scores reproduce the stated model calculation to roughly
1e-12, with no prediction differences. The fixed sigmoid kernel has substantial
negative curvature even after restriction to the SVM's equality-constrained
directions. These are separate findings: the first checks the instrument;
the second identifies an optimization property, not the cause of poor accuracy.

## Frozen inputs and selection

[SVM_REPLAY.md](../../SVM_REPLAY.md) and code were frozen at
4665e078d50d630dc45290e2e3ce2b61bbe927ea. The original models are from
e698173/job 398709 and retain their original result/model/score hashes.
All original preparation arrays and the five direct scientific files are
unchanged. Our diagnostic refuses calls to SVC.fit during its empirical work.
Software fixtures train only constructed examples and are not experiment fits.

The 512 training rows were selected by sorted UID, a seeded permutation
(20260921, role 601), and selection without replacement, then UID order.
There was no score-based selection or replacement draw. The subset has 309
original and 203 synthetic rows. Observed benign/attack counts are 251/261
at p00 and 341/171 at p30. Both models share the same kernel inputs,
gamma 0.020833333524383626 and coef0=0. Selection-identity SHA-256:
94ae9a462848009c885c9a23e1f7cc2e00c5fbbaa413b30e31c82b2fbd5476e0.

## Independent score reconstruction

In float64, compute each score directly as the sum of signed support-vector
coefficients multiplied by tanh(gamma*dot(x,support)+coef0), plus intercept.
This does not call the library's decision function to construct the manual score.
Compare it with a separate native-library replay and the preserved test scores.

| Model | Training rows | Maximum training error | Test rows | Maximum test error |
|---|---:|---:|---:|---:|
| p00 | 4,464 | 1.53477e-12 | 2,232 | 1.53477e-12 |
| p30 | 4,464 | 1.67688e-12 | 2,232 | 1.33582e-12 |

All 13,392 row/model evaluations pass the frozen elementwise tolerance
1e-8+1e-10*abs(native). The largest error/tolerance ratio is 7.08663e-5.
There are zero near-zero ambiguous scores and zero prediction disagreements.
Fresh native test scores and labels match the original arrays bit for bit.

All 1,689/1,901 support vectors match their indexed training rows exactly;
coefficient signs match observed labels, maximum alpha is 1, and signed
coefficient sums are 0. Replayed observed-label training accuracy remains
63.32885/59.72222%; true-label accuracy remains63.32885/55.48835%.
Thus the earlier weak metrics are not explained by a wrong sigmoid formula,
score sign, model-to-input binding, or saved-score corruption in this audit.
This does not independently identify the authors' intended missing settings.

## Kernel spectrum and constrained directions

K is the sigmoid similarity matrix. H K H removes its constant direction,
where H=I-11'/512. The declared numerical tolerance is
1e-10*max(1,max(abs(eigenvalues))). Counts treat magnitudes within tolerance
as numerically unresolved, not exactly zero.

| Matrix | Minimum eigenvalue | Maximum eigenvalue | Negative count | Near-zero count | Positive count | Tolerance |
|---|---:|---:|---:|---:|---:|---:|
| K | -23.45098 | 148.40311 | 366 | 45 | 101 | 1.48403e-8 |
| H K H | -19.96835 | 147.88472 | 366 | 46 | 100 | 1.47885e-8 |

The negative directions are far larger than the declared roundoff threshold.
Minimum-eigenvector normalized residuals are 3.55105e-16 and 3.09733e-16.
The centered witness has sum residual 1.55431e-15. For either observed-label
map, its corresponding dual direction has equality residual 2.05391e-15
and quadratic form-19.968349293974605. This checks relevance to the SVM
dual equality, not only an arbitrary negative direction of the raw matrix.

In the usual binary SVM dual, Q=diag(y)Kdiag(y) and y'd=0. Taking
d=diag(y)v for this zero-sum witness gives d'Qd=v'Kv<0. Hence the dual's
quadratic term is not concave along that equality-constrained direction.
Both classes are represented and C>0, so feasible interior points exist;
this is a property of the constrained problem, not a certificate of a
feasible improvement from the particular fitted point on its box boundary.

The raw kernel is exactly symmetric in the saved calculation; diagonals range
from 0.02459 to 1. Off-diagonal abs(K)>=.99 occurs in about 0.35317% of entries.
Thresholded absolute-spectrum ratios are 6.14203e9 and 6.73006e9. These
ratios exclude eigenvalues inside the numerical tolerance and are not
unconditional condition numbers or a demonstrated explanation of accuracy.

A negative direction in this principal submatrix embeds in the full fixed
training matrix by setting other coordinates to zero. This is strong numerical
evidence for the stated finite matrix, not an interval-certified theorem
about every input, parameter, or version. No population inference comes from
the 512-row subset. The general possibility is described by the
[LIBSVM authors](https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html); the measured
witness here is our own diagnostic.

## What changed, and what remains open

The score-replay explanation is closed to the declared tolerance for these
artifacts. The usual convex/concave optimization guarantee does not apply to
this declared sigmoid kernel on these inputs, despite the solver's earlier
success status. However, we did NOT test KKT conditions, a feasible descent
direction at the fitted coefficients, the global optimum, or how much of the
performance gap was caused by this geometry. No better detector was fitted.

The earlier fixed-score cutoff exclusion remains unchanged. It does not become
an all-sigmoid-SVM impossibility result, and neither the good forest/AdaBoost
results nor this weak SVM result establish author intent. Missing gamma/coef0
choices and other software completions remain unresolved.

Stop this diagnostic rather than turn it into an undeclared tuning search.
For the next planned coverage step, specify the paper's feed-forward baseline
and a bounded pilot, using the already identified standard cross-entropy
repair explicitly. Keep a separately justified finite SVM parameter sensitivity
open for later; no such fit or neural run is launched in this record.

## Execution and verification

Job 398978 completed 0:0 in 32 seconds; the diagnostic program took 5.61120
seconds. The request was 1 CPU/8 GiB/10 minutes/no GPU. Slurm allocated 2
logical CPUs on a node with two threads per core while CPUs/Task remained 1;
the program fixed numerical-library threads to 1. Report the actual allocation,
not one allocated logical CPU. Process peak RSS was 215,816 KiB; Slurm
sampled 201,844 KiB.

Before freeze, 325 main-suite tests passed and four AdaBoost fit tests were
skipped there; all seven AdaBoost tests passed in the isolated environment.
All 11 diagnostic tests passed locally and on the compute node, including a
complete constructed-artifact run. The three saved diagnostic archives pass
hash checks, selection reconstruction, all replay-difference checks, and
independent minimum-witness/residual verification. The local artifact-audit
output matches the cluster's byte for byte. Local verification did not
recompute kernels or fit models.

The original consumed inputs, metadata, model files, score files, and result
records passed before/after hash checks. Their old pair audit still matches
after transfer. No empirical fit, failed diagnostic attempt, alternate subset,
parameter change, or outcome-dependent retry occurred.

## Records

[Complete result](result.json), [artifact audit](artifact_audit.json),
[execution record](execution.json), and [original job output](slurm-398978.out).
The matrix, eigenvalues, witnesses, identities, and replay arrays remain
outside Git in data/derived/takiddin-2021-robust-poisoning/svm-replay-20260921-attempt1.

Read-only verification uses checks/svm_replay.py --audit-only --output with
that directory, in the pinned SVM environment. Its stdout must reproduce
artifact_audit.json. The prior SVM pair still verifies with verify_baseline_pair.py.
