# Read-only SVM score replay and kernel diagnostic

Recorded 2026-09-21 before inspecting the empirical kernel. User approved
this follow-up after the first SVM pair. Zero experimental fits are permitted.

## Questions and competing explanations

1. Do the saved margins really equal the stated sigmoid-kernel calculation
   using the fitted support vectors, coefficients and intercept? A disagreement
   outside declared numerical error would require an implementation/instrument
   diagnosis before interpreting the poor performance.
2. Does the fixed training kernel have substantial negative curvature,
   including after restricting to zero-sum directions? An affirmative result
   would remove the usual concave-dual guarantee for this kernel on these
   inputs. It would not establish that the saved solution is suboptimal,
   quantify the performance caused by the kernel, or rule out other settings.

These are numerical/instrument and optimization-geometry diagnostics of an
existing interpreted pilot, not a new reproduction, statistical test, or
hyperparameter search. The source model remains Section III-C p.2679:
C=1 and a sigmoid kernel; omissions are frozen in SVM_PILOT.md.

## Immutable inputs

Use the original p00/p30 preparations and fitted models from job 398709,
scientific commit e69817379a28752c909ac4fc45b15e380dbed1c4. Result hashes:

- p00: 920beced622b9ee93a51cef34857dbd2f266df5e3b41b18c5abe2c84988d4746
- p30: 3ea84776a1c63f31b3ac95d075cb3b4af220d501bb0218fdc5db5e65bd1b8ac2

Verify these records, their model/score/source hashes, the frozen preparation
metadata and consumed arrays before loading our own model pickles. Use the
unchanged requirements-svm.txt environment. Do not regenerate data, refit,
resave models, change parameters, or calculate a new detector's performance.
Recheck input/model hashes after the diagnostic to establish nonmutation.

## Independent replay

For each original model, in float64 and chunks of 128 observations, calculate

    margin(x) = sum_j dual_coef[j] * tanh(gamma * dot(x, support_j) + coef0) + intercept

on all 4,464 training and 2,232 test rows. Use public binary dual_coef_ and
intercept_, with classes [0,1]. Independently check support vectors against
their saved training indices, coefficient signs against observed labels,
alpha bounds 0<=abs(coef)<=C+1e-10, and the equality abs(sum(coef))<=1e-8.
Use the fitted numeric gamma, not a newly estimated value.

Fresh native test scores/labels must exactly match their saved counterparts.
For train and test, manual/native margin differences must satisfy
abs(delta)<=1e-8+1e-10*abs(native). Report maximum absolute error and maximum
error/tolerance ratio. Report manual/native prediction disagreements and the
number of native margins within that numerical tolerance of zero. Disagreements
outside that near-zero set fail the check. Native labels use margin>=0.
Report training accuracy again only as a consistency check, not a new fit.
On failed replay, preserve results and stop before assigning a kernel cause.

The [LIBSVM kernel definition](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
specifies tanh(gamma*inner_product+coef0). The library's
[FAQ](https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html) motivates checking
sigmoid matrices for non-positive-definiteness, not assuming it explains this
particular outcome.

## Outcome-independent kernel subset

From the 4,464 training rows, sort indices by unique training UID. Apply
NumPy default_rng(SeedSequence([20260921,601])) to permute that sorted index
list; take the first 512 without replacement, then order the selected rows by
UID. Include original and synthetic rows as selected; no label balancing,
score-based selection or resampling. The same feature rows serve both models,
whose fitted gamma and coefficient must match exactly. Require both observed
classes at each poisoning level; fail rather than draw another subset.
Record selected original/synthetic and class counts and the identity hash.

Compute only the declared sigmoid Gram matrix K for those rows. Store K and
the selected identities/labels locally outside Git. Record asymmetry, diagonal
range and off-diagonal fraction abs(K)>=.99. Require finite entries and
max(abs(K-K.T))<=1e-12; use (K+K.T)/2 for spectral work.

Inspect eigenvalues of K and H K H, H=I-11'/512, using symmetric float64
eigendecomposition. Centering removes the constant direction. For each matrix
let tau=1e-10*max(1,max(abs(eigenvalues))). Report min/max eigenvalue, counts
below -tau/within tau/above tau, negative spectral mass, and the ratio of
largest to smallest abs(eigenvalue)>tau (a thresholded spectrum ratio, NOT
an unconditional condition number). Save all eigenvalues and minimum-eigenvalue
witnesses; verify normalized residual<=1e-10 and Rayleigh quotient agreement.

For a negative centered witness v, require abs(sum(v))<=1e-8. The same
zero-sum direction has v'Kv<0. With binary signs y, d=diag(y)v satisfies
y'd=0 and d'diag(y)Kdiag(y)d=v'Kv. Verify both poisoning-label maps. This
connects the observation to the dual's equality-constrained curvature rather
than relying solely on an arbitrary negative direction of K. It is not a
claim that d is a feasible descent direction at the fitted box boundary, a
KKT test, or a proof about the attained local/global optimum.

## Interpretation, budget, and stopping

Float64 witnesses well beyond the declared tolerance are numerical evidence,
not interval-certified proofs. A negative direction in this fixed principal
submatrix also embeds in the full training matrix; absence on 512 rows would
not establish that the full matrix is positive semidefinite. No population
sampling inference or confidence interval is claimed from this subset.

If replay passes and curvature is negative, report both and retain the opening
that omitted parameters may matter. If spectra do not show a negative direction,
report that too; do not select more subsets until a negative result appears.
No alternate gamma, coefficient, kernel, calibration, seed, or model fitting.

One Slurm job: 10 minutes, 1 CPU, 8 GiB, no GPU. Real scoring/kernel computation
stays on the compute node. Local runs use constructed software fixtures or
verify already-saved diagnostic arrays. Tests include hand-calculated margins,
stock-library parity, known PSD/indefinite matrices, zero-sum curvature,
threshold tolerances, stable subset selection, and guard/refit refusal.
Save attempts, arrays, source hashes and failures. Stop after artifact checks
and a journal entry naming the next justified question; do not publish.
