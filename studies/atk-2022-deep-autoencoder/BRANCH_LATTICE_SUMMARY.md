# Paper 1 Branch-Lattice Count and Budget

**Generated:** 2026-07-24
**Manifest:** `config/branch_lattice.toml`
**Machine summary:** `results/branch_lattice_summary.json`
**Generator:** `src/branch_lattice.py`

The committed JSON stores counts and SHA-256 inventories. Pass
`--include-branches` to the generator when an execution needs the complete
expanded branch records; the expansion is deterministic and is not committed.

## Coverage closure

- 22 dataset/model/table families.
- 22 printed-anchor configurations, which cannot be rejected by screening.
- 899 additional interpretive configurations.
- **921 total compatible paper-consistent semantic configurations.**
- 22 separately identified corrected-control configurations.
- All 36 rows in `AMBIGUITY_REGISTER.md` map to an executable dimension,
  fixed paper behavior, global frozen envelope, non-executable node, locked
  dependency, or required timing record; the mapping is machine-checked.
- Every option and every compatible option pair is present in every family to
  which it applies; this is checked by executable tests.
- Four impossible threshold pairs are machine-readable exclusions: the three
  ROC-derived formulas cannot execute without validation labels, and a printed
  supplied constant cannot also be a dataset-specific derivation.
- Coupled choices are placed in the same family and therefore covered
  together.
- The seven-class `multiclass_labels` reading applies only to ISET, where the
  six generated attack identities exist. SGCC has only supplied
  benign/malicious labels; its impossible seven-class dimension was removed
  before replacement results under decision
  `../../docs/decisions/2026-07-24-sgcc-multiclass-label-scope.md`.
- The unconstrained Cartesian product would contain **52,566,274,080**
  configurations. It is excluded because arbitrary higher-order mixtures are
  not distinct source-grounded interpretations. The exclusion is explicit,
  not hidden.

The coverage claim is therefore: all registered textual options, all allowed
option pairs, every printed anchor, and full treatment of promoted coherent
branches—not every arbitrary higher-order combination of unrelated omissions.

## Screening budget

Each semantic configuration receives three screening seeds on 10% of the
eligible data for three epochs:

| Quantity | Point estimate | Conservative 2x |
|---|---:|---:|
| Screening attempts | 2,763 | 2,763 |
| GPU-hours | 558.7 | 1,117.4 |
| CPU-hours | 57.4 | 114.8 |
| GPU-job wall-hours, serial | 149.1 | 298.1 |
| Ideal GPU wall time at three simultaneous jobs | 49.7 h | 99.4 h |

Queue delay and data-cache construction are not included. The time estimates
use observed Paper 1 runtimes when available and conservative estimates
otherwise. A bounded runtime-only calibration may update estimates but cannot
change branches, outcomes, or promotion rules.

## Promotion rule

The ROC point reported by the paper imposes the lower bound

`AUC >= (DR + 1 - FA) / 2`.

Interpretive branches promote when any of their three screening seeds reaches
that model/dataset bound minus 0.20. A branch up to another 0.05 below the
promotion floor receives a 10-epoch borderline rerun before rejection.
Printed-anchor and corrected-control branches bypass screen rejection.
Implementation/resource failures are retained as failures, not treated as
negative model evidence.

| Dataset/model | Reported-point AUC bound | Screen floor | Borderline floor |
|---|---:|---:|---:|
| SGCC FC-SAE | 0.845 | 0.645 | 0.595 |
| SGCC LSTM-SAE | 0.870 | 0.670 | 0.620 |
| SGCC FC-VAE | 0.905 | 0.705 | 0.655 |
| SGCC LSTM-VAE | 0.935 | 0.735 | 0.685 |
| SGCC LSTM-AEA | 0.960 | 0.760 | 0.710 |
| SGCC Naive Bayes | 0.795 | 0.595 | 0.545 |
| SGCC ARIMA | 0.890 | 0.690 | 0.640 |
| SGCC one-class SVM | 0.9125 | 0.7125 | 0.6625 |
| SGCC supervised feed-forward | 0.9075 | 0.7075 | 0.6575 |
| SGCC supervised LSTM | 0.9125 | 0.7125 | 0.6625 |
| SGCC multiclass SVM | 0.9225 | 0.7225 | 0.6725 |
| ISET FC-SAE | 0.830 | 0.630 | 0.580 |
| ISET LSTM-SAE | 0.860 | 0.660 | 0.610 |
| ISET FC-VAE | 0.885 | 0.685 | 0.635 |
| ISET LSTM-VAE | 0.920 | 0.720 | 0.670 |
| ISET LSTM-AEA | 0.945 | 0.745 | 0.695 |
| ISET Naive Bayes | 0.775 | 0.575 | 0.525 |
| ISET ARIMA | 0.870 | 0.670 | 0.620 |
| ISET one-class SVM | 0.905 | 0.705 | 0.655 |
| ISET supervised feed-forward | 0.895 | 0.695 | 0.645 |
| ISET supervised LSTM | 0.9025 | 0.7025 | 0.6525 |
| ISET multiclass SVM | 0.915 | 0.715 | 0.665 |

## Full-treatment bounds

If every interpretive branch promoted, the declared sequential search and
three-seed confirmation would require:

- 281,418 attempts;
- approximately 2,197,988.5 GPU-hours;
- approximately 59,558.8 CPU-hours.

This is an intentionally conservative all-promote upper bound, not an
execution proposal. It charges every promoted case for Algorithm 6's most
expensive retained reading: 70 layer/width coordinate evaluations, four
optimizers, four dropout rates, eight activation pairs, and three confirmation
seeds. Literal uniform-width search needs 36 evaluations, while Tables II-V can
also be replayed directly from the published Table I settings. The
threshold-free screening funnel is necessary to make the finite audit
executable without discarding branches capable of approaching the reported ROC
points.

The separate corrected track contains 22 machine-identified model-family
controls and 555
search/confirmation attempts under the current Table III-IV-V multiplier:
approximately 4,545 GPU-hours and 288 CPU-hours before the 2x uncertainty
factor. It is scheduled only after the paper-consistent reconstruction and is
reported separately.

## Explicit non-executable cases

1. Attack 3's printed `tf = ti - tl`.
2. The undefined “median of IQR of the ROC curve” threshold operation.
3. The literal SVM kernel/gamma reversal, which is not a valid API
   configuration.

Each has all minimal registered executable repairs in the manifest.

## Explicit exclusions

1. The 52.57-billion arbitrary full Cartesian product.
2. Test-label-selected thresholds as paper-consistent; these remain an
   externally added leakage demonstration only.
3. Dot-product attention as paper-consistent because the paper specifies a
   feed-forward alignment network.
4. Undocumented ISET subsets beyond all Code=1 meters and repeated seeded
   approximately-3,000-meter samples.
5. An unbounded Cartesian product of unspecified Keras versions/backends;
   one semantic implementation is locked, and behavior-changing backend
   sensitivities are X-track rather than paper-reproduction branches.
