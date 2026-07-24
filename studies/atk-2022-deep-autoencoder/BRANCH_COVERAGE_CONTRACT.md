# Paper 1 Branch-Coverage Contract

**Created:** 2026-07-23  
**Status:** Machine-readable coverage v1 generated; structural implementation
pending
**Authority:** The exact Paper 1 PDF only  
**Historical implementation:** `implementation-v1`, quarantined by
`PAPER_TO_CODE_TRACEABILITY.md`

## Tracks

| Track | Question | May support a paper-reproduction claim? |
|---|---|---|
| `P` — printed | What happens when executable statements are applied exactly where printed? | Yes |
| `I` — interpretations | What happens under every materially defensible reading of ambiguous or contradictory text? | Yes, branch-specifically |
| `C` — corrected | What happens under leakage-free methodology we scientifically recommend? | No; causal/control evidence only |
| `X` — externally motivated | Could an unstated precursor, literature convention, or tool default explain the numbers? | No; author-implementation possibility only |

## Coverage rule

Every material row below must end in one of:

- an executable frozen option set;
- a `NON-EXECUTABLE` printed node plus all minimal repairs;
- a reasoned exclusion.

No outcome may be inspected before the option set, dependencies, seeds,
budgets, and promotion rule for that row are frozen.

## Data and split dimensions

| ID | Material text issue | `P` / `I` options to execute | `C` control |
|---|---|---|---|
| D01 | SGCC supplies 1,034 daily values per customer while the model prose specifies 48 inputs | Full 1,034 sequence; contiguous 48-day nonoverlapping windows; rolling 48-day windows; first 48; last 48; deterministic 1,034-to-48 binned means | Time-based customer-disjoint windows with train-only preprocessing |
| D02 | SGCC missing-value treatment omitted | Drop incomplete customer; zero fill; within-row interpolation plus edge median; per-customer mean fill | Train-derived imputation with missingness indicators and sensitivity analysis |
| D03 | ISET contains incomplete and 46/50-slot DST days but a sample has 48 values | Complete slots 1-48 only; deterministic trim of extra slots; duplicate-slot aggregation; within-day interpolation when a 48-slot grid can be recovered | Calendar-aware resampling fitted without test leakage |
| D04 | “Around 3,000” ISET homes versus 4,225 allocation-code-1 meters | All 4,225; deterministic seeded 3,000-meter subsets repeated across seeds | All eligible residential meters |
| D05 | Apply attacks to “all customers,” but test customers must be unseen | `M_all`: attacks from every customer; `M_B2`: attacks only from held-out customers | Split customers first; create train-only supervised attacks and held-out test attacks separately |
| D06 | Rows are called profiles and customers | Customer-disjoint 2:1; row/profile-random 2:1 | Customer-disjoint train/validation/test |
| D07 | Normalize “both classes” to zero mean and unit variance | Joint B+M feature-wise; each class separately feature-wise; per-profile; per-customer across time | Fit feature scaling on training benign data only and apply unchanged |
| D08 | Neural validation is mentioned but its construction and final refit are absent | No validation/fixed epochs; B1 holdout with no refit; cross-validation on B1 with final refit on all B1; holdout then final refit | Customer-disjoint validation, frozen selection, final refit only if predeclared |
| D09 | Anomaly ADASYN is explicitly placed in the test set | Apply ADASYN to B2+M exactly as printed; no-ADASYN diagnostic | No test resampling; report natural test set and an optional balanced subset made only by selecting real rows |
| D10 | Supervised ADASYN is explicitly before the split, while general prose says customer-disjoint | ADASYN before row split; customer split then ADASYN each stated training population; both crossed with D05 | Customer split first; ADASYN training only; untouched validation/test |
| D11 | ADASYN parameters, implementation, and seed omitted | Frozen implementation-version defaults; `n_neighbors` finite envelope `{3,5,10}`; all declared data seeds | Training-only ADASYN tuned inside validation, plus no-ADASYN/class-weight baselines |

## Attack dimensions

| ID | Issue | `P` / `I` options to execute | `C` control |
|---|---|---|---|
| A01 | Attack-1 fixed factor scope | One factor per profile; one factor per customer matrix; one factor for the complete generated dataset | Per-profile seeded factor with severity strata |
| A02 | Attack-2 dynamic factor granularity | Per half-hour reading; per hour applied to both half-hours | Per-reading seeded factor with severity strata |
| A03 | Attack-3 prints `tf=ti-tl` and incompatible start/length ranges | Preserve subtraction as `NON-EXECUTABLE`; addition with valid-fit start; printed start 0-19 with end truncation; printed start with circular wrap | Valid bounded intervals with explicit severity bins |
| A04 | Hourly equations versus 48 half-hours | Map one hour to two slots; apply indices directly to 48 slots | Two slots per hour |
| A05 | Attack randomness/repetition omitted | Fixed generated attack set per data seed; regenerate per model seed; regenerate per experiment | Freeze attacks independently of model seeds and repeat across attack-data seeds |

Attacks 4, 5, and 6 otherwise have one directly executable interpretation:
daily mean, dynamic factor times daily mean, and reversed day.

## Architecture and training dimensions

| ID | Issue | `P` / `I` options to execute | `C` control |
|---|---|---|---|
| M01 | Table I/Section IV-C specify all encoder hidden widths and the full opposite-order decoder. Figs./prose depict a latent layer, while Algorithms 2/5 directly reuse the terminal encoder state or attention context | Always instantiate the full printed hidden-layer mirror; for SAE/AEA run both a distinct latent projection and the algorithm-literal existing bottleneck representation; VAE uses its explicit latent distribution | Capacity-matched bottleneck autoencoder with latent width chosen on validation |
| M02 | Width of a distinct latent projection/distribution is absent | Widths `{2,8,16,32,48,100}` wherever the distinct-latent branch applies | Validation-selected compact latent width |
| M03 | LSTM input described as a 48-neuron time series | 48 time steps × 1 feature; 1 time step × 48 features | 48 time steps × 1 feature |
| M04 | Decoder input/state schedule omitted beyond initialization | Repeat latent at every step; latent at first step then zeros; autoregressive previous reconstruction; mirrored-state versus top-state-only initialization | Standard seq2seq decoder with explicitly validated schedule |
| M05 | Attention alignment topology/tensor notation incomplete | Feed-forward alignment with the explicitly printed previous-decoder-state query and concatenation; literal sum branch where dimensions permit | Standard additive attention with masks and auditable weights |
| M06 | VAE Eq. (10) uses squared L2 while common implementations use mean MSE | Sum-squared reconstruction + KL; mean-MSE + KL; both with frozen KL reductions | Validated ELBO with explicit likelihood scale and KL schedule |
| M07 | VAE reconstruction probability lacks decoder variance and sample count | Learned diagonal decoder mean/variance with low likelihood anomalous and Monte Carlo counts `{1,10,100}`; fixed-variance sensitivity; MSE-high and MSE+KL-high branches required by conflicting later prose | Calibrated negative log likelihood plus deterministic reconstruction baselines |
| M08 | VAE score orientation conflicts | Low probability anomalous; high cost anomalous | High negative-log-likelihood anomalous |
| M09 | Epochs, batch, learning rate, and convergence omitted; Algorithm 6's scalar `N_l` loop cannot directly yield Table I's unequal layer widths | Batch `{32,512}`; epochs `{10,30,100}`; fixed-epoch and early-stop/refit; optimizer defaults plus predeclared log-scale learning-rate neighborhood; literal uniform-width 36-evaluation search, per-layer coordinate 86-evaluation search, and direct Table-I replay | Nested validation with early stopping and repeated seeds |
| M10 | Dropout placement omitted | Dense dropout after all hidden layers, encoder hidden layers only, or bottleneck only; LSTM input dropout, recurrent dropout, or an equal input/recurrent split using the same printed total rate | Validation-selected regularization |
| M11 | Backend/version/initialization omitted | One locked Keras-compatible semantic implementation with repeated seeds; unbounded version/backend combinations are explicitly excluded from the paper-consistent lattice | One modern locked reference implementation plus X-track backend sensitivity |

## Evaluation and benchmark dimensions

| ID | Issue | `P` / `I` options to execute | `C` control |
|---|---|---|---|
| E01 | Printed thresholds are known but the “median of IQR of ROC” rule is not executable as written | Printed constants; median threshold within central ROC quartiles; midpoint/median of threshold IQR; per-dataset versus ISET-transferred threshold | Threshold selected on validation by a frozen objective |
| E02 | Validation labels needed for ROC/DR conflict with benign-only XTR | Malicious validation generated from B1; heldout subset of B2 reserved for validation; printed thresholds without derivation | Customer-disjoint validation attacks, untouched test |
| E03 | Table V FA varies despite “same” model/threshold and apparently common benign data | One model/common benign set; independent retraining per attack; independent benign split per attack; retrain-and-resplit; full set and seeded 3,000-per-class size | Repeated per-attack evaluation with a frozen model and common untouched benign test; explain invariant FA |
| E04 | “Multiclass SVM” class cardinality omitted | Binary benign/attack; seven classes (benign + six attacks), collapsed to binary for DR/FA | Multiclass and binary baselines reported separately |
| E05 | ARIMA autoregressive order and anomaly score omitted | `p` envelope `{0,1,2,5}` with `d=1,q=0`; per-profile versus pooled fitting; residual MSE versus likelihood | Validation-selected time-series baseline |
| E06 | SVM sample caps are absent from the paper | Full-data fit; any capped run labeled resource diagnostic only | Scalable full-data baseline |
| E07 | Supervised output width, loss, labels, and decision omitted | 2-way Softmax/categorical; 1-way Sigmoid/binary where shape permits; attack-type multiclass variants | Calibrated binary and multiclass classifiers |
| E08 | Table IV timing lacks hardware, epochs, and timing boundary | Every training/stopping branch records fit-only and end-to-end wall time; no equality claim without matching provenance | Hardware-normalized throughput and full provenance |

## Corrected-control minimum

The corrected track must include, at minimum:

1. customer-disjoint train/validation/test partitions;
2. transformations fitted on training data only;
3. no synthetic or label-informed modification of validation/test;
4. anomaly models trained on benign training data only;
5. supervised ADASYN, if used, applied only to training;
6. thresholds and hyperparameters selected only on validation;
7. untouched-test metrics with seed distributions and confidence intervals;
8. simple baselines capable of exposing attack triviality and leakage;
9. one scientifically appropriate temporal model and one non-temporal model;
10. explicit comparison against every reported table cell without treating a
    corrected match as reproduction.

## Dependency and execution policy

- The option list is a lattice, not an unconstrained Cartesian product.
  Impossible combinations are excluded with machine-readable reasons.
- Every single interpretation option is executed. Material interactions are
  crossed when one choice changes the inputs seen by another.
- A predeclared AUC screening funnel may avoid full training of branches that
  cannot reach the reported ROC point even under an oracle threshold.
- Screening never deletes a result. A branch promotes according to a frozen
  generous bound, not because its preliminary result is convenient.
- Hyperparameter search is nested inside each semantic branch; semantic
  branches are never selected by best test performance.
- Final reports include a coverage appendix listing every option, combination,
  exclusion, failure, and surviving branch.

## Remaining freeze work

- [x] Convert this table into machine-readable branch IDs and dependency rules.
- [x] Calculate the branch count and compute budget.
- [x] Freeze AUC promotion bounds from each reported DR/FA pair.
- [x] Map all 36 registered ambiguity rows to executable dimensions, fixed
      behaviors, global envelopes, non-executable records, or timing records.
- Freeze seeds, confidence intervals, multiplicity treatment, and tolerances.
- Add structural tests derived directly from each paper locator.
- Review the generated branch count and exclusions before execution.

The current generated closure contains 921 compatible paper-consistent
semantic configurations across 22 families, 22 separate corrected controls,
and 2,763 three-seed screen attempts. All 36 ambiguity-register rows have
machine-validated coverage references. Threshold formula and derivation scope
are independent dimensions; incompatible formula/label/scope pairs are
explicitly excluded rather than counted as executable branches. See
`BRANCH_LATTICE_SUMMARY.md` and
`results/branch_lattice_summary.json`.
