# Ambiguity Register

No single “primary branch” now stands in for Paper 1. Every option below is
required unless the final freeze document gives a source-grounded exclusion.
Corrected controls remain a separate result family. The dependency-aware
canonical enumeration is `BRANCH_COVERAGE_CONTRACT.md`.

| ID | Paper issue | Required paper-consistent options | Separate corrected/control options |
|---|---|---|---|
| A01 | SGCC has 1,034 daily values; model prose says 48 | Full 1,034 sequence; nonoverlapping and rolling 48-day windows; first/last 48; deterministic binning to 48 | Time-based customer-disjoint windows |
| A02 | SGCC missing-value handling absent | Drop incomplete rows; zero; interpolation plus edge median; per-customer mean | Train-derived imputation plus missingness sensitivity |
| A03 | ISET residential IDs and “around 3,000” selection absent | All allocation `Code=1`; repeated deterministic 3,000-meter subsets | All eligible residential meters |
| A04 | ISET has DST slots 49/50 and incomplete days | Complete 1-48 only; deterministic trim; duplicate-slot aggregation; recoverable interpolation | Calendar-aware resampling |
| A05 | Rows called both profiles and customers | Customer-disjoint 2:1; row/profile-random 2:1 | Customer-disjoint train/validation/test |
| A06 | `M` described as all customers, yet test customers are unseen | Attacks from held-out B2; attacks from all customers | Split customers first and generate attacks independently inside train/test |
| A07 | Normalization scope/order unclear | Joint B+M feature-wise; per-class feature-wise; per-profile; per-customer | Training-benign-only scaling |
| A08 | ADASYN defaults, implementation, neighbors, and seed absent | Locked-version defaults; neighbors `{3,5,10}`; all declared data seeds | Training-only tuning plus no-ADASYN/class-weight controls |
| A09 | Validation with benign-only `X_TR` cannot yield ROC/DR | Printed thresholds; malicious validation derived from B1; B2 validation carve-out | Customer-disjoint validation attacks |
| A10 | “Median of IQR of ROC” undefined | Printed constants; median central ROC thresholds; threshold-IQR midpoint/median; per-dataset and ISET-transferred variants | Frozen validation objective |
| A11 | VAE reconstruction probability undefined/direction conflicts | Learned/fixed decoder variance; Monte Carlo `{1,10,100}`; low probability; high MSE; high MSE+KL | Calibrated negative log likelihood |
| A12 | Latent width is absent | Widths `{2,8,16,32,48,100}` whenever a distinct latent projection/distribution is instantiated | Validation-selected compact width |
| A13 | Figs./prose depict a latent layer, while Algorithms 2/5 directly use the terminal encoder state or attention context | Always use every printed encoder hidden width and its full mirror; for SAE/AEA run both a distinct latent projection and the algorithm-literal existing state/context bottleneck; VAE retains its explicit latent distribution | Capacity-matched architecture |
| A14 | Decoder scheduling/state transfer absent | Repeat latent; first-step latent then zeros; autoregressive reconstruction; mirrored versus top-state initialization | Validated standard seq2seq schedule |
| A15 | Attention scorer topology absent; concatenate/sum conflict | Feed-forward alignment with the printed previous-decoder-state query and concatenate; literal sum where dimensions permit | Standard masked additive attention |
| A16 | Attack 1 factor scope absent | One draw per profile; per customer matrix; per generated dataset | Per-profile severity-stratified draws |
| A17 | Attack 3 prints `tf=ti-tl`, invalid | Preserve as non-executable; valid addition; printed start with truncation; circular wrap | Valid bounded severity strata |
| A18 | Epochs/batch/learning rate/convergence absent | Batch `{32,512}`; epochs `{10,30,100}`; fixed and early-stop/refit; frozen optimizer-default neighborhood | Nested validation |
| A19 | Dropout placement absent | Dense dropout on all hidden layers, encoder hidden layers only, or bottleneck only; LSTM input, recurrent, or equal input/recurrent split | Validation-selected regularization |
| A20 | Random seeds/runs absent | Three fixed seeds; preserve all | No replacement seeds |
| A21 | ARIMA omits `p`, fit unit, and score | `p={0,1,2,5}`, `d=1,q=0`; pooled/per-profile; residual MSE/likelihood | Validation-selected time-series baseline |
| A22 | SVM kernel/gamma phrase reversed | kernel sigmoid, gamma scale | Literal reversal is invalid API |
| A23 | Table V FA varies though benign set is fixed | Fixed common model; independent retraining; independent benign split; retrain-and-resplit | Common untouched benign set exposing invariant FA |
| A24 | Table IV hardware/training boundary absent | Our fully specified wall-clock fit time | No direct absolute-time equivalence claim |
| A25 | Only ISET threshold derivation is described | Same printed thresholds on both datasets; dataset-specific derived thresholds | Validation-only thresholds |
| A26 | Supervised output cardinality, loss, label encoding, and decision rule absent | 2-way Softmax/categorical; 1-way Sigmoid/binary where valid; attack-type multiclass | Calibrated binary and multiclass controls |
| A27 | Paper's “around 3,000” ISET units conflicts with 4,225 Code=1 meters | Covered by A03; both full and repeated 3,000 subsets required | All eligible meters |
| A28 | “Multiclass SVM” class cardinality unstated | Binary benign/malicious; seven-class attack type collapsed for DR/FA | Report binary and multiclass separately |
| A29 | Table V evaluation sample count per class unstated | Full heldout set; repeated seeded 3,000-per-class subsets | Full heldout set |
| A30 | Attack 3 start distribution unexecutable as printed | Covered by A17; valid-fit, truncation, and wrap repairs all required | Valid-fit intervals |
| A31 | VAE squared-L2 versus mean-MSE reduction changes KL weight | Sum squared + KL; mean MSE + KL | Explicit likelihood scale |
| A32 | Supervised paragraph requires M for all customers but implementation-v1 uses B2 attacks only | All-customer M; B2-only M as contradiction branch | Split-local train/test attacks |
| A33 | Neural validation carve-out and final refit are unspecified | No validation; no-refit holdout; cross-validation/refit; holdout/refit | Frozen validation and refit policy |
| A34 | LSTM “48 neurons” versus temporal input layout | 48 time steps × 1 feature; 1 time step × 48 features | 48 time steps × 1 feature |
| A35 | Table V “multiple experiments” identity is unspecified | Common model; independent models; independent splits; both | Repeated fixed-model common-test evaluation |
| A36 | Algorithm 6 loops over one `N_l` value but Table I reports unequal widths within a model | Literal uniform-width staged search (36 evaluations); per-layer coordinate staged search (86); direct replay of the published Table I optimum for Tables II-V | Nested validation with a frozen capacity search |

Entries A27--A30 were originally registered on 2026-07-21. The register was
expanded on 2026-07-23 after the blanket fidelity claim was invalidated and the
user required exhaustive finite coverage of every material textual
interpretation plus separate corrected controls. Historical implementation-v1
results retain their original fingerprints and do not retroactively cover the
new options.
