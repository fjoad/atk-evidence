# Paper 1 Paper-to-Code Traceability

**Audit date:** 2026-07-24
**Status:** Gate-C crosswalk complete; Gate-D exact-cache/GPU checks remain
**Paper:** *Deep Autoencoder-Based Anomaly Detection of Electricity Theft
Cyberattacks in Smart Grids*  
**PDF SHA-256:**
`f3098e0c27ee19b27bea026aedc3d10e5dbb0c46f5cd01ed5bd5c05b7dcf850f`

This document uses the PDF as the authority. The old contract, ambiguity
register, tests, cache, and results are audited objects. “Registered” does not
mean “paper-exact.”

## Initial crosswalk

| Topic | Paper locator and requirement | Current implementation | Classification | Consequence |
|---|---|---|---|---|
| ISET source | Sec. II: approximately 3,000 residential units, 30-minute readings, about 1.5 years | Checksum-gated CER archives; allocation `Code=1`; opt-in all-4,225 and deterministic seeded-3,000 selections | `AMBIGUOUS-BRANCH`, both population readings structurally implemented | Exact files are strong; results must identify the population branch and seed |
| ISET sample | Sec. II-B and Fig. 2: one day; model input has 48 values | Opt-in strict slots 1-48, trim DST slots 49/50, mean duplicate slots, or interpolate a recoverable 48-slot grid | `AMBIGUOUS-BRANCH`; all frozen day readings implemented | Each cache records its day policy; the historical cache remains strict-only |
| SGCC sample | Sec. II-A says daily values over three years; Sec. III-A says model input has 48 neurons | Opt-in full sequence, non-overlapping/rolling 48-day windows, first/last 48 days, and deterministic 48-bin means | `PAPER-CONTRADICTION`; all frozen finite readings structurally implemented | Each result must name representation and source-customer split identity |
| SGCC missing values | Not specified | Opt-in drop incomplete customers, zero fill, within-row interpolation plus benign-B1 median edge fill, and customer-mean fill | `AMBIGUOUS-BRANCH`; all registered readings implemented | Missing policy remains assumption-bound and is fingerprinted |
| Attack 1 | Eq. (1): one fixed random factor in [0.1, 0.8] for all samples of an attacked profile; scope outside one profile is absent | Opt-in one factor per profile, customer matrix, or generated dataset | Profile scope is `EXACT`; broader scope is `AMBIGUOUS-BRANCH`; all registered readings implemented | Results must name the factor scope |
| Attack 2 | Eq. (2): dynamic factor in [0.1, 0.8] per reading, while prose alternates hourly and half-hour samples | Opt-in factor per half-hour or one hourly factor repeated over two slots | `AMBIGUOUS-BRANCH`, both readings implemented | Structural tests verify factor reuse |
| Attack 3 | Eq. (3): zero a 4-24 hour interval, but prints `tf = ti - tl` | Printed subtraction retained as non-executable; opt-in valid-fit addition, printed-start truncation, and printed-start wrap under both hour mappings | `NON-EXECUTABLE` printed node plus all registered minimal repairs implemented | Results must name interval repair and hour mapping |
| Attacks 4-6 | Eqs. (4)-(6): daily mean, dynamic fraction times mean, reverse day | Implemented directly | `EXACT` | Structural tests exist |
| Attack regeneration | Random draws are required but reuse across models/experiments is unstated | Opt-in fixed per data seed, regenerated per model seed, or deterministically regenerated per experiment | `AMBIGUOUS-BRANCH`; all registered seed schedules implemented | Each derived cache records the resolved seed and derivation |
| ISET attack population | Sec. II-B: apply all six attacks to all customers; Sec. II-C: training customers do not appear in test | Implementation-v1 generates only B2 attacks; opt-in preparation now executes either all-customer M or B2-only M and preserves the choice in metadata | `PAPER-CONTRADICTION` with both branches implemented | Full caches for the new branches have not yet been built |
| Anomaly split | Sec. II-C: split benign customers 2:1 into B1/B2; later prose also uses rows/profiles | Opt-in customer-disjoint or row/profile-random 2:1 split with source-profile and meter provenance | Customer branch is `EXACT`; row branch is `AMBIGUOUS-BRANCH`; both implemented | Fixture tests prove row disjointness does not imply unseen meters |
| Neural anomaly training population | Sec. II-C defines `X_TR = B1`; Secs. III-E/IV-B mention cross-validation/validation but give no fraction or final refit rule | Ordinary and DDP runners execute fixed epochs on all B1, holdout without refit, holdout then all-B1 refit, and five-fold-or-maximum-feasible cross-validation then all-B1 refit; every selection/refit history and timing is retained | All frozen A33 `AMBIGUOUS-BRANCH` policies implemented and branch-ID wired | Real multi-GPU Gate-D smoke remains before scaled execution |
| Normalization | Sec. II-C: normalize both classes to zero mean/unit variance before split | Feature-wise joint B+M scaling, fitted using B plus held-out-only M | `AMBIGUOUS-BRANCH` | Axis and joint-vs-separate scope are unstated; all-customer M would change the scaler |
| Anomaly ADASYN | Sec. II-C: balance B2+M in the testing set; benign is ISET minority | Applied after B2+M | `EXACT` in placement; parameters ambiguous | Defaults/seed are an assumption |
| Supervised population | Sec. II-C: concatenate benign and malicious classes for all customers, apply ADASYN, then split 2:1 | Implementation-v1 cache concatenates all 2,251,290 benign profiles with 4,504,626 B2 attacks; opt-in all-customer M generates six attacks for every source profile | Historical cache `MISMATCH`; source-v2 branch structurally implemented | Existing Table III benchmark rows from cache SHA-256 `ab88f...b98a987` remain ineligible; replacement cache is pending |
| Supervised disjointness | Sec. II-C first says splitting is over customers; supervised paragraph says disjoint train/test after ADASYN | Both row-stratified pre-split ADASYN and customer-disjoint training-only ADASYN are executable; corrected test rows are untouched | `PAPER-CONTRADICTION` / both `AMBIGUOUS-BRANCH` paths implemented | Stable branch and content-addressed preparation IDs prevent cache/result conflation |
| FC-SAE layers | Fig. 3/Sec. III-A: hidden encoder layers in addition to a latent layer; Table I/Sec. IV-C: four encoder widths (400,300,200,100), opposite decoder, `L=8` total hidden layers | Implementation-v1 has only seven hidden transformations; opt-in source-v2 has all four encoder and four opposite-order decoder widths plus both latent-placement readings | Historical `MISMATCH`; source-v2 structurally implemented | All current FC-SAE result artifacts remain invalid for the printed architecture; replacement runs are pending |
| LSTM-SAE layers | Fig. 3 depicts a latent layer, but Algorithm 2 explicitly sets the decoder's initial hidden/cell states to the encoder's terminal states; Table I/IV-C specify encoder (500,300) and decoder (300,500) | Encoder (500,300) terminal states feed decoder (300,500) | `AMBIGUOUS-BRANCH`, structurally faithful to Algorithm 2 | Retain as the existing-bottleneck branch; distinct-projection and decoder-schedule branches remain required |
| LSTM decoder schedule | Algorithms 2/4 omit the ongoing decoder input; Algorithm 5 explicitly feeds the reconstructed value back with attention context | Source-v2 executes repeat-latent, first-latent-then-zero, and autoregressive reconstruction plus mirrored/top-only state transfer for SAE/VAE/AEA | All registered `AMBIGUOUS-BRANCH` schedules structurally implemented | Every result must name schedule/state; repeat/first-step AEA and Algorithm-5 autoregressive AEA remain separate branches |
| FC-VAE layers | Fig. 4 places a latent distribution after all hidden encoder layers; Table I/Sec. IV-C specify encoder (500,400,300,100), full opposite decoder, and `L=8` hidden layers | Implementation-v1 omits the fourth hidden encoder/decoder widths; source-v2 instantiates all four widths around a distinct latent distribution | Historical `MISMATCH`; source-v2 structurally implemented | Current FC-VAE artifacts remain ineligible; replacement runs are pending |
| LSTM-VAE layers | Fig. 4/Table I/Sec. IV-C: encoder (400,300), distinct latent distribution, decoder (300,400) | Encoder (400,300), latent distribution, decoder (300,400) | `EXACT` for hidden-layer structure; latent width assumed | Decoder, latent-width, and probability-score ambiguities remain |
| AEA layers | Fig. 5/Algorithm 5 show encoder (500,300,200), attention context, and decoder (200,300,500); following prose also says the context is passed to a latent layer | Encoder/decoder widths match and attention context feeds the decoder without a separate projection | `AMBIGUOUS-BRANCH`, structurally faithful to Algorithm 5/Fig. 5 | Retain as the existing-context branch; distinct-projection and attention-topology branches remain required |
| Attention | Sec. III-C, Eqs. (11)-(13) and Algorithm 5: feed-forward alignment uses the previous decoder state; context is a weighted encoder-state sum; prose says concatenate context with reconstructed output while Algorithm 5 uses a summation symbol | Trainable decoder-conditioned additive attention executes concatenate/literal-sum merges for repeat/first-step schedules and a dedicated autoregressive decoder that feeds the prior scalar reconstruction back at every step | All registered `AMBIGUOUS-BRANCH` merge/schedule readings structurally implemented | Externally motivated scorer topologies remain corrected controls, not paper branches |
| SAE/AEA loss and score | Sec. III-A/C: MSE reconstruction error; higher means anomaly | Mean per-sample squared error; higher means anomaly | `EXACT` | Threshold scale still depends on preprocessing |
| VAE training loss | Eq. (10): squared L2 reconstruction term plus KL | Source-v2 executes both literal sum-squared-plus-KL and common mean-MSE-plus-KL; the learned-variance prose branch uses the corresponding summed/mean Gaussian data term so its variance head is trainable | Historical `MISMATCH`; both frozen A31 branches structurally implemented | The deterministic branches differ by the input dimension before KL; learned-variance NLL is a prose-consistent completion rather than the literal squared-error term |
| VAE anomaly score | Sec. III-B: Monte Carlo reconstruction probability under Gaussian encoder/decoder; low probability means anomaly | Historical artifacts use deterministic MSE surrogates; source-v2 computes stable multivariate Gaussian reconstruction density for MC `{1,10,100}` with fixed variance or a separately trained decoder variance head | Historical `NOT-IMPLEMENTED`; source-v2 score operation and both loss-reduction branches implemented and runner-wired | Fixed variance plus literal Eq. (10) is the direct printed-loss branch; learned-head Gaussian NLL is a prose-consistent completion |
| VAE orientation | Sec. III-B says low probability is anomalous; Sec. III-C later says probability greater than threshold is anomalous | Runner now preserves lower-is-anomaly for raw probability and higher-is-anomaly for MSE/MSE+KL cost branches | `PAPER-CONTRADICTION` with orientations separated | Do not transform one branch post hoc to match a threshold |
| Threshold values | Sec. IV-B: 0.58/0.61/0.43/0.47/0.51 | Uses those printed values | `EXACT` numerically | Applying them to a different score definition is not exact |
| Threshold derivation | Secs. III-D/E and IV-B: ROC/IQR rule on ISET cross-validation/validation | Ordinary and DDP runners independently execute printed constants or three ROC/IQR repairs, ISET-transfer versus dataset-specific scope, and B1-generated-attack/B2-carve-out/no-derivation populations; B2 validation identities are removed from final test | Printed phrase remains `NON-EXECUTABLE`; all frozen executable formula/scope/population branches are structurally implemented | SGCC transfer requires a frozen ISET artifact; corrected controls separately use validation Youden J |
| Metrics | Sec. III-D: DR, FA, SP, PR, `ACC=(DR+SP)/2`, F1, ROC-AUC | Formulas implemented from one confusion matrix and score vector | `EXACT` | Metric code is eligible |
| Training duration and convergence | Algorithms 1-5 say iterate until convergence; epochs, batch, learning rate, initialization, and stopping are absent | Frozen epoch/batch envelope plus executable fixed, early-stop/no-refit, holdout/refit, and cross-validation/refit policies | `AMBIGUOUS-BRANCH`; ordinary/DDP policies implemented | Later confirmatory envelope freeze remains required |
| Keras | Sec. IV: Keras Sequential API | Keras 3 functional models with Torch backend | `AMBIGUOUS-BRANCH` | Layer semantics can match, but backend/version is not known |
| SVM training cap | No sample cap stated | Both historical 12,000/30,000 caps and uncapped full-data fits execute | `ADDED` cap diagnostic plus corrected/full-data control | Current capped rows remain diagnostics; future results must identify the branch |
| ARIMA | Sec. IV-C gives differencing 1 and moving average 0; autoregressive order and score procedure absent | All 16 frozen combinations of `p={0,1,2,5}`, pooled/per-profile fit, and residual-MSE/Gaussian-likelihood score execute | `AMBIGUOUS-BRANCH` with explicit finite completions | No one completion can establish failure of every unspecified ARIMA procedure |
| “Multiclass SVM” labels | Sec. IV-A names multiclass SVM; class construction is absent | Binary benign/malicious and benign-plus-six-attack labels execute for ISET; SGCC remains binary because it has no six attack IDs | `AMBIGUOUS-BRANCH` on ISET; seven-class SGCC is not source-executable | Every result records label construction; impossible SGCC request fails loudly |
| Table IV size | Table IV: 0.5/0.75/1.0 of `X_TR`; text calls full `|X_TR|` 60 million | Treats 60M as 48 scalar readings per profile; 15% carve-out produces 61.206M | `INFERRED AMBIGUOUS-BRANCH` | Close cardinality supports but does not prove the interpretation |
| Table IV time | Table IV gives minutes but no hardware, epoch count, repetitions, or timing boundary | Records local/the cluster fit and wall times | `NON-EXECUTABLE` for direct equality | Report timing provenance; do not make an absolute reproduction claim |
| Table V balance | Sec. IV-D: no ADASYN because data are already balanced | Equal benign/attack rows | `EXACT` | Balance itself matches |
| Table V sample count | Not stated | Direct Table-V runner executes full heldout and deterministic seeded-3,000-per-class branches | `AMBIGUOUS-BRANCH`, both sizes implemented | Historical 3,000-only derivation remains a bounded diagnostic |
| Table V experiment identity | Sec. IV-D says “multiple experiments”; reported FA varies by attack | Direct runner executes common model/common benign, per-attack retraining, per-attack benign resampling, and both; all six score/identity sets are persisted | All registered `AMBIGUOUS-BRANCH` identities implemented | Full-set resampling is explicitly recorded as degenerate; fixed model/common benign must have invariant FA |

The repository's 137 study tests and 10 project tests pass. This is not
counterevidence to the findings above: the tests verify the implementation
against its frozen contract. In particular,
`test_paper_optimizers_activations_dropout_and_mirroring` explicitly expects
only three FC-SAE decoder layers. The test therefore preserves the mismatch
instead of detecting it. Gate C requires tests derived from source claims and
runtime layer counts, not tests derived from the existing builder.

## Result eligibility after this audit

| Existing result family | Status for paper-reproduction claims | Retained use |
|---|---|---|
| FC-SAE, any dataset/table | **INVALIDATED** | Implementation-v1 diagnostic only; printed FC architecture was not instantiated |
| FC-VAE, any dataset/table | **INVALIDATED** | Implementation-v1 surrogate diagnostic only; architecture and score differ |
| LSTM-VAE, any dataset/table | **INVALIDATED for stated VAE detector** | MSE-surrogate branch only |
| LSTM-SAE Table II/III/IV | **QUARANTINED** | Algorithm-2 existing-bottleneck branch; data representation, validation, and decoder schedule remain unresolved |
| LSTM-AEA Table II/III/IV | **QUARANTINED** | Algorithm-5 existing-context branch pending attention and data/training review |
| Current Table V derivations | **INVALIDATED as direct Table V reproduction** | Fixed-model/invariant-FA diagnostic demonstrating that the published varying FA requires a different experiment identity |
| ISET supervised rows from cache v1 | **INVALIDATED** | Diagnostic only; malicious population contradicts the supervised paragraph |
| Capped SVM and pooled ARIMA rows | **QUARANTINED / assumption-bound** | Resource/implementation branches, not exact benchmark failures |
| Metric arithmetic audit | **VALID STATIC EVIDENCE** | Independent of model/data implementation |
| Source checksums and raw-data verification | **VALID PROVENANCE EVIDENCE** | Input identity is unaffected |

## Required correction branches before more compute

1. Instantiate and structurally test every paper model with all printed encoder
   hidden layers and the full opposite-order decoder. For SAE/AEA, retain both
   distinct-latent and algorithm-literal existing-bottleneck branches; for VAE,
   retain the explicit latent distribution under the frozen width envelope.
2. Rebuild ISET data with an all-customer `M` branch and preserve the
   heldout-customer `M` branch as the competing resolution of the paper's
   contradiction.
3. Fix the supervised ISET path to use the branch's full declared `M`
   population.
4. ~~Decide whether neural final fits use all B1 after validation or remain on
   a training-only subset.~~ All four frozen policies now execute in the
   ordinary and DDP runners with branch-ID dispatch.
5. ~~Implement explicitly parameterized reconstruction-probability VAE
   branches.~~ Fixed/learned decoder variance and 1/10/100 Monte Carlo draws
   execute with explicit score direction.
6. ~~Define Table V as explicit per-attack experiments and a full-set
   branch.~~ Four identity readings and two sizes execute; the fixed-model
   invariant-FA case is retained as a structural diagnostic.
7. ~~Remove resource caps from any row labeled full reproduction.~~ Capped
   diagnostics and uncapped full-data controls are separate branches.
