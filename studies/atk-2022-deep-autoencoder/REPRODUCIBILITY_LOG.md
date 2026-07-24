# Reproducibility log

## 2026-07-24 - Gate-C branch execution closure and bounded Gate-D sanity

- Added stable branch-ID resolution from the frozen lattice through the public
  runner. Each branch maps to its dataset preparation, source-v2/corrected
  model contract, classical completion, validation/refit policy, threshold
  rule/scope/population, and Table-V identity.
- Made branch caches content-addressed by preparation semantics and embedded
  their branch/preparation identity in cache metadata. A mismatched or
  historical cache now fails closed instead of being reused by filename.
- Brought the DDP neural path to semantic parity for validation, final refit,
  thresholds, score orientation, model overrides, preparation policies, and
  immutable fingerprints. The short Slurm wrapper accepts a config and direct
  runner arguments; no job was submitted.
- Implemented all 16 frozen ARIMA completions, capped/full-data SVM paths, and
  ISET binary/seven-class SVM labels. Removed the seven-class dimension from
  SGCC because SGCC supplies no six attack identities; the runner also rejects
  such a request explicitly.
- Implemented the separately labeled corrected-control contract: train-only
  isolation/ADASYN, full-data classical fits, likelihood VAE scoring, binary
  sigmoid heads, and validation-selected Youden-J thresholds.
- Added zero-trust closure tests that fail if any paper dimension is not
  consumed by a preparation/model/execution surface or if a corrected-control
  score name claims calibration/probability semantics the runner does not
  execute.
- Added `PAPER_WORKFLOW.md`, a source-first flowchart that separates explicit
  paper steps from ambiguities, contradictions, paper-consistent branches, and
  corrected controls.
- Full verification passes: 137 study tests and 10 project tests. Bounded
  neural source cells pass 27 tests in 3.855 s runner time (7.72 s wall);
  data/classical/metric/ordinary/ISET/DDP cells pass 71 tests in 4.595 s
  (7.16 s wall).
- A real SGCC printed FC-SAE anchor preflight verified the source SHA-256 and
  prepared 42,367 retained customers in 6.806 s (9.55 s wall). The exact-ISET
  anchor stopped at missing content-addressed cache
  `prep-20792e1602ac8e5d` in 3.31 s, as intended. No production cache was
  rebuilt and no the cluster job was submitted.

## 2026-07-24 - Validation, threshold scope, and final-refit branches

- Corrected the branch contract so threshold formula and threshold derivation
  scope are independent. Four impossible formula/label/scope pairs are now
  machine-readable exclusions rather than counted cases.
- Implemented deterministic B1-generated-attack validation, stratified B2
  validation carve-out with exact final-test row removal, and printed-threshold
  no-derivation populations in the ordinary runner.
- Implemented dataset-specific ROC selection and ISET-derived transfer.
  SGCC transfer fails closed unless a frozen threshold artifact is supplied.
- Implemented fixed epochs, holdout/no-refit, holdout/all-training refit, and
  five-fold-or-maximum-feasible cross-validation/all-training refit for both
  anomaly and supervised neural models. All selection and refit histories and
  timings are retained.
- A two-threshold fixture exposed an empty observed IQR under interpolated
  quartiles. The `threshold_iqr_median` repair now records a
  median-all-finite fallback for that degenerate case instead of emitting NaN.
- The full deterministic suite passes: 125 study tests and 10 project tests.
  No production cache was rebuilt and no the cluster job was submitted.

## 2026-07-23 - Table V experiment identities

- Added a direct Table V execution path for all four frozen experiment
  identities: common model/common benign rows, independent model retraining,
  independent benign resampling, and retraining plus resampling.
- Crossed every identity with full-heldout and deterministic seeded-3,000
  evaluation sizes. Full-set “resplitting” is correctly recorded as
  degenerate because selecting the whole set cannot change its identities.
- Independent retraining uses deterministic child seeds from each outer model
  seed. Every six-column run persists all scores, predictions, labels, sample
  identities, child seeds, fit count, timing, and per-column metrics.
- Corrected the historical Table V evaluator to honor lower-is-anomalous VAE
  probability scores; it had previously hard-coded higher-is-anomalous.
- Added the pre-existing ISET runner test module to the canonical test script.
  Regression result: 113 study tests and 10 repository tests passed. No
  production experiment or the cluster job was launched.

## 2026-07-23 - Attack-regeneration schedules

- Implemented all three frozen attack-reuse readings: one generated attack set
  per data seed, regeneration per model seed, and deterministic regeneration
  per experiment index.
- The latter two require their missing identifiers and fail loudly otherwise.
  Every cache records the resolved attack seed and its complete derivation;
  an explicit manual seed is accepted only for the fixed-seed branch.
- Regression result: 107 study tests and 10 repository tests passed. No
  production cache was rebuilt and no the cluster job was submitted.

## 2026-07-23 - ISET day and split-unit branches

- Implemented all four frozen ways to create the paper's 48-value ISET day:
  require exactly slots 1--48, trim documented DST slots 49/50, average
  duplicate meter/day/slot rows, or interpolate a recoverable 48-slot grid.
- Implemented customer-disjoint and row/profile-random ISET 2:1 splits.
  Source-profile digests and meter populations are preserved separately. A
  fixture proves the row-random branch has disjoint profiles but overlapping
  meters, so it cannot support the paper's unseen-customer statement.
- Heldout attacks are selected by source-profile identity rather than only by
  meter ID. This is required for the row-random branch, where the same meter
  can contribute profiles to both B1 and B2.
- Regression result: 106 study tests and 10 repository tests passed. No
  production cache was rebuilt and no the cluster job was submitted.

## 2026-07-23 - SGCC representation and missing-data branches

- Implemented all six frozen resolutions of the SGCC 1,034-days-versus-48-
  inputs contradiction: the complete chronological sequence, non-overlapping
  48-day windows, rolling 48-day windows, first 48 days, last 48 days, and 48
  deterministic contiguous-bin means.
- Implemented all four missing-data readings: drop incomplete customers, zero
  fill, within-customer interpolation plus benign-B1 edge medians, and
  per-customer mean fill.
- Every generated window has a stable sample ID and source-customer ID.
  Customer-disjoint splitting keeps source identities disjoint across
  anomaly train/validation/test; the competing row-random branch is explicit
  and a fixture test demonstrates that it crosses source customers.
- The ordinary and DDP runners expose the same choices. Preparation semantics
  are now part of immutable run fingerprints and preflight records, preventing
  semantic branches with coincidentally identical arrays from colliding.
- Regression result: 104 study tests and 10 repository tests passed. No
  production cache was rebuilt and no the cluster job was submitted.

## 2026-07-23 - Attack and residential-population branches

- Added every registered Attack-1 factor scope: one factor per profile, per
  customer matrix, or per generated dataset.
- Added both Attack-2 time readings: a new factor per half-hour and one factor
  repeated across each pair of half-hours.
- Preserved Attack 3's printed subtraction as a non-executable manifest node
  and implemented its three frozen minimal repairs: valid-fit addition,
  printed-start truncation, and printed-start circular wrap. Each repair runs
  under both direct 48-index and two-slots-per-hour mappings.
- Added both ISET residential-population readings: all 4,225 eligible
  allocation-code-1 meters and a deterministic seeded 3,000-meter subset.
  Every choice and attack seed is written into cache metadata.
- Corrected a pre-test CLI wiring mistake that sent the new ISET-only options
  to SGCC and omitted them from ISET. No data cache or result was produced
  while that mistake existed.
- Regression result: 101 study tests and 10 repository tests passed. No
  production cache was rebuilt and no the cluster job was submitted.

## 2026-07-23 - Decoder, VAE-probability, and threshold branches

- Added opt-in recurrent input layouts (48 steps by one feature and one step
  by 48 features), repeat/first-step/autoregressive SAE/VAE decoder schedules,
  mirrored/top-only state transfer, and concatenate/literal-sum attention
  merges. Autoregressive attention remains explicitly pending.
- Implemented all eight frozen VAE score IDs. Monte Carlo reconstruction
  density uses a stable multivariate Gaussian calculation for 1, 10, or 100
  latent draws and supports fixed variance or a separately trained decoder
  variance head. The ordinary neural runner now evaluates raw probability as
  lower-is-anomaly and keeps MSE/MSE+KL as higher-is-anomaly branches.
- Implemented supplied printed/dataset thresholds plus central-ROC-point
  median, threshold-IQR midpoint, and threshold-IQR median interpretations.
  Validation-population construction remains a separate unfinished gate.
- Tiny tests verify finite learned-variance training/scoring, deterministic
  fixed-variance Monte Carlo scoring, all score directions/counts, decoder
  shapes, attention merges, and threshold derivations.
- Regression result: 96 study tests and 10 repository tests passed. No
  production cache was rebuilt and no the cluster job was submitted.

## 2026-07-23 - Executable data-policy branches

- Preserved implementation-v1 defaults and added opt-in SGCC/ISET preparation
  semantics for all registered scaling readings: joint feature-wise,
  class-specific feature-wise, per-profile, and training-benign-only.
- Added the paper-printed anomaly-test ADASYN path and a separate no-test-
  resampling corrected path.
- Added supervised pre-split ADASYN and customer-split/training-only ADASYN.
  Fixture tests verify that corrected validation/test populations contain no
  synthetic rows and that original supervised customer/meter identities do
  not cross the train/test boundary.
- Added both sides of the ISET A06/A32 contradiction: attacks from held-out B2
  customers only and attacks from every customer. The all-customer fixture
  generates exactly six malicious profiles per benign source profile while
  retaining B2-only unseen anomaly evaluation.
- Exposed the choices through the existing `prepare_data.py` entry point. No
  production cache was rebuilt and no the cluster job was submitted.
- Regression result: 85 study tests and 10 repository tests passed.

## 2026-07-20 - Initial audit and acquisition

- Read the complete 12-page journal article and visually checked its figures and tables.
- Confirmed that the article reports timing but no hardware or sufficiently detailed software/training configuration.
- Located the SGCC dataset through the corresponding author's page and its linked repository.
- Cloned repository commit `8db682e65422d24689a61bd044eab7235121c5df`.
- Recorded SHA-256 checksums for all three multipart archive files.
- The built-in macOS unzip produced a CRC failure because it does not support the archive format.
- Installed Homebrew `sevenzip` 26.02.
- Tested the three-volume archive successfully with 7-Zip, then extracted `data.csv`.
- Verified extracted CSV SHA-256: `99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7`.
- Profiled the CSV: 42,372 customers, 1,034 date columns, 8.53% positive labels, 25.64% missing consumption cells.
- Observed that source date columns are lexicographically rather than chronologically ordered.
- Located the official Irish CER/ISET record: DOI `10.7929/ISSDA/BX59EU`.
- Downloaded and checksum-verified its unrestricted manifest and documentation.
- Confirmed through official metadata that the six consumption archives are restricted and total approximately 658 MB.
- Searched this machine for a previously downloaded authorized CER archive; none was found.
- Located ScienceDB DOI `10.57760/sciencedb.17619`, a public third-party deposit
  containing all six official archive filenames, byte sizes, and MD5 values.
  Anonymous range requests verified all six direct objects on 2026-07-21.
- Downloaded all six archives locally. Every file matches the official MD5 and
  size and passes ZIP integrity. The allocation CSV contains 6,445 unique
  assignments and matches a second public allocation workbook over every row
  and semantic column after blank normalization.
- A complete 157,992,996-row scan found all 4,225 residential meters, all 485
  SME meters, no unallocated reading meters, no malformed three-field rows, and
  no invalid or negative values. The 540 suffix-51--95 records belong only to
  two unused `other` meters; residential data are unaffected.
- Created a project-local Python 3.12 environment and froze its dependency versions.
- Implemented and unit-tested all six CER attack functions. The controlled bypass implementation interprets the stated duration as hours and corrects the printed subtraction of interval length to addition; both choices are explicitly documented.
- Implemented a machine-readable audit of Tables II and III. The reported precision values do not agree with DR and FA under the stated balanced test-set protocol, and the rows imply different positive-class prevalences.
- Ran a first SGCC customer-profile sanity audit using a fixed 60/20/20 customer split, train-only preprocessing, and a validation-selected 5% false-alarm threshold. This is not an exact paper reproduction because the paper does not define its SGCC 48-value input construction.
- Preliminary SGCC sanity result: simple summary/logistic/PCA models achieved test ROC-AUC values of approximately 0.59-0.70, not saturation. This result concerns the real SGCC customer labels; it does not test the much easier synthetic CER attacks.
- Located and read the two same-author precursor papers containing the basic-AE and VAE experiments. Visually verified their threshold and result tables against the journal article.
- Confirmed exact recurrence of the ISET FC/LSTM basic-AE thresholds and TPR/FPR/AUC values, and exact recurrence of the FC/LSTM VAE thresholds and DR/FA values.
- The ISSCS precursor states that malicious data are used only for testing but derives thresholds from ROC curves; the EUSIPCO precursor explicitly applies ADASYN within the test set and also derives thresholds from ROC curves. This is evidence of test-dependent model selection under the written protocols, not evidence that the numerical measurements were fabricated.
- Neither precursor supplies the missing hardware, epoch, batch-size, seed, repetition, or uncertainty information.

## Local execution environment

- Hardware: Apple MacBook Pro, Apple M1 Max, 10 CPU cores, 32 GPU cores, 64 GB unified memory.
- OS: macOS 26.5.2, build 25F84.
- Experiment environment: a project-local Python virtual environment will be used and package versions frozen after installation.

Hardware serial numbers and device identifiers are intentionally omitted.

## Decisions requiring explicit provenance

- Do not use an unofficial CER mirror unless its legal status is clear and its files match the official MD5 values.
- Never modify raw downloaded files in place.
- Preserve failed extraction artifacts until the verified copy and checksums are recorded; failed artifacts are excluded from all analyses.
- Separate paper-literal reproduction from controlled scientific evaluation.
- For the controlled CER bypass attack, use 4-24 actual hours (8-48 half-hour slots) and `tf = ti + tl`; retain a separate literal implementation only if needed for diagnostic comparison.

## 2026-07-21 - the cluster exact-shape resource probes

- Verified the SGCC checksum again on the cluster and retained the full 1,034-feature input, paper-width models, global batch 512, and four equal local batches of 128.
- Used four Tesla V100-PCIE-16GB GPUs because the cluster's T4 nodes expose only three GPUs and cannot evenly shard 512 four ways.
- Measured rough full-epoch extrapolations of 844.96 seconds for LSTM-SAE, 75.09 seconds for LSTM-VAE, and 1,218.92 seconds for the supervised LSTM. These are short resource/timing probes, not Table II results.
- The LSTM-AEA attention call attempted to allocate 101.96 GiB per rank from two `[128, 1034, 200]` tensors and failed its primary-batch resource gate. No smaller batch was silently substituted.
- An initial supervised-LSTM probe using Torch-native BCE/Adam hit a CUDA input assertion. A discriminating rerun used the compiled Keras BCE and Adam plus synchronized finite checks for the loss, gradients, parameters, and optimizer state. It completed cleanly and all four ranks ended with the same parameter hash, invalidating the earlier failure as evidence about the paper-literal model.
- Preserved exact job IDs, commits, timings, memory peaks, hashes, failure scope, and remote artifact paths in `results/panther_resource_probes.json`.
- Isolated a small ADASYN environment sensitivity: identical inputs, code,
  packages, and seed produced 77,708--77,712 supervised rows as OMP/MKL thread
  settings changed. The the cluster four-GPU branch fixes both variables at 2 and
  records actual partition sizes; this difference is too small to explain the
  much larger completed classical metric gaps.
- the cluster job 348195 completed 12 immutable classical Table II attempts. All
  manifests and fingerprints verified. Naive Bayes, ARIMA, one-class SVM, and
  multiclass SVM reproduced DR values of 7.97%, 2.10%, 61.78%, and 53.51%
  against reported 75%, 88%, 91%, and 92%; none reproduced its complete metric
  pattern. These are partial exploratory Table II results, not a final verdict.
- Implemented the explicit four-`v100_16GB` production DDP branch for neural
  Table II attempts. It retains the global batch, compiled Keras loss/optimizer,
  all samples, global validation, existing EarlyStopping restoration, original
  test order, and immutable rank-0 persistence while fingerprinting distributed
  shuffle and stochastic-stream choices. All 66 study and 7 repository tests
  pass; a real four-GPU end-to-end validation remains required before matrix
  execution.

## 2026-07-21 - Independent paper re-read and implementation fidelity audit

- Re-read all 12 pages of the journal article end to end and audited every
  paper-literal source module and the frozen TOML contract against it.
- Confirmed the implementation matches the paper or a registered branch for
  architectures, attacks, metrics, thresholds, preprocessing order, and
  benchmark settings; no silent repair or undocumented extra method was found.
- Registered previously untracked items before any Table III--V execution:
  ambiguities A27--A30 and three non-blocking internal inconsistencies
  (conflicting headline improvement ranges, the hourly SGCC attack description
  against daily data, and the ~3,000-vs-4,225 residential meter count).
- Cross-checked the transcribed `reported/` tables against the PDF; all
  spot-checked cells match, and Table II satisfies SP = 100 - FA row by row.
- Noted an operational constraint: editing `config/exploratory_reproduction.toml`
  changes `contract_sha256` and invalidates resume-skip fingerprints for every
  SGCC Table II attempt; ISET-phase contract additions (`table_v_samples`, the
  ScienceDB allocation branch) belong in the planned ISET gate change.

## 2026-07-23 - Zero-trust fidelity correction

- Re-rendered and visually inspected all 12 pages of the exact PDF, SHA-256
  `f3098e0c27ee19b27bea026aedc3d10e5dbb0c46f5cd01ed5bd5c05b7dcf850f`.
- Treated the existing contract and ambiguity register as audit objects rather
  than sources of truth. This invalidated the 2026-07-21 blanket `VERIFIED`
  conclusion.
- Confirmed by runtime layer inventory that FC-SAE implements seven hidden
  transformations rather than the paper-described four encoder plus four
  opposite decoder layers. FC-VAE also does not instantiate the explicit
  four-plus-four hidden layout under the paper's own layer-count wording.
- Found that ISET supervised preparation uses malicious profiles from held-out
  B2 customers only, contrary to the supervised paragraph's all-customer
  benign-plus-malicious population.
- Confirmed that the paper's VAE reconstruction probability is not
  implemented; current VAE scores are explicit MSE-based surrogates.
- Quarantined all existing results by row, submitted no new jobs, and created
  `PAPER_TO_CODE_TRACEABILITY.md` plus the gated zero-trust audit plan. Jobs
  already running were not cancelled or mutated.

## 2026-07-23 - Exhaustive interpretations and corrected controls

- Recorded the user's requirement to execute three non-conflated families:
  the printed method even where statistically improper, every materially
  defensible interpretation of the paper's ambiguous or contradictory text,
  and a scientifically corrected control.
- Replaced the historical one-primary-branch policy with a finite
  dependency-aware coverage contract. “All” is closed against every material
  PDF statement, omission, contradiction, and invalid expression; exclusions
  require written reasons.
- ADASYN now explicitly spans the paper-printed test-set/pre-split procedures
  and separate corrected controls with untouched test data and training-only
  resampling.
- Expanded `AMBIGUITY_REGISTER.md`, created
  `BRANCH_COVERAGE_CONTRACT.md`, and recorded the evidence-contract decision.
  No new experiment was submitted.
- Converted the contract to `config/branch_lattice.toml` and generated 921
  stable paper-consistent configurations across 22 model/data families: 22
  printed anchors and 899 interpretive cases, plus 22 separately identified
  corrected controls. Five tests verify stable IDs, complete compatible
  option/pair coverage, forbidden-pair exclusion, corrected-control identity,
  and a complete machine mapping for all 36 ambiguity-register rows.
- The bounded screen contains 2,763 attempts (three seeds), estimated at 558.7
  GPU-hours and 57.4 CPU-hours. At three simultaneous GPU jobs this is 49.7
  ideal active hours or 99.4 hours with the frozen 2x runtime factor, excluding
  queue time. The all-promote full-treatment upper bound is deliberately much
  larger and is not the execution plan.
- A visual recheck of Algorithm 6 invalidated the initial eight-evaluation
  search budget. The printed staged loop requires 36 evaluations under the
  literal uniform-width reading, while a per-layer coordinate interpretation
  that can produce Table I's unequal widths requires 86. Both and direct
  Table-I replay are retained; the 86-evaluation branch defines the
  all-promote upper bound of 281,418 attempts (about 2,197,989 GPU-hours and
  59,559 CPU-hours), not the screening plan.
- A visual recheck of Figs. 3-5, Algorithms 2/5, and Table I corrected the
  architecture reading. All printed encoder widths and their full mirrors are
  mandatory, invalidating implementation-v1 FC-SAE/FC-VAE. The recurrent
  figures/prose depict latent representations, but Algorithms 2/5 directly use
  the terminal encoder state or attention context; LSTM-SAE/AEA remain
  quarantined as those algorithm-literal branches, with distinct-projection
  branches also required. LSTM-VAE has the structural components but not a
  paper-specified latent width or reconstruction-probability score.
- Added an opt-in `paper_source_v2` model contract without mutating the
  implementation-v1 defaults or artifacts. It instantiates the missing fourth
  FC decoder/encoder widths, all printed recurrent mirrors, six latent widths,
  both SAE/AEA latent-placement readings, and the registered dense/LSTM
  dropout-placement branches. Four source-derived tests inspect runtime layer
  names, widths, latent units, and dropout arguments. No training job was
  submitted.

## 2026-07-22 - the cluster status audit and refill submissions

- User confirmed batch 512 remains the frozen primary (GPU-utilization motive
  recorded); batch 32 stays the declared sensitivity branch.
- Audited the the cluster queue, sacct history, and immutable attempt tree.
  Complete: classical 12/12, fc_sae 3/3, supervised_feed_forward 3/3,
  lstm_vae seed 11. Failed: fc_vae 3/3 at the DDP post-optimizer gate
  (VAE+Adam signature); two pre-DDP single-GPU lstm_sae OOM attempts retained
  as resource evidence.
- First neural exploratory metrics recorded in the evidence ledger: anomaly
  autoencoder rows are far below their reported patterns at the printed
  thresholds (AUC near 50%), while supervised feed-forward is broadly near its
  reported row.
- Synchronized the the cluster clone to commit `44ec6f3` (docs-only delta).
- Submitted job 354017 (`run_model_ddp.sbatch lstm_sae 11`) and job 354018
  (FC-VAE first-update diagnostic, resubmitting cancelled 348303). Job 348223
  (single-T4 batch-32 lstm_sae sensitivity) left running at >22 h within its
  two-day limit. Three-job cap respected.

## 2026-07-22/23 - Exact ISET preparation and first LSTM-SAE result

- Implemented the named ScienceDB semantic-allocation branch without changing
  the preserved SGCC contract. The parser now applies the paper's residential
  selection before rejecting malformed time codes from unused `other` meters.
- Prepared all exact CER/ISET inputs in about 2 h 40 m. The checksummed cache
  contains 4,225 residential meters; 2,251,290 complete benign daily profiles;
  4,504,626 attack profiles; the paper-positioned anomaly/supervised ADASYN
  partitions; and 3,000 original samples per Table V class.
- The reconstructed half/three-quarter/full Table IV subsets contain
  637,565/956,347/1,275,130 daily profiles, or
  30,603,120/45,904,656/61,206,240 scalar half-hour readings. This independently
  explains the paper's rounded 30M/45M/60M sample-size labels.
- At the user's direction, cancelled the single-T4 batch-32 sensitivity after
  23:27:30. It produced no metric outcome.
- Four-V100 job 354017 completed LSTM-SAE seed 11 after 10 epochs:
  Slurm elapsed 2:24:26, recorded pipeline 2:24:03, fit 2:20:38,
  DR 6.78%, FA 2.22%, AUC 51.89%, versus reported 86%, 12%, and 85%.
- Verified the retained score arrays. An oracle threshold selected on the test
  labels reaches only 55.52% balanced accuracy on the paper-primary test set;
  reversed score direction also remains near chance. Original-only AUC is
  59.73%, so test-set ADASYN worsens but does not create the failure.
- The LSTM-SAE score correlates 0.999999999999996 with mean squared standardized
  input, and FC-SAE/LSTM-SAE scores correlate 0.999999999985. Because 71.60% of
  standardized values are negative while the paper-selected decoder outputs
  are nonnegative, the completed SAEs effectively reconstruct zero and reduce
  to the same input-energy score in this literal branch.
- FC-VAE diagnostic job 354018 completed. The first distributed Adam step was
  finite on every rank; extreme rank-local loss and `z_log_var` gradient scale
  narrow but do not resolve the later failure.
- Added four study-root entry points (`download_data.py`, `prepare_data.py`,
  `run_experiment.py`, `analyze_results.py`) so normal reproduction does not
  require navigating the internal audit modules. A real-data SGCC preflight
  completed successfully in 7.8 seconds.
- Final verification for this update: 72 study tests and 9 repository tests
  passed; strict data verification selected the complete named ScienceDB
  branch; all 40 copied Table II attempts passed artifact-integrity inspection
  (23 matching the current execution contract, 17 retained nonmatching
  historical attempts), and the aggregate selected 20 successful cells.

## 2026-07-23 - Exact-ISET execution gate

- Implemented one small Tables III--V adapter over the existing paper-literal
  models, metrics, and immutable-attempt code. The public entry point remains
  `run_experiment.py`; no second scheduler or orchestration layer was added.
- Table III trains all eleven paper models. Each anomaly model's Table V values
  are derived from the fixed 3,000 benign and six attack identity sets within
  that same Table III score vector, with the same printed threshold. There is
  no attack-specific retraining or threshold selection.
- Table IV retrains only the five anomaly models on the deterministic nested
  half, three-quarter, and full benign-training subsets. The subset identity is
  included in each immutable fingerprint.
- To avoid copying shared multi-gigabyte string provenance into every attempt,
  ISET attempts persist their run-specific score/prediction vectors and bind
  row `i` to row `i` of the checksum-verified cache in the fingerprint and
  execution metadata.
- A real-cache preflight verified SHA-256
  `ab88f180feafb7351ef4530cba2e48a3cbc180af268f8b68016aefc50b98a987`,
  loaded the complete 3.2-GiB cache in 13.65 seconds, and recovered all expected
  cardinalities and 48 features.
- A bounded one-epoch FC-SAE fixture smoke completed construction, Keras fit,
  Table III scoring, Table V identity derivation, and compact artifact
  construction in 6.40 seconds wall time (1.36 seconds fit, 0.54 seconds
  scoring). This is a code-path check, not a reported experimental result.
- Verification after implementation: 72 study tests and 9 repository tests
  pass. The next experimental gate is one timed full-data FC-SAE Table III/V
  seed and one Table IV-half seed on the cluster 16-GB GPUs.

## 2026-07-23 - First exact-ISET result cells

- Transferred the prepared cache to the cluster with resumable `rsync`, set mode
  0600, and reverified SHA-256 before submission. The remote clone ran the
  same source content as commit `db148c6`; later bookkeeping commits preserve
  that execution fingerprint.
- Job 354933 completed FC-SAE seed 11 on the half Table IV training subset:
  30 epochs, 446.17 seconds fit, 79.35 seconds score, 12:27 Slurm elapsed,
  DR 22.67%, FA 37.30%, balanced ACC 42.68%, and AUC 42.56%.
- Job 354932 completed FC-SAE seed 11 on full ISET for Tables III/V:
  30 epochs, 810.28 seconds fit, 76.03 seconds score, 18:21 Slurm elapsed,
  DR 22.50%, FA 37.13%, balanced ACC 42.69%, and AUC 42.59%.
  Full-minus-half balanced ACC is 0.007 percentage points, not the paper's
  reported 13-point increase.
- FC-SAE Table V average is DR 22.04%, FA 49.73%. Per-attack DR spans 1.47%
  to 65.43%. FA is exactly 49.73% for every attack because the fixed benign
  identities, model, threshold, and score direction are reused.
- The FC-SAE score has 0.99945 correlation with standardized input energy.
  Printed-direction oracle ACC is 50.08%; reversed-direction oracle ACC is
  57.72% on paper-primary rows and 64.36% on original-only rows.
- Job 354939 completed FC-VAE seed 11 in 10 epochs (8:39 Slurm elapsed):
  DR 40.43%, FA 53.86%, balanced ACC 43.28%, AUC 40.82%. Its Table V
  average is DR 40.02%, FA 66.03%; reconstruction and MSE-plus-KL surrogate
  branches are numerically indistinguishable at the recorded precision.
- Every listed attempt passed its immutable artifact hashes. These are
  exploratory one-seed outcomes, not a confirmatory paper verdict.

## 2026-07-23 - Eq. (10) loss-reduction branches

- Re-rendered and visually checked the source page containing Eq. (10). It
  prints a squared L2 reconstruction term plus KL, which sums squared
  residuals over the input dimensions. The existing/common Keras reading
  averages MSE before the same KL term.
- Added explicit `sum_squared_plus_kl` and `mean_mse_plus_kl` branches to
  FC-VAE and LSTM-VAE. With zero KL, a four-feature fixture proves losses of
  4 and 1 respectively, preventing materially different KL scales from being
  conflated.
- Learned-decoder-variance branches use the corresponding summed or mean
  Gaussian data term so the variance head is trainable. This is labeled a
  prose-consistent likelihood completion; fixed variance with summed squared
  error is the direct Eq. (10) branch.
- No production cache was rebuilt and no compute job was submitted. Fourteen
  focused source-model tests pass; the complete suite passes with 116 study
  tests and 10 project tests.

## 2026-07-24 - Algorithm 5 autoregressive attention

- Re-rendered and visually checked Fig. 5, Fig. 6, Eqs. (11)-(13), and
  Algorithm 5. Unlike Algorithms 2/4, Algorithm 5 explicitly feeds the
  reconstructed value back into the decoder at every time step. The prose
  calls for concatenation with the attention context, while line 21 uses a
  summation symbol, so both remain required textual branches.
- Added a dedicated recurrent attention decoder that recomputes alignment from
  the previous decoder state, feeds back the prior scalar reconstruction, and
  emits all per-step attention weights. It crosses both merge readings, both
  state-transfer readings, both LSTM input layouts, and both latent-placement
  readings.
- Fixture predictions are finite, attention weights normalize over all encoder
  steps, one-step-input attention degenerates explicitly to its single encoder
  step, and a one-batch training update is finite.
- No production cache was rebuilt and no compute job was submitted. Fifteen
  focused source-model tests pass; the complete suite passes with 117 study
  tests and 10 project tests.
