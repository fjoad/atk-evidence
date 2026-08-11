# ATK Evidence — Evidence and Causal Learnings

**Last updated:** 2026-08-11

## Purpose

This document preserves why conclusions changed: former belief, supporting or
disconfirming evidence, root cause, current conclusion, confidence, and
remaining uncertainty. `STATUS.md` says what is true now; `CONTEXT.md` is
compact active memory; this file prevents durable evidence from being silently
rewritten or lost during compaction.

## Evidence vocabulary

- **VERIFIED:** directly reproduced or confirmed by a discriminating artifact or audit.
- **OBSERVED:** visible in an output or trace, but the cause may remain unresolved.
- **INFERRED:** best explanation supported by evidence but not isolated experimentally.
- **HYPOTHESIS:** plausible claim awaiting a discriminating test.
- **INVALIDATED:** contradicted by later evidence and retained to prevent repetition.
- **OPEN:** unresolved or awaiting external state.

Rank evidence by directness, discriminating power, and provenance. A direct user
statement is primary evidence for project intent; paper text, data artifacts,
and repeated experiments determine technical conclusions.

## Causal record

### Program-level reproducibility hypothesis

- **Former belief/status:** Initial concern was expressed informally as doubt that LSTM or attention could cause the reported gains.
- **Supporting evidence:** No confirmatory experiment has yet been completed.
  Exploratory Paper 1 evidence now includes 20 successful Table II cells:
  every completed anomaly-detector row is far from its reported pattern, while
  the supervised feed-forward positive control learns strong separation.
  Static inconsistencies and omissions remain motivation, not proof.
- **Root cause:** Paper 1 has one strong candidate mechanism in the literal
  SGCC branch (standardized targets outside the decoder output domain), but no
  program-level cause has been established and other papers remain untested.
- **Current conclusion + label:** **HYPOTHESIS** — selected papers' complete numerical result patterns will not reproduce reliably within predeclared paper-consistent implementation spaces.
- **Remaining uncertainty / blast radius:** Entire claim remains open; each paper requires an independent verdict and may falsify the hypothesis.
- **Source artifacts:** `docs/VISION.md`, target PDFs kept locally, `studies/atk-2022-deep-autoencoder/EXPERIMENT_SPEC.md`.

### SGCC-derived 48-value proxy experiment

- **Former belief/status:** It was treated temporarily as a useful mechanism check while CER access was blocked.
- **Disconfirming evidence:** Its 48 values are consecutive SGCC days, whereas the relevant paper experiment uses 48 CER half-hour readings. The paper does not define that proxy construction.
- **Root cause:** Work continued with a substitute representation instead of stopping at the exact-data gate.
- **Current conclusion + label:** **INVALIDATED as reproduction evidence** — preserve its artifacts for transparency, but it cannot support or refute the Paper 1 hypothesis.
- **Remaining uncertainty / blast radius:** It may have exploratory value only if explicitly requested later.
- **Source artifacts:** `studies/atk-2022-deep-autoencoder/src/sgcc_attack_pilot.py`, `studies/atk-2022-deep-autoencoder/results/sgcc_attack_pilot.json`.

### Paper 1 dataset availability

- **Former belief/status:** Both named datasets might be directly downloadable.
- **Disconfirming or supporting evidence:** SGCC was acquired from the author-linked repository and checksum-verified. Official ISSDA metadata marks the CER consumption archives restricted. A 2024 ScienceDB deposit by Zehao Song (DOI `10.57760/sciencedb.17619`) exposes `File1.txt.zip` through `File6.txt.zip`; every displayed filename, byte size, and MD5 matches the official ISSDA manifest, and anonymous one-byte range requests to all six download endpoints returned the corresponding full object sizes. The deposit supplies a converted allocation CSV rather than the official `.tab` artifact.
- **Root cause:** Official access remains approval-gated, but a third party deposited byte-identical copies of the six consumption archives in a public repository.
- **Current conclusion + label:** **VERIFIED** — all six exact archives are local, match the official size/MD5 values, and pass ZIP integrity. **VERIFIED for the named exploratory branch** — the allocation CSV contains 6,445 unique assignments, matches a second public allocation workbook across all five semantic columns, and covers every residential reading meter. It remains explicitly non-identical to the official `.tab` serialization.
- **Remaining uncertainty / blast radius:** ScienceDB's redistribution authority is less clear than ISSDA's, so raw files remain local and the report must distinguish cryptographic archive identity from semantic allocation identity. The official `.tab`, if acquired, could still reveal a serialization or provenance difference; any residential-ID difference would invalidate the semantic branch.
- **Source artifacts:** `studies/atk-2022-deep-autoencoder/results/cer_sciencedb_acquisition.json`, `docs/decisions/2026-07-21-cer-sciencedb-semantic-allocation.md`, local ignored `data/raw/cer-sciencedb/`.

### CER full-content and allocation audit

- **Former belief/status:** Matching archive MD5s and plausible allocation counts were sufficient to proceed, subject to a later parser check.
- **Supporting evidence:** A full scan found 157,992,996 well-formed readings, 6,435 meters, no invalid or negative values, and no reading without an allocation. All 4,225 residential and 485 SME allocation IDs occur. Ten absent allocation IDs are all `other`. Documented suffixes 49--50 occur 24,462 times. An additional 540 rows use suffixes 51--95, but they belong only to meters 1208 and 5221, both `other`; no residential profile is affected.
- **Root cause:** **OPEN** for the two non-residential malformed meter streams; the official FAQ explains 49--50 but not 51--95.
- **Current conclusion + label:** **VERIFIED** — the residential selection needed by Paper 1 is complete and does not contain the unexpected suffixes. **OBSERVED** — validating every category before applying the paper's residential filter would fail on irrelevant `other` rows and is the wrong operation order for this branch.
- **Remaining uncertainty / blast radius:** The two malformed non-residential streams must remain documented and excluded only through the declared residential allocation filter, not a global silent cleanup.
- **Source artifacts:** `studies/atk-2022-deep-autoencoder/results/cer_sciencedb_acquisition.json`, official local `CER_FAQ.pdf`.

### Exact CER/ISET paper-literal preparation

- **Former belief/status:** Tables III--V were blocked behind an unimplemented
  allocation branch even after the six exact consumption archives and
  semantically validated allocation CSV were local.
- **Supporting evidence:** The explicit
  `sciencedb-csv-semantic-equivalence-v1` gate now verifies all seven files,
  filters residential IDs before validating irrelevant non-residential time
  codes, constructs complete 48-slot days, applies all six paper equations,
  performs the paper-positioned joint standardization and ADASYN, and persists
  a checksummed cache. Preparation completed in about 2 h 40 m with 4,225
  residential meters, 2,251,290 benign daily profiles, and 4,504,626 attack
  profiles. The half/three-quarter/full Table IV partitions contain 30,603,120,
  45,904,656, and 61,206,240 scalar readings.
- **Root cause:** The earlier hard block applied official binary identity to
  the allocation table even after the decision record had approved a separate
  semantic-equivalence branch; the code and contract had not yet implemented
  that distinction.
- **Current conclusion + label:** **VERIFIED** — the exact archive content and
  declared allocation branch now produce a complete, internally checked ISET
  experiment cache. **INFERRED with high confidence** — Table IV's published
  30M/45M/60M “sample sizes” count scalar half-hour readings rather than
  48-value model examples, because the independently reconstructed counts
  align within 2.0%.
- **Remaining uncertainty / blast radius:** The allocation serialization is
  still not the official `.tab`; this remains an exploratory branch. No
  Tables III--V model result exists until the prepared cache is wired to an
  immutable execution path and actually run.
- **Source artifacts:** ignored
  `data/derived/atk-2022-deep-autoencoder/iset-paper-literal.{npz,json}`;
  NPZ SHA-256
  `ab88f180feafb7351ef4530cba2e48a3cbc180af268f8b68016aefc50b98a987`;
  `config/exploratory_iset.toml`.

### Paper 1 reported training times

- **Former belief/status:** A single one-to-four-hour estimate was being used informally for the remaining neural experiments.
- **Disconfirming or supporting evidence:** Table IV reports full-ISET training times of 137, 183, 141, 188, and 193 minutes for FC-SAE, LSTM-SAE, FC-VAE, LSTM-VAE, and LSTM-AEA. The paper supplies no hardware, epochs, batch size, software versions, timing boundary, or repetitions, and these claims concern ISET rather than SGCC Table II.
- **Root cause:** The paper's timing table was treated as if it were a directly portable runtime forecast despite missing experimental context and a different dataset/table.
- **Current conclusion + label:** **VERIFIED** as a transcription of the paper; **INVALIDATED** as a direct the cluster SGCC ETA.
- **Remaining uncertainty / blast radius:** The reported times may still be useful as qualitative context, but cannot validate or refute measured the cluster timings.
- **Source artifacts:** `studies/atk-2022-deep-autoencoder/reported/table_4.csv`, local source PDF page 11.

### the cluster recurrent resource and timing probes

- **Former belief/status:** The recurrent SGCC branches might share a roughly similar one-to-four-hour runtime and could proceed directly to full training.
- **Disconfirming or supporting evidence:** Four-V100 probes retained the full 1,034-feature models and global batch 512 while timing four updates on 2,048 samples. Rough full-epoch extrapolations were 844.96 seconds for LSTM-SAE, 75.09 seconds for LSTM-VAE, and 1,218.92 seconds for the supervised LSTM. The LSTM-AEA attention call attempted a 101.96-GiB allocation per rank from local `[128, 1034, 200]` tensors and failed before its first update.
- **Root cause:** The architectures have substantially different sequence operations and activation-memory demands; one aggregate ETA concealed those differences.
- **Current conclusion + label:** **OBSERVED** — LSTM-SAE, LSTM-VAE, and supervised LSTM fit on four 16-GB V100s at local batch 128, with rough 10--30 epoch per-seed ranges of 2.35--7.04, 0.21--0.63, and 3.39--10.16 hours. **OBSERVED** — the registered LSTM-AEA implementation is infeasible at the primary global batch on this allocation.
- **Remaining uncertainty / blast radius:** Four deterministic steps are only a rough timing sample. They exclude startup, scoring, persistence, scheduler variation, and the unknown stopping epoch. LSTM-AEA may fit under a smaller paper-consistent batch or different documented attention interpretation; neither has been tested. No probe metric is a Table II result.
- **Source artifacts:** `studies/atk-2022-deep-autoencoder/results/panther_resource_probes.json`, remote immutable probe JSON and SLURM logs listed there.

### Supervised-LSTM probe numerical failure

- **Former belief/status:** Job 348256's CUDA BCE assertion could have indicated that the paper-literal supervised LSTM became numerically invalid after one update.
- **Disconfirming evidence:** That attempt used hand-coded Torch BCE and Torch-native Adam. Job 348262 reran the identical model/data shape using the compiled Keras BCE and Adam after DDP gradient reduction. It completed, kept loss, gradients, parameters, and optimizer state finite on all ranks, and produced one identical final parameter hash across all four ranks.
- **Root cause:** The approximate resource-probe optimizer/loss path was not procedure-equivalent; the exact component responsible within that obsolete path was not separately isolated.
- **Current conclusion + label:** **INVALIDATED as model evidence** — the earlier assertion cannot be attributed to the paper-literal supervised architecture. **OBSERVED** — the corrected sampled execution is numerically stable for five updates.
- **Remaining uncertainty / blast radius:** This establishes short-run numerical feasibility only, not full-training stability or Table II accuracy.
- **Source artifacts:** `studies/atk-2022-deep-autoencoder/results/panther_resource_probes.json`, the cluster logs `slurm-348256.out` and `slurm-348262.out`.

### ADASYN environment sensitivity

- **Former belief/status:** Fixed input, package versions, source code, and random seed were expected to make the paper-positioned ADASYN preparation cardinality deterministic.
- **Disconfirming evidence:** On the same local input and packages, OMP/MKL thread counts 1, 2, and 4 produced 77,712, 77,712, and 77,708 supervised rows; the unconstrained local environment produced 77,710. the cluster preflights recorded 77,712 rows at 2 and 16 threads and 77,710 at 8 threads.
- **Root cause:** **INFERRED** — thread-dependent nearest-neighbor ordering or numerical tie handling changes ADASYN's per-sample allocation rounding. The exact low-level operation has not been isolated.
- **Current conclusion + label:** **OBSERVED** — the execution thread environment is a material provenance field for exact data construction. The current four-GPU the cluster branch freezes OMP/MKL at 2 and derives rather than hard-codes partition sizes.
- **Remaining uncertainty / blast radius:** The observed range is only four of roughly 77,700 rows and cannot plausibly explain the large completed classical metric gaps by itself. Confirmatory branches must freeze or explicitly vary the thread environment.
- **Source artifacts:** `studies/atk-2022-deep-autoencoder/results/adasyn_thread_sensitivity.json`, `docs/decisions/2026-07-21-adasyn-threading-branch.md`, the cluster preflight JSON.

### Paper 1 classical Table II exploratory results

- **Former belief/status:** No end-to-end exact-SGCC Table II metric had yet been reproduced, so concerns about saturation and implausible model rankings remained conjectural.
- **Supporting evidence:** the cluster job 348195 completed Naive Bayes, ARIMA, one-class SVM, and multiclass SVM for seeds 11, 22, and 33. All 12 immutable attempts passed declared artifact hashes and current fingerprint verification. Reproduced mean DR values were 7.97%, 2.10%, 61.78%, and 53.51%, versus reported 75%, 88%, 91%, and 92%. Reproduced AUC values were 54.59%, 49.36%, 61.58%, and 56.98%, versus 73%, 88%, 89%, and 90%. The complete per-row metric patterns were `NOT_CLOSE_MATCH`.
- **Root cause:** **OPEN** — fixed paper thresholds perform poorly under the registered preprocessing and benchmark interpretations, but the relative contributions of preprocessing order, ADASYN placement, benchmark hyperparameter omissions, and score definitions have not yet been isolated.
- **Current conclusion + label:** **OBSERVED** — none of the four completed classical branches reproduces its reported Table II pattern in this frozen exploratory implementation. This is substantive result evidence, not merely a static inconsistency.
- **Remaining uncertainty / blast radius:** The current classical search caps and unresolved paper omissions define a finite branch, neural rows are incomplete, and the experiment was not preregistered before all prior exploratory work. Do not promote this partial table to a final paper-level verdict.
- **Source artifacts:** `studies/atk-2022-deep-autoencoder/results/table_2_classical_panther.json`, the cluster immutable attempt tree and aggregate output, job 348195 log.

### Paper 1 implementation fidelity audit

- **Former belief/status:** The exploratory implementation was built module by
  module against the frozen contract, but no single end-to-end audit had
  compared the full paper text against the full `src/` tree after completion.
- **Supporting evidence:** An independent complete re-read of all 12 paper
  pages followed by a file-by-file audit of `paper_literal_models.py`,
  `paper_literal_data.py`, `paper_literal_iset.py`, `attacks.py`,
  `paper_literal_metrics.py`, `paper_literal_benchmarks.py`,
  `paper_literal_runner.py`, `cer_parser.py`, and the frozen TOML contract
  (2026-07-21).
- **Root cause:** Not applicable; this is a static verification pass.
- **Current conclusion + label:** **VERIFIED** — every Table I architecture
  detail, the six attack equations, the metric definitions (including the
  paper's balanced-accuracy "ACC"), the printed thresholds, the preprocessing
  order (joint pre-split scaling, test-set ADASYN, supervised pre-split
  ADASYN), and the benchmark settings match the paper or a registered branch;
  no silent repair or extra method was found. **OBSERVED** — four implemented
  micro-choices had no individual register entry and were added pre-outcome as
  A27--A30 (residential meter-count gap, multiclass-SVM label cardinality,
  Table V per-class sample cap, attack-3 start distribution). **OBSERVED** —
  known assumption-bound departures, all recorded: primary batch 512 with the
  Keras-default 32 as a declared sensitivity (A18); a 15% benign validation
  carve-out that gives neural anomaly models 85% of B1 while classical anomaly
  models use all of B1; EarlyStopping restore-best-weights; SVM training caps
  (12k/30k); vectorized pooled ARIMA(1,1,0); four-GPU DDP execution semantics.
- **Remaining uncertainty / blast radius:** The separate ISET contract now
  contains `table_v_samples_per_class` and the ScienceDB allocation branch, and
  four-V100 DDP has completed end to end. Tables III--V still need their
  dataset/table-specific run metadata and aggregation path; neither gap changes
  the completed static audit.
- **Source artifacts:** `studies/atk-2022-deep-autoencoder/AMBIGUITY_REGISTER.md`
  (A27--A30), `EXPERIMENT_SPEC.md` non-blocking inconsistencies,
  `REPRODUCIBILITY_LOG.md` 2026-07-21 fidelity-audit entry.

### Paper 1 zero-trust fidelity correction

- **Former belief/status:** The 2026-07-21 audit labeled the complete
  implementation `VERIFIED` because each major choice appeared to match the
  paper or a registered branch.
- **Disconfirming evidence:** A fresh PDF-first pass on 2026-07-23, performed
  without treating the existing contract as authoritative, found concrete
  counterexamples. The built FC-SAE contains hidden widths
  `(400,300,200,100)|(200,300,400)`, i.e. seven hidden transformations, while
  Table I and Section IV-C describe four encoder plus four mirrored decoder
  hidden layers and Fig. 3/Section III-A place a distinct latent layer between
  them. FC-VAE similarly omits the fourth encoder, fourth decoder, and distinct
  placement implied by Fig. 4. A visual source recheck established a genuine
  recurrent ambiguity: the figures/prose describe a latent representation,
  while Algorithms 2 and 5 directly reuse the terminal encoder state or
  attention context. Implementation-v1 LSTM-SAE and LSTM-AEA therefore remain
  eligible only as those algorithm-literal branches, not as a unique reading.
  The
  ISET supervised cache constructs malicious data only from held-out B2
  customers, whereas the supervised paragraph explicitly requires benign and
  malicious classes for all customers. The VAE implementation scores
  deterministic reconstruction MSE and MSE+KL surrogates rather than the
  Monte Carlo Gaussian reconstruction probability stated in Section III-B.
  Additional blast-radius findings include an unstated 15% B1 validation
  carve-out used for neural but not classical final fits, resource-capped SVM
  rows, a 3,000-sample Table V cap, and a fixed-model Table V derivation that
  forces FA to be invariant despite the reported attack-dependent FA values.
  All 72 study tests and 10 project tests still pass because they validate the
  frozen implementation contract; one model test explicitly expects the
  three-layer FC decoder and therefore codifies rather than detects the
  layer-count mismatch.
- **Root cause:** **INFERRED** — the first audit checked the implementation
  against a contract that had already embedded reasonable resolutions and
  then treated “registered” as equivalent to “faithful.” It did not require a
  source-claim-to-runtime-layer/population trace or a row-level eligibility
  decision.
- **Current conclusion + label:** **INVALIDATED** — the former blanket
  `VERIFIED` fidelity conclusion is false. **OBSERVED** — FC-SAE, FC-VAE, the
  implementation-v1 ISET supervised cache, and the current Table V derivation
  contain material mismatches for their claimed paper rows. **OBSERVED** — VAE
  outputs are surrogate branches, not implementations of the stated anomaly
  score. LSTM-SAE and LSTM-AEA are quarantined as Algorithm-2/5 bottleneck
  branches, while LSTM-VAE matches the printed hidden-layer/distribution
  structure but still lacks a specified latent width and reconstruction-
  probability score. All old artifacts remain immutable; affected results are
  invalidated or quarantined by row.
- **Remaining uncertainty / blast radius:** Some failures arise from genuine
  paper contradictions or missing details rather than coding errors. They
  require explicit competing branches, not a single “corrected” pipeline.
  Exact data provenance and the metric arithmetic audit remain valid. No
  intent inference follows from this correction.
- **Source artifacts:**
  `studies/atk-2022-deep-autoencoder/PAPER_TO_CODE_TRACEABILITY.md`,
  `docs/plans/2026-07-23-paper-1-zero-trust-fidelity-audit.md`, PDF SHA-256
  `f3098e0c27ee19b27bea026aedc3d10e5dbb0c46f5cd01ed5bd5c05b7dcf850f`,
  and the Keras runtime layer inventory recorded in the audit session.

### Paper 1 independent source re-audit

- **Former belief/status:** The 2026-07-24 `METHOD.md` was treated as the
  complete source freeze and the project had already crossed its checkpoint.
- **Evidence:** On 2026-08-11 the exact 12-page PDF was independently
  fingerprinted (`f3098e…f850f`), text-extracted, and visually inspected page by
  page before the existing reconstruction was opened. Every reported table was
  re-transcribed and the prose, equations, algorithms, figures, and tables were
  reconciled. The prior reconstruction was largely accurate, but the fresh pass
  found an incorrect statement that Tables II/III had seven rather than six
  benchmark rows and several omitted source problems: Eq. (9)'s KL arguments
  and bound use incompatible variables; Algorithms 3/4 do not guarantee a
  positive Gaussian variance; Algorithm 5 defines a decoder input using the
  reconstruction before decoding; and the precision prose describes recall
  while its formula defines precision. A renewed calculation also confirmed
  that Table-II Naive Bayes `DR=PR=75` cannot yield `F1=77`, and that the
  DR/FA/PR rows in neither Table II nor Table III admit one common class
  prevalence even with ±0.5-percentage-point rounding allowance.
- **Root cause:** **INFERRED** — the July source freeze focused correctly on
  executable data/model ambiguities but did not make the paper's mathematical
  derivation and full reported-metric identities first-class source checks.
  It also described `P0` too loosely as preserving every printed step even
  though the non-executable Attack 3 necessarily uses a declared repair.
- **Current conclusion + label:** **VERIFIED** — the high-level paper flow,
  reported table transcriptions, primary model widths/settings, preprocessing
  order, and previously identified contradictions remain source-accurate.
  **VERIFIED CORRECTION** — `P0-ISET-FCSAE` is an executable paper-primary
  `P+I` completion, not a fully literal program; the unmodified printed Attack
  3 is retained as a non-executable result. **VERIFIED** — the added
  mathematical and reported-metric inconsistencies are internal properties of
  the source, not experimental non-reproduction evidence and not evidence of
  intent.
- **Remaining uncertainty / blast radius:** The five-file implementation has
  not yet been re-audited against the corrected 2026-08-11 source table. No
  earlier run becomes eligible merely because most of `METHOD.md` was
  confirmed. Existing code and experiments remain frozen until the renewed
  source-freeze checkpoint is accepted.
- **Source artifacts:**
  `studies/atk-2022-deep-autoencoder/METHOD.md`,
  `studies/atk-2022-deep-autoencoder/PAPER_WORKFLOW.md`, the local ignored PDF,
  and `studies/atk-2022-deep-autoencoder/results/reported_metrics_audit.{csv,json}`.

### Paper 1 exhaustive-interpretation and corrected-control mandate

- **Former belief/status:** The exploratory contract selected one primary
  author-intent interpretation per ambiguity and treated competing choices
  mainly as optional diagnostics.
- **Disconfirming evidence:** The 2026-07-23 fidelity correction showed that a
  documented primary choice can still encode the wrong experiment. The user
  explicitly required testing the printed procedure, every materially
  defensible interpretation of ambiguous or contradictory text, and the
  scientifically corrected procedure. ADASYN is the canonical example: the
  paper-printed test-set use must be run despite leakage, while a corrected
  training-only/no-test-resampling control must also be run.
- **Root cause:** **INFERRED** — selecting one reasonable interpretation
  prematurely collapses the uncertainty the reproduction is supposed to test.
  It also leaves an easy post hoc objection that a different textual reading
  might explain the published result.
- **Current conclusion + label:** **VERIFIED DECISION** — Paper 1 uses a frozen
  finite branch lattice with three separate families: printed (`P`), all
  materially defensible paper interpretations (`I`), and corrected controls
  (`C`). Externally motivated possibilities (`X`) are separately labeled and
  cannot count as paper reproduction. Every branch and exclusion is preserved;
  no best-branch reporting is allowed.
- **Current closure:** **VERIFIED STATIC AUDIT** — the machine-readable v1
  lattice contains 921 compatible paper-consistent semantic configurations
  across 22 model/data families (22 printed anchors plus 899 interpretive
  cases), plus
  22 separately identified corrected controls. It maps all 36
  ambiguity-register rows to executable dimensions, fixed behavior, global
  envelopes, non-executable nodes, locked dependencies, or timing records.
  Five executable tests verify stable IDs, complete compatible option/pair
  coverage, forbidden-pair exclusion, the row mapping, and corrected-control
  identity. The bounded three-seed screen is 2,763 attempts, estimated at
  558.7 GPU-hours and 57.4 CPU-hours; this is a planning result, not
  experimental evidence.
- **Threshold-lattice correction:** **INVALIDATED** — `dataset_specific` was
  formerly encoded as a threshold formula beside `printed_constant` and the
  three ROC/IQR repairs. The PDF makes dataset specificity a derivation scope,
  not an operation. The corrected contract independently crosses four
  threshold formulas with ISET-transferred versus dataset-specific scope and
  machine-excludes four impossible pairings: an ROC-derived formula with no
  validation labels, and a supplied printed constant with dataset-specific
  derivation. Constraint-aware pair coverage reduces the semantic count from
  942 to 921 without removing any executable textual interpretation.
- **Source correction during closure:** **OBSERVED** — Algorithm 6 requires
  staged depth/width, optimizer, dropout, and activation searches. Its scalar
  `N_l` loop implies 36 evaluations under a uniform-width reading, but that
  reading cannot directly yield Table I's unequal within-model widths. The
  lattice therefore retains literal 36-evaluation search, an 86-evaluation
  per-layer coordinate interpretation, and direct Table-I replay for Tables
  II-V. The former arbitrary eight-evaluation planning assumption was removed.
- **Architecture correction during closure:** **OBSERVED** — Table I and
  Section IV-C require every printed encoder hidden width and its full mirror;
  this independently invalidates the seven-hidden FC-SAE and six-hidden FC-VAE
  implementations. Figs./prose depict latent layers, but Algorithms 2/5 use
  terminal states or attention context directly. SAE/AEA therefore retain both
  distinct-projection and algorithm-bottleneck branches; VAEs retain explicit
  latent distributions. The omitted distinct latent width has the frozen
  `{2,8,16,32,48,100}` envelope.
- **Implementation state:** **VERIFIED STRUCTURAL** — an opt-in
  `paper_source_v2` builder now preserves implementation-v1 by default while
  constructing the full printed FC/recurrent mirrors, all six latent widths,
  both SAE/AEA latent-placement branches, and all registered dense/LSTM
  dropout placements. Four source-derived runtime-layer tests pass. The SGCC
  and ISET preparers now also execute every registered scaling reading,
  printed/no-test anomaly ADASYN, pre-split/training-only supervised ADASYN,
  and both ISET attack-population readings. Fixture tests verify all-customer
  attack cardinality, untouched corrected tests, and original-identity
  disjointness. Source-v2 now also implements both recurrent input layouts,
  both state-transfer policies, all SAE/VAE decoder schedules, both attention
  merges, fixed/learned-variance Monte Carlo Gaussian VAE probability with
  explicit score direction, and three deterministic repairs of the undefined
  ROC/IQR threshold rule. ISET now executes all registered Attack-1 scopes,
  Attack-2 granularities, Attack-3 minimal repairs, hour mappings, and
  all-4,225/seeded-3,000 residential populations. SGCC now executes all six
  frozen 1,034-versus-48 representations, all four missing-data readings, and
  customer-disjoint/row-random splits with source-customer provenance. ISET
  now executes all four 48-slot day policies and both customer/row split units,
  with heldout attacks joined by source-profile identity. A one-step
  learned-variance VAE fixture trains and scores finitely. This is a
  structural gate only; it is not a full-data result and no production cache
  was rebuilt. Algorithm 5's explicit reconstructed-value feedback now
  has a dedicated autoregressive AEA decoder across both merge, state, input,
  and latent readings; fixture attention weights normalize at every output
  step and a training update is finite. The visually checked
  Eq. (10) squared-L2 term now has a literal sum-squared branch and a separate
  common mean-MSE reading in both FC/LSTM VAE builders; deterministic fixtures
  prove their reconstruction terms differ by the feature count before KL.
  Learned-variance branches apply the analogous Gaussian-data-term reduction
  and remain labeled a prose-consistent completion. Direct Table V execution
  now covers all four model/benign identity readings and both evaluation
  sizes, persists every column-specific score/identity, and correctly applies
  lower-is-anomalous VAE probability orientation. All 16 frozen ARIMA
  order/pooling/score completions execute; capped versus full-data SVM fits are
  distinct; and binary versus benign-plus-six-attack SVM labels execute on
  ISET. The seven-class reading is excluded from SGCC because SGCC supplies no
  six attack identities.
- **Validation/threshold implementation:** **VERIFIED STRUCTURAL** — the
  ordinary runner now executes all frozen A09/A10/A25/A33 branches. B1
  validation attacks are generated deterministically with the six printed
  transforms in prepared model-input space; B2 validation is stratified and
  removed by exact row identity from final test; printed constants perform no
  label derivation. Threshold formula and derivation scope are independent,
  and SGCC ISET-transfer requires a frozen external artifact. Fixed epochs,
  holdout/no-refit, holdout/all-training refit, and
  five-fold-or-maximum-feasible cross-validation/all-training refit produce
  distinct fit topologies and preserve every fit history/timing. The
  `threshold_iqr_median` repair records a median-of-all-finite fallback only
  when interpolated quartiles contain no observed candidate. This is fixture
  evidence, not a model result. The DDP path now mirrors the ordinary
  validation/refit/threshold policies. Every frozen branch resolves by stable
  ID to preparation, model, classical, validation, threshold, and Table-V
  arguments, and source-v2 caches are keyed and checked by a
  content-addressed preparation ID. Corrected controls use a separate model
  contract and validation-selected Youden-J thresholds rather than being
  presented as another reading of the paper's ROC/IQR phrase.
- **Bounded execution gate:** **OBSERVED** — 137 study tests and 10 project
  tests pass. Twenty-seven neural/source tests execute all model families,
  frozen topologies, score directions, both supervised heads, and finite
  one-step updates in 3.855 test-runner seconds (7.72 s wall). Seventy-one
  data/classical/metric/ordinary/ISET/DDP tests pass in 4.595 test-runner
  seconds (7.16 s wall). The real SGCC printed FC-SAE anchor verifies the raw
  SHA-256 and prepares 42,367 retained customers in 6.806 s (9.55 s wall).
  The ISET printed anchor stops at missing cache
  `prep-20792e1602ac8e5d`, proving that the historical implementation-v1 cache
  is not silently accepted.
- **Remaining uncertainty / blast radius:** “All” is bounded to the exact PDF
  and a predeclared finite hyperparameter envelope; it is not a claim over
  infinitely many undisclosed programs. The 52.57-billion arbitrary Cartesian
  product is explicitly excluded because unrelated higher-order mixtures are
  not distinct source-grounded interpretations. The exact source-v2 ISET
  branch cache and one real multi-GPU DDP smoke remain required before scaled
  execution; local structural sanity cannot establish distributed numerical
  behavior or any reproduction result.
- **Source artifacts:**
  `docs/decisions/2026-07-23-exhaustive-interpretations-and-corrected-controls.md`,
  `studies/atk-2022-deep-autoencoder/BRANCH_COVERAGE_CONTRACT.md`,
  `studies/atk-2022-deep-autoencoder/config/branch_lattice.toml`,
  `studies/atk-2022-deep-autoencoder/results/branch_lattice_summary.json`, and
  the expanded `AMBIGUITY_REGISTER.md`,
  `studies/atk-2022-deep-autoencoder/PAPER_WORKFLOW.md`,
  `studies/atk-2022-deep-autoencoder/results/gate_d_bounded_sanity_20260724.json`,
  and `docs/decisions/2026-07-24-sgcc-multiclass-label-scope.md`.

### Paper 1 neural Table II exploratory cells and score mechanism

- **Former belief/status:** No neural Table II cell had a completed exact-SGCC
  outcome; whether the anomaly-side collapse seen in the classical rows would
  extend to the proposed autoencoders was open.
- **Supporting evidence:** the cluster DDP attempts completed 2026-07-21/22 with
  all internal finite/agreement gates passing. FC-SAE (seeds 11/22/33):
  DR 6.90%, FA 2.27%, AUC 51.90%, identical to two decimals across all three
  seeds, versus reported DR 83 / FA 14 / AUC 83. LSTM-VAE (seed 11):
  DR 7.39-7.45%, FA ~2.6%, AUC ~51.8-51.9% across both registered score
  branches, versus reported 93 / 6 / 90. Supervised feed-forward (three
  seeds): DR 83.79-88.56%, FA 10.41-21.05%, AUC 92.02-94.65%, versus reported
  91 / 9.5 / 89. LSTM-SAE seed 11 completed with 2:24:26 Slurm elapsed
  (2:24:03 recorded pipeline time; 2:20:38 fit):
  DR 6.78%, FA 2.22%, AUC 51.89%, versus reported 86 / 12 / 85. Using the
  labels to choose the best possible diagnostic threshold lifts balanced
  accuracy only from 52.28% to 55.52%; excluding paper-generated ADASYN test
  rows lifts AUC only to 59.73%.
- **Root cause:** **INFERRED for the completed SAE branch** — joint
  zero-mean/unit-variance preprocessing puts 71.60% of anomaly-test values
  below zero, while the paper-selected Softmax/sigmoid decoders cannot output
  negative values. LSTM-SAE reconstruction MSE correlates
  0.999999999999996 with mean squared standardized input (the score obtained
  by reconstructing zero), with mean absolute difference below `9e-7`.
  FC-SAE and LSTM-SAE seed-11 scores correlate 0.999999999985. Thus these
  architectures have converged to effectively the same input-energy score;
  the printed LSTM structure contributes no material separation in this
  literal branch. This mechanism has not yet been isolated by a controlled
  output-activation/scaling experiment.
- **Current conclusion + label:** **OBSERVED** — every completed neural
  anomaly-detector row is far from its reported Table II pattern, and the
  LSTM-SAE failure is not a fixed-threshold artifact. **OBSERVED** — the
  supervised feed-forward benchmark strongly separates the same prepared
  labeled data, so “the prepared data contain no learnable signal” is not a
  sufficient explanation. This materially strengthens the Paper 1 exploratory
  non-reproduction hypothesis.
- **Remaining uncertainty / blast radius:** LSTM-SAE has only one completed
  seed, so its seed stability is open; FC-SAE's three nearly identical seeds
  do rule out a best/lucky-seed explanation for that model. Ten other neural
  cells are unrun and three FC-VAE attempts failed. No paper-level or
  confirmatory verdict may be drawn from this partial exploratory table.
- **Source artifacts:** the cluster `data/derived/atk-2022-deep-autoencoder/runs`
  immutable attempts; job 354017; committed
  `results/table_2_neural_score_sanity.json`.

### Paper 1 first exact-ISET Tables III--V exploratory cells

- **Former belief/status:** The exact ISET data and all paper-positioned
  partitions were prepared, but there was no model result with which to test
  the reported Tables III--V values or the hypothesis that more training data
  would produce little benefit under the literal architecture.
- **Supporting evidence:** the cluster jobs 354932, 354933, and 354939 completed
  exact-ISET FC-SAE full, FC-SAE half, and FC-VAE full cells for seed 11. All
  immutable artifact hashes verify. FC-SAE full reproduced DR 22.50%, FA
  37.13%, balanced ACC 42.69%, and AUC 42.59% versus reported 81%, 15%, 83%,
  and 81%. Its half-data ACC was 42.68% versus reported 70%; the full-minus-half
  change was only 0.007 percentage points. FC-VAE reproduced DR 40.43%, FA
  53.86%, ACC 43.28%, and AUC 40.82% versus reported 88%, 11%, 88.5%, and 85%.
  Their Table V average DR/FA values were 22.04%/49.73% and 40.02%/66.03%,
  versus reported 81%/15.5% and 88%/10.5%.
- **Root cause:** **INFERRED for FC-SAE** — its reconstruction score correlates
  0.99945 with mean squared standardized input while 70.58% of ISET test
  values are negative and its paper-selected Softmax decoder is nonnegative
  and unit-sum. Attacks often have lower score/energy than benign rows. An
  oracle threshold in the printed direction reaches only 50.08% ACC; reversing
  the direction after seeing labels reaches 57.72% on paper-primary rows and
  64.36% on original-only rows, still below the reported 83%. **OPEN for
  FC-VAE** — its reconstruction and MSE-plus-KL surrogate results are
  numerically indistinguishable, but the precise latent/decoder failure
  mechanism has not yet been isolated.
- **Current conclusion + label:** **OBSERVED** — the first two exact-ISET
  architectures do not reproduce their Table III or V metric patterns in this
  frozen exploratory implementation. **OBSERVED** — FC-SAE's claimed Table IV
  accuracy increase from half to full training data is absent in seed 11.
  **VERIFIED structural consequence** — because Table V uses one fixed benign
  set, model, score direction, and threshold, its FA is identical across all
  six attacks; the paper's attack-varying FA cells require some unreported
  change in at least one of those elements.
- **Remaining uncertainty / blast radius:** These are one-seed results from two
  architectures. Recurrent/attention, supervised, classical, remaining seeds,
  and the two other Table IV sizes are still running or unrun. They strengthen
  the exploratory non-reproduction hypothesis but do not establish a
  confirmatory paper-level verdict or intent.
- **Source artifacts:** exact cache SHA-256
  `ab88f180feafb7351ef4530cba2e48a3cbc180af268f8b68016aefc50b98a987`;
  the cluster jobs 354932, 354933, 354939; immutable `table_3/iset` and
  `table_4/iset/half` attempts; committed
  `results/iset_score_sanity_seed11.json`.

### Paper 1 FC-SAE output-activation control

- **Former belief/status:** The near-zero-reconstruction Softmax anchor left
  open whether incompatibility between standardized negative targets and a
  nonnegative unit-sum decoder was sufficient to explain the large Table III
  gap.
- **Supporting evidence:** Panther job 373805 changed only the final FC-SAE
  activation from Softmax to linear while preserving data hash, architecture,
  seed 11, batch 512, optimizer, training rule, threshold, score, and test
  population. It completed 100 epochs in 1:05:43. Reproduced
  DR/FA/ACC/AUC/F1 were 12.32/30.78/40.77/28.14/21.61%, versus the paper's
  81/15/83/81/81%. Audit job 373824 found a 50.04% best possible balanced ACC
  in the printed score direction and 67.56% after reversing direction. Mean
  error was 0.537 for benign and 0.281 for malicious profiles. Score
  correlation with zero reconstruction fell from the Softmax anchor's 0.99946
  to 0.82089. Table-V FA was exactly 30.0696% across all six attacks on its
  common all-benign evaluation population.
- **Root cause:** **OBSERVED bounded contrast** — the linear decoder changes
  the learned score substantially and reduces fixed-threshold FA, but the
  malicious/benign score ordering remains opposite the printed rule and
  discrimination remains far below the reported result.
- **Current conclusion + label:** **OBSERVED** — output activation alone is not
  a sufficient explanation for the frozen baseline's non-reproduction. The
  result strengthens the need to locate divergence in shared data/evaluation
  choices or other model details before repeating seeds.
- **Remaining uncertainty / blast radius:** This is one seed of a corrected
  control and does not test the printed ADASYN operation, joint changes to
  scaling and output, other architectures, or benchmark models. It is not a
  paper-level verdict and says nothing about author intent.
- **Source artifacts:** cluster jobs 373805 and 373824; immutable attempt
  `seed_11_b4375c29b822_table_v`; committed
  `results/iset_fc_sae_linear_seed11_score_audit_20260811.json`.

### FC-VAE DDP post-optimizer gate failure

- **Former belief/status:** The production DDP path was expected to execute all
  five autoencoders after fc_sae, lstm_vae, and supervised_feed_forward
  completed through it.
- **Disconfirming evidence:** All three fc_vae seeds fail with
  `FloatingPointError: distributed post-optimizer parameter/state gate failed
  on at least one rank (code=2)` (attempts of 2026-07-21). VAE+SGD (lstm_vae)
  completed; non-VAE Adam models (fc_sae, supervised benchmarks) completed.
- **Root cause:** **OPEN** — job 354018 showed that the first DDP Adam update is
  finite on all four ranks, so the original failure occurs later. Rank-local
  first-step losses ranged from roughly 42 to 42 million and the largest
  gradient magnitude was about 238 million on `z_log_var/kernel`, identifying
  extreme rank/input sensitivity without yet isolating the later nonfinite
  state.
- **Current conclusion + label:** **OBSERVED** — fc_vae cannot currently
  produce Table II evidence on the DDP path; failures are preserved as
  outcomes and no batch/model reduction was silently applied.
- **Remaining uncertainty / blast radius:** Whether later instability is in
  DDP synchronization, Adam state accumulation, the VAE loss scale, or their
  interaction; FC-VAE rows stay empty until a discriminating later-step
  diagnostic or a procedure-equivalent stable run resolves it.
- **Source artifacts:** FC-VAE attempt manifests of 2026-07-21, job 354018,
  `scripts/cluster/diagnose_fc_vae_first_step.sbatch`.

### Single-T4 batch-32 LSTM-SAE runtime

- **Former belief/status:** The declared batch-32 sensitivity might complete
  acceptably on one available 16-GB T4 under the two-day Slurm limit.
- **Disconfirming evidence:** the cluster job 348223 ran the full 1,034-step
  LSTM-SAE, seed 11, with `exploratory_batch32.toml` for 23:27:30 without
  producing a completed attempt. Its non-verbose Keras fit made the Slurm log
  silent during training. The user explicitly directed cancellation on
  2026-07-22; `sacct` recorded `CANCELLED` and it disappeared from `squeue`.
- **Root cause:** **INFERRED** — batch 32 requires roughly sixteen times as many
  optimizer steps per epoch as global batch 512, on one slower T4 rather than
  the four-V100 primary path, while retaining the full recurrent sequence and
  widths.
- **Current conclusion + label:** **OBSERVED** — the single-T4 batch-32
  sensitivity is operationally impractical under the current branch. The
  interrupted job is resource/runtime evidence only, not a metric outcome or
  evidence that the model failed statistically.
- **Remaining uncertainty / blast radius:** It is unknown how many epochs had
  completed because progress was not persisted per epoch. This does not affect
  the primary four-V100 batch-512 job 354017 or any completed Table II cell.
- **Source artifacts:** the cluster job 348223 accounting record and
  `scripts/cluster/run_model.sbatch`.

### Experiment-first workflow correction

- **Former belief/status:** Rigor was pursued by implementing exhaustive
  ambiguity coverage, immutable evidence machinery, distributed execution, and
  publication scaffolding before a compact source-faithful route had produced
  an eligible full result.
- **Disconfirming evidence:** The Paper 1 internal Python tree reached 21,414
  lines including tests; several long-running results were later invalidated or
  quarantined after a fresh PDF read found architecture, population, and score
  mismatches. The user correctly observed that engineering and documentation
  time had overtaken eligible experimental runtime.
- **Root cause:** **INFERRED** — implementation began before a complete
  PDF-derived method freeze, and later audits checked code against contracts
  that already embedded assumptions. When mismatches appeared, new framework
  layers were added instead of first rebuilding one small straight-through
  experiment.
- **Current conclusion + label:** **VERIFIED DECISION** — `RUNBOOK.md` governs
  an experiment-first sequence: PDF-derived `METHOD.md`, exact data, five real
  scientific files, tiny sanity checks, one eligible full anchor, remaining
  tables/seeds, material interpretation branches, confirmation, and only then
  reporting/publication. The existing branch engine is preserved as a forensic
  coverage tool but removed from the critical path.
- **Remaining uncertainty / blast radius:** The compact Paper 1 route is not yet
  implemented and no historical artifact is upgraded by this decision. The
  first discriminating milestone is a fresh eligible full ISET Table III
  FC-SAE result with complete timing.
- **Source artifacts:** `RUNBOOK.md`,
  `docs/plans/2026-07-24-paper-1-minimal-reimplementation.md`, and
  `docs/decisions/2026-07-24-experiment-first-paper-workflow.md`.

### Full compact ISET test-set ADASYN executability

- **Former belief/status:** Full compact preparation was expected to finish
  the paper-printed test-set ADASYN as a routine final preprocessing step.
- **Evidence:** The source-derived route first materialized 2,251,290 strict
  benign profiles, 13,507,740 attack profiles, 1,500,520 B1 rows, 750,770 B2
  rows, and the exact 14,258,510-row `B2+M` population. With 48 features,
  sklearn's default nearest-neighbor selector chooses brute force.
  imbalanced-learn ADASYN's first full call therefore queries 750,770 minority
  rows against all 14,258,510 rows: about 10.7 trillion profile-distance
  comparisons. It then performs another minority-only neighbor search. The
  full attempt was interrupted after 4,724.52 seconds wall and 33,665.16
  CPU-seconds inside this first call without creating the resampled array.
  Panther job 373799 then measured 250 exact 48-feature queries against all
  14,258,510 references on 16 CPU cores: 3,564,627,500 distance pairs in
  16.1217 seconds. Linear extrapolation gives 13.4485 hours for the first search
  and 0.7081 hours for the minority-only search, or 14.1566 wall-hours before
  synthesis, allocation, serialization, and retries.
- **Root cause:** **VERIFIED algorithmically; OBSERVED operationally** —
  ADASYN is printed after construction of an unusually large all-customer
  malicious test population, and the frozen executable completion selected
  imbalanced-learn defaults. Those defaults are exact and expensive at this
  cardinality/dimension. This is not a model-training failure.
- **Current conclusion + label:** **OBSERVED** — exact pre-ADASYN Paper-1 data
  construction is complete and auditable. **OPEN** — the full default-ADASYN
  result has not completed and must not be silently represented as P0.
  The exact `B2+M` no-resampling interpretation is run separately as
  `I-ADASYN-NONE`; a scalable approximate-neighbor sensitivity will also be
  labeled separately. **OBSERVED/BOUNDARY** — 14.16 hours of neighbor search is
  compatible with an overnight preprocessing job. The paper omits its library,
  search method, hardware, and timing boundary, so this benchmark does not
  support a claim that the authors could not have run ADASYN.
- **Remaining uncertainty / blast radius:** ADASYN does not alter B1 or model
  fitting, nor malicious-class DR on the original rows. It can alter FA by
  adding synthetic benign rows. Thus the model may be trained and its
  original-population DR/FA/AUC measured while P0's synthetic-benign FA
  remains unresolved.
- **Source artifacts:** fresh
  `data/derived/atk-2022-deep-autoencoder/reproduction/p0-full` cache (ignored),
  `METHOD.md`, compact `prepare_data.py`, the full preparation attempt of
  2026-07-24, and
  `studies/atk-2022-deep-autoencoder/results/adasyn_default_runtime_estimate_20260811.json`.

### Local-execution policy correction

- **Former belief/status:** When the cluster appeared unreachable, the compact
  FC-SAE seed-11 anchor was started on the local Mac as a procedural fallback.
- **Corrective evidence:** The user explicitly reconfirmed that experimental
  computation for this project must never run locally. The the institution VPN tunnel was
  in fact active on `utun4`; the first SSH command remained inside the command
  sandbox and needed an explicitly escalated retry. Full Access had remained
  enabled throughout.
- **Current conclusion + label:** **INVALIDATED** — local fallback execution
  is not authorized and cannot produce eligible project evidence. The FC-SAE
  attempt was interrupted after ten epochs and 3,204.82 seconds wall. It
  produced only a configuration stub, no weights, scores, or result record.
  **VERIFIED DECISION** — all preparation, training, and scoring jobs run on
  the cluster compute nodes; local work is limited to code, documentation,
  lightweight inspection, and transfer/monitoring.
- **Remaining uncertainty / blast radius:** No metric or model artifact from
  the interrupted attempt exists, so no reported numerical conclusion changes.
  The the cluster seed-11 attempt must start fresh from the pushed commit and
  verified data hashes.

### Compact batch-512 ISET FC-SAE baseline

- **Former belief/status:** The renewed five-file implementation had one
  batch-32 sensitivity far from the paper, but the frozen batch-512 primary
  execution completion had not run. It remained possible that the earlier
  result was materially caused by the different batch size or incomplete
  execution contract.
- **Evidence:** Panther job 373789 ran commit `c8c136f` on one 16-GB V100 and
  completed successfully in 53:12. Source verification and exact preparation
  yielded 2,251,290 benign profiles, 1,500,520 B1 profiles, 750,770 B2 profiles,
  and 13,507,740 malicious profiles. The 450,448-parameter FC-SAE trained for
  74 epochs, restoring epoch 69. It reproduced DR 26.18%, FA 58.22%, balanced
  ACC 33.98%, AUC 31.04%, and F1 40.46%, versus reported 81%, 15%, 83%, 81%,
  and 81%. Table-V FA was invariant at 57.9152% for all six attacks. These
  values closely repeat the earlier batch-32 sensitivity
  (26.44/58.51/33.97/31.03/40.78%).
- **Root cause:** **OPEN for the full numerical gap. VERIFIED arithmetic for
  the ADASYN boundary** — printed ISET ADASYN oversamples benign test rows and
  therefore cannot alter this trained model's predictions or DR on the
  preserved malicious rows. With DR fixed at 26.18%, even a hypothetical
  perfect FA of 0% would yield only 63.09% balanced ACC, below the reported
  83%. ADASYN can still change FA, AUC, precision, and other distribution-
  dependent metrics, so this no-resampling run is not the completed printed
  evaluation.
- **Score-audit evidence:** Panther job 373800 reloaded the saved attempt. An
  oracle threshold chosen on the test labels reached only 50.0019% balanced ACC
  in the printed high-error direction. Reversing the direction reached 66.2554%
  ACC and 68.9561% AUC, still far from the reported row. The trained
  reconstruction-error vector correlated 0.999459 with the zero-reconstruction
  score vector. **INFERRED with strong mechanistic support** — ranking is
  dominated by standardized input energy. This is consistent with the printed
  combination of negative standardized inputs and a nonnegative sum-to-one
  Softmax reconstruction layer; the latter cannot reconstruct the former.
- **Current conclusion + label:** **OBSERVED** — changing from batch 32 to the
  frozen batch 512 does not rescue the seed-11 result, and the completed compact
  anchor is far from the reported Table-III and Table-V patterns. **VERIFIED
  structural consequence** — a common model, common benign set, and fixed
  threshold require common Table-V FA, contrary to the attack-varying values
  printed by the paper.
- **Remaining uncertainty / blast radius:** This is one primary seed and an
  explicitly no-resampling interpretation, not confirmatory `P0` and not a
  paper-level verdict. A scalable separately labeled ADASYN implementation may
  resolve the remaining benign-side metrics but cannot repair this model's
  observed DR. Seeds 22 and 33 remain; a linear reconstruction output is the
  first separately labeled structural repair and must not replace the Softmax
  baseline.
- **Source artifacts:** Panther jobs 373789 and 373800,
  `studies/atk-2022-deep-autoencoder/results/compact_route_fc_sae_seed11_batch512_20260811.json`, and
  `studies/atk-2022-deep-autoencoder/results/iset_fc_sae_seed11_score_audit_20260811.json`.

### Compact ISET Naive Bayes benchmark breadth row

- **Former belief/status:** No preserved ISET/Table-III benchmark attempt met
  the renewed source, population, and provenance gates. The cheaper benchmark
  rows were needed before spending additional depth on proposed models.
- **Evidence:** Panther job 373833 ran the complete all-customer original
  `B+M` population (15,759,030 profiles), an exact seed-11 row-random 2:1
  split, Gaussian Naive Bayes with `var_smoothing=1e-9`, and a 0.5
  positive-probability threshold. It completed in 1m36s (41.30s inside the
  runner). Reproduced DR/FA/ACC/F1/AUC were
  88.78/44.53/72.12/90.50/79.17%, versus reported
  73/18/77.5/73/70%.
- **Root cause:** **OPEN** — the paper does not identify the Naive Bayes
  variant and requires ADASYN before the supervised split. This first
  completion deliberately preserves original rows and therefore cannot isolate
  model choice from resampling.
- **Current conclusion + label:** **OBSERVED** — this explicit Gaussian-NB,
  no-supervised-ADASYN completion does not reproduce the reported Table-III
  metric pattern. Its fixed operating point has both much higher DR and much
  higher FA than reported. It is breadth evidence, not the printed branch or a
  paper-level verdict.
- **Remaining uncertainty / blast radius:** A paper-consistent ADASYN
  completion and other unspecified Naive Bayes variants remain outside this
  single attempt. Repeated seeds wait until the breadth map is complete.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/iset_naive_bayes_seed11_20260811.json`.

### Compact ISET ARIMA benchmark breadth row

- **Former belief/status:** The ISET ARIMA row had no renewed compact-route
  result; the paper fixes only `d=1` and `q=0` and leaves the autoregressive
  order, fit unit, and anomaly score unspecified.
- **Evidence:** Panther job 373836 ran the predeclared smallest completion:
  pooled ARIMA(1,1,0)-style OLS over first-difference transitions, residual MSE
  per 48-slot profile, threshold 0.58, all 1,500,520 B1 rows, and all
  14,258,510 original B2+M rows. It completed in 1m02s. Reproduced
  DR/FA/ACC/F1/AUC were 21.48/57.20/32.14/34.46/24.72%, versus reported
  86/12/87/86/87%.
- **Root cause:** **OPEN** — this score ranks the selected attacked profiles
  below benign profiles (AUC 24.72%), but the missing ARIMA definitions prevent
  attributing the paper-level gap to one unique implementation.
- **Current conclusion + label:** **OBSERVED** — the smallest registered pooled
  ARIMA completion does not reproduce the Table-III row and is qualitatively
  opposite its reported separation.
- **Remaining uncertainty / blast radius:** Other predeclared `p`, per-profile
  fit, and likelihood-score interpretations remain unexecuted. No intent or
  infinite-space claim follows.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/iset_arima_p110_pooled_mse_seed11_20260811.json`.

### Compact ISET one-class SVM benchmark breadth row

- **Former belief/status:** The paper's one-class SVM row could not execute
  directly because its kernel/Gamma wording is not a valid API pair, and a
  full kernel fit/score is resource-prohibitive at the paper-derived scale.
- **Evidence:** Panther job 373837 ran the registered repair
  `kernel=sigmoid, gamma=scale, nu=0.5`, trained on a deterministic 12,000-row
  B1 cap, and evaluated a deterministic 30,000-row original B2+M cap. It used
  6,003 support vectors and completed in 1m04s. Reproduced
  DR/FA/ACC/F1/AUC were 91.87/50.94/70.47/94.35/79.67%, versus reported
  90/9/90.5/89.5/87%.
- **Root cause:** **OPEN** — this repaired fixed threshold labels about half of
  benign profiles anomalous. The source's invalid SVM wording and full-scale
  kernel cost prevent treating the bounded result as a unique literal row.
- **Current conclusion + label:** **OBSERVED** — detection rate alone is near
  the paper value, but the reported low-FA operating point is not reproduced;
  the full metric pattern is a non-match for this bounded completion.
- **Remaining uncertainty / blast radius:** Training and test caps are explicit
  resource assumptions. Full-data or other predeclared SVM interpretations
  remain outside this diagnostic.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/iset_one_class_svm_seed11_20260811.json`.

## How to add a learning

Use: former belief/status; evidence; root cause if isolated; current conclusion
with label; remaining uncertainty; and source artifacts. Preserve invalidated
beliefs rather than deleting them.
