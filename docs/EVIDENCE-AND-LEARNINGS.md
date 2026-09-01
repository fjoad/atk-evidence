# ATK Evidence — Evidence and Causal Learnings

**Last updated:** 2026-08-24

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
- **Disconfirming or supporting evidence:** SGCC was acquired from the author-linked repository and checksum-verified. Official ISSDA metadata marks the CER files restricted. A 2024 ScienceDB deposit by Zehao Song (DOI `10.57760/sciencedb.17619`) exposes `File1.txt.zip` through `File6.txt.zip`; every displayed filename, byte size, and MD5 matches the official ISSDA manifest, and anonymous one-byte range requests to all six download endpoints returned the corresponding full object sizes. The deposit also supplies a converted allocation CSV. A public GitHub workbook independently matches all 6,445 rows across the five semantic allocation columns after the declared blank/zero normalization. On 2026-08-25, official Dataverse file-ID-808 metadata confirmed that the frozen 196,316-byte `.tab` is an archival ingest of an originally uploaded 185,480-byte XLSX.
- **Root cause:** Official access remains approval-gated, but third parties publicly deposited byte-identical consumption archives and two semantically agreeing allocation representations. The later phrase “missing exact `.tab` data” conflated missing Dataverse serialization bytes with missing allocation information.
- **Current conclusion + label:** **VERIFIED** — all six exact archives are local, match the official size/MD5 values, and pass ZIP integrity. **VERIFIED for the named semantic branch** — the allocation CSV contains 6,445 unique assignments, matches a second public allocation workbook across all five semantic columns, and covers every residential reading meter. **VERIFIED DISTINCTION** — the allocation data are available; only the byte-identical restricted Dataverse `.tab` serialization is absent.
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

### Completed ISET benchmark score-vector audit

- **Former belief/status:** Fixed-threshold non-matches were known for Naive
  Bayes, pooled ARIMA, and capped one-class SVM, but it was unclear whether each
  was merely a poor threshold or a deeper ranking failure.
- **Evidence:** Panther audit jobs 373854 and 373855 loaded the preserved score
  vectors and computed paper-direction oracle thresholds, reversed-direction
  controls, closest ROC points to the paper's DR/FA pair, score distributions,
  and attack-specific DR. Naive Bayes reaches at most 74.74% oracle balanced
  ACC; its closest point to reported DR=73/FA=18 is DR=71.82/FA=23.00, a 5.00
  percentage-point maximum gap. Pooled ARIMA reaches only 50.00% oracle ACC in
  the paper direction; reversed direction reaches 69.74%. Its closest
  paper-direction point is DR=29.44/FA=68.56, 56.56 points from the target.
  Benign mean residual MSE is 1.143 versus 0.439 for malicious rows, and attack
  4 has 0% DR: profile-flattening attacks make this high-residual anomaly rule
  point backward. Capped one-class SVM reaches 73.86% oracle ACC; its closest
  target point is DR=71.76/FA=27.31, still 18.31 points away.
- **Root cause:** **PARTLY INFERRED.** The pooled ARIMA failure follows
  mechanistically from using residual magnitude against attacks that often
  flatten or suppress variation. Naive Bayes instead has moderately useful
  ranking and is sensitive to its unspecified model/decision completion.
  One-class SVM separates scores somewhat but not enough to reach the claimed
  high-DR/low-FA corner in its bounded branch.
- **Current conclusion + label:** **OBSERVED** — threshold adjustment alone
  cannot rescue pooled ARIMA or the capped one-class SVM to their reported
  Table-III operating points. **OBSERVED/OPEN** — Naive Bayes is a weaker
  non-match: its omitted decision procedure could materially change the row,
  although this completion still misses the printed target.
- **Statistical boundary:** The millions of profile rows are not independent.
  Days share meters and the six malicious siblings are transforms of the same
  benign profile. Treating rows as iid would manufacture tiny confidence
  intervals through pseudo-replication. Confirmatory uncertainty must combine
  repeated training seeds with meter-level clustered resampling or aggregation.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/iset_benchmark_breadth_score_audit_20260811.json`.

### Compact ISET supervised feed-forward benchmark breadth row

- **Former belief/status:** The completed anomaly and classical rows did not
  establish whether the exact ISET attack population itself contains an easily
  learnable supervised signal under the paper's named deep benchmark.
- **Evidence:** Panther job 373838 used all 15,759,030 original `B+M` profiles,
  a seed-11 exact row-random 2:1 split, five 500-unit ReLU hidden layers,
  Adamax, and the predeclared two-class Softmax/categorical completion. The
  paper-positioned supervised ADASYN was explicitly omitted. The job completed
  in 1:29:56. At the ordinary 0.5 probability cutoff it reproduced
  DR/FA/ACC/F1/AUC = 96.41/23.72/86.35/96.24/97.05%, versus reported
  90/11/89.5/89.5/88%. Panther audit job 374255 found a best balanced
  threshold ACC of 91.66%; threshold 0.824 gives DR=91.83% and FA=9.17%, only
  1.83 percentage points from both printed targets.
- **Root cause:** **INFERRED** — the fixed-row difference is primarily an
  operating-point choice, not inadequate ranking. The paper does not specify
  how supervised probabilities are thresholded or whether their cutoff was
  selected on validation data.
- **Current conclusion + label:** **OBSERVED** — this completion learns stronger
  class separation than the paper reports and can closely approach its DR/FA
  pair through threshold selection. The supervised feed-forward row is
  technically plausible and is a positive control against the claim that the
  prepared attacks contain no learnable signal.
- **Remaining uncertainty / blast radius:** The threshold was selected
  retrospectively on the test ROC and is diagnostic, not an eligible
  reproduction procedure. Pre-split ADASYN remains omitted, the paper's
  head/loss and threshold rule remain unstated, and this is one seed. It does
  not validate any proposed autoencoder row.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/iset_supervised_feed_forward_seed11_20260811.json`.

### Compact ISET multiclass SVM benchmark breadth row

- **Former belief/status:** The final named benchmark row had not been tested;
  the source's kernel/Gamma wording and multiclass label scope both required an
  executable completion.
- **Evidence:** Panther job 373840 used the registered seven-class
  `kernel=sigmoid, gamma=scale, C=1` repair and deterministic 30,000-row
  train/test caps. It completed in 2m27s with 23,234 support vectors. Fixed
  DR/FA/ACC/F1/AUC = 85.94/55.67/65.14/88.04/73.06%, versus reported
  91/8/91.5/90.5/89%. Audit job 374302 gives best balanced ACC 71.14%; the
  closest threshold has DR=67.62/FA=31.44, a 23.44-point maximum target gap.
- **Root cause:** **OPEN** — this completion separates the classes somewhat but
  cannot produce the claimed high-detection/low-FA corner. Source ambiguity,
  omitted ADASYN, and deterministic caps remain material.
- **Current conclusion + label:** **OBSERVED** — threshold adjustment does not
  rescue this bounded multiclass-SVM completion to the reported row.
- **Remaining uncertainty / blast radius:** This is not an uncapped full cell
  and does not cover every repair of the malformed SVM prose.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/iset_multiclass_svm_seed11_20260811.json`.

### Compact ISET FC-VAE proposed-model breadth row

- **Former belief/status:** The paper names reconstruction probability but
  omits the variance, sampling, aggregation, scale, and a valid executable
  variance parameterization; the first predeclared completion was untested.
- **Evidence:** Panther job 373842 executed the Table-I FC-VAE with latent width
  100, mean-MSE plus analytic-KL training, and deterministic fixed-unit score
  `exp(-0.5 * profile MSE)`. It restored epoch-2 weights after seven epochs and
  completed in 6m14s. Fixed DR/FA/ACC/F1/AUC =
  11.51/32.62/39.45/20.32/30.13%, versus reported
  88/11/88.5/88.5/85%. Audit job 374303 gives only 50.00% oracle ACC in the
  paper's low-probability direction and 66.70% after reversing direction.
  Malicious mean probability is 0.750 versus benign 0.567; the trained score
  correlates 0.99957 with the zero-reconstruction score.
- **Root cause:** **INFERRED for this completion** — standardized attacked rows
  often have lower reconstruction error and therefore higher probability than
  benign rows, opposite the paper's VAE direction; the trained reconstruction
  again adds little beyond an input-energy/zero-output control.
- **Current conclusion + label:** **OBSERVED** — this registered FC-VAE
  completion fundamentally fails in the paper's score direction; no threshold
  approaches the reported point.
- **Remaining uncertainty / blast radius:** The source does not uniquely define
  reconstruction probability, so materially distinct predeclared probability
  completions remain open. This result cannot close an infinite VAE space.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/iset_fc_vae_seed11_20260811.json`.

### Recurrent breadth operational failures and bounded repair

- **Former belief/status:** Jobs 373839/373841/373843/373844 were expected to
  yield scientific breadth rows from the tested compact route.
- **Evidence:** Supervised-LSTM job 373839 completed training and saved weights
  but its 8,192-row inference batch exhausted the 16-GB V100. The other three
  jobs stopped before training because diagnostic inventory called `.shape` on
  a recurrent layer's list of output tensors. The exact failures and artifacts
  are preserved. Commit `c735dd9` records a list of output shapes and sets
  recurrent inference batch 512. Focused compact tests pass. Replacement jobs
  374310--374313 are queued from that commit.
- **Root cause:** **VERIFIED** — both are operational harness defects outside
  model mathematics. Scoring batch size changes only memory partitioning;
  inventory serialization changes only diagnostics. The replacement rows later
  completed and are recorded separately below.
- **Current conclusion + label:** **INVALIDATED as scientific failures** — none
  of these four failed attempts says whether the corresponding paper result is
  reproducible. **VERIFIED** — the smallest repairs leave data, architecture,
  optimizer, training batch, score definition, and metrics unchanged.
- **Remaining uncertainty / blast radius:** Four initial replacement
  submissions 374306--374309 were cancelled while pending after a stale
  upstream-branch configuration was discovered; no stale-code job executed.
  The operational question is now resolved by the audited replacement rows;
  their scientific uncertainty is recorded in their own sections.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/recurrent_breadth_operational_failures_20260811.json`.

### Compact ISET supervised-LSTM benchmark breadth row

- **Former belief/status:** The first attempt trained but its oversized scoring
  batch failed, so no valid model result existed.
- **Evidence:** Scientifically unchanged replacement job 374310 used recurrent
  score batch 512, completed six epochs in 6:54:51, and restored epoch-1
  weights. Every saved test probability is exactly 1.0. Fixed
  DR/FA/ACC/F1/AUC = 100/100/50/92.32/50%, versus reported
  90.5/10/90/90/89%. Audit job 374387 gives oracle ACC 50% in either score
  direction and a minimum 90-point joint DR/FA gap.
- **Root cause:** **OBSERVED, mechanism not yet isolated** — the registered
  four-by-300 ReLU-LSTM, sigmoid/BCE completion saturates to a constant positive
  prediction after its first epoch and never recovers.
- **Current conclusion + label:** **OBSERVED** — this completion fundamentally
  fails to rank benign and malicious profiles; threshold selection cannot
  reproduce the paper row.
- **Remaining uncertainty / blast radius:** This is one seed and omits the
  paper-positioned supervised ADASYN. The paper also omits the supervised head,
  loss, threshold rule, and training defaults, so materially distinct
  completions remain open.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/iset_supervised_lstm_seed11_20260812.json`.

### Second recurrent pre-training batch leak

- **Former belief/status:** Commit `c735dd9` bounded recurrent scoring at 512,
  so replacement proposed-model jobs were expected to reach training.
- **Evidence:** Jobs 374311--374313 failed before training because the separate
  untrained sanity probe still passed all 10,000 sampled rows through each
  recurrent model at once. Commit `4469a53` chunks the identical diagnostic by
  the already-recorded `score_batch`; a regression test verifies the maximum
  call size. Jobs 374388--374390 were subsequently rejected before execution
  by the immutable-attempt guard because they retained the prior score-batch
  identity. Job 374391 used the numerically equivalent, lower-memory,
  explicitly recorded score batch 256 and completed. Dependent jobs
  374392--374393 were cancelled before execution and replaced by independent
  jobs 374395--374396, which also completed.
- **Root cause:** **VERIFIED** — a second inference path bypassed the common
  batch-scoring loop.
- **Current conclusion + label:** **INVALIDATED as scientific failures** — the
  three attempts contain no evidence about model performance. The repair
  changes memory partitioning only, not values or scientific settings.
- **Remaining uncertainty / blast radius:** Proposed recurrent breadth is now
  closed for the registered completions; the scientific limitations are
  recorded in the LSTM-SAE, LSTM-VAE, and LSTM-AEA sections below.

### Compact ISET LSTM-SAE breadth row

- **Former belief/status:** The registered Algorithm-2 completion had not
  completed a full train-and-score run.
- **Evidence:** Panther job 374391 restored epoch-20 weights after 25 epochs and
  produced fixed DR/FA/ACC/F1/AUC =
  14.78/40.96/36.91/25.25/33.09%, versus reported
  85/13/86/85/82%. Audit 374433 gives paper-direction oracle ACC 50.004%,
  reversed-direction ACC 64.38%, and a 47.11-point minimum joint DR/FA gap.
  Benign mean MSE is 1.087 versus malicious 0.519; trained scores correlate
  0.97495 with the zero-reconstruction control.
- **Root cause:** **INFERRED for this completion** — malicious transforms are
  generally lower-error than benign rows under the learned reconstruction, so
  the score order opposes the paper's higher-error-is-anomalous rule and remains
  largely driven by input energy.
- **Current conclusion + label:** **OBSERVED** — no paper-direction threshold
  recovers the reported operating point for this one-seed completion.
- **Remaining uncertainty / blast radius:** Test-set ADASYN is omitted, and the
  source leaves materially different decoder-input completions open. This is
  not a conclusion over those branches or repeated seeds.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/iset_lstm_sae_seed11_20260812.json`.

### Compact ISET LSTM-VAE breadth row

- **Former belief/status:** The registered Algorithm-4/fixed-unit-probability
  completion had trained, but GPU memory exhaustion during scoring left no
  scientific result.
- **Evidence:** Panther job 374395 trained 23 epochs and preserved epoch-18
  weights. No-gradient recovery job 374441 loaded those exact weights in a
  fresh process and scored the original 14,258,510 rows without retraining.
  Fixed DR/FA/ACC/F1/AUC = 10.02/25.79/42.11/17.98/29.83%, versus reported
  91/7/92/91/86%. Audit 378014 gives paper-direction oracle ACC 50.002%,
  reversed-direction ACC 66.93%, and a 58.48-point minimum joint DR/FA gap.
  Malicious mean reconstruction probability is 0.783 versus benign 0.630 even
  though the paper declares lower probability anomalous. Trained scores
  correlate 0.93379 with the zero-reconstruction control.
- **Root cause:** **INFERRED for this completion** — the selected probability
  transformation preserves an MSE ordering in which attacks generally appear
  more probable than benign profiles, opposite the paper's decision rule; the
  reconstruction remains strongly influenced by input energy.
- **Current conclusion + label:** **OBSERVED** — threshold selection cannot
  recover the reported operating point for this registered completion.
- **Remaining uncertainty / blast radius:** Test-set ADASYN is omitted. The
  source does not uniquely define reconstruction probability, variance,
  sampling/aggregation, or decoder input, and this is one seed. Materially
  distinct predeclared completions remain open.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/iset_lstm_vae_seed11_20260818.json`.

### Compact ISET LSTM-AEA breadth row

- **Former belief/status:** The registered Algorithm-5 attention completion was
  still training, so its Table-III row and score direction were unresolved.
- **Evidence:** Panther job 374396 completed all 100 epochs in 43:41:50; its
  lowest training loss occurred at epoch 100. Fixed DR/FA/ACC/F1/AUC =
  25.43/58.22/33.60/39.53/29.93%, versus reported
  94/5/94.5/93.5/90%. Audit 378015 gives paper-direction oracle ACC 50.002%,
  reversed-direction ACC 66.52%, and a 60.11-point minimum joint DR/FA gap.
  Benign mean MSE is 1.286 versus malicious 0.645 even though the paper
  declares higher error anomalous. Trained scores correlate 0.97843 with the
  zero-reconstruction control.
- **Root cause:** **INFERRED for this completion** — attacks generally receive
  less reconstruction error than benign profiles and input energy dominates
  the score order; the registered attention layer does not reverse it.
- **Current conclusion + label:** **OBSERVED** — threshold selection cannot
  recover the reported operating point for this registered completion.
- **Remaining uncertainty / blast radius:** Test-set ADASYN is omitted, the
  paper leaves the attention query/key/value and decoder recurrence materially
  underspecified, and this is one seed. Other predeclared completions remain
  open.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/iset_lstm_aea_seed11_20260818.json`.

### Completed Table-III model-family breadth map

- **Former belief/status:** The FC-SAE anchor had a large gap, but the project
  did not yet know whether the same failure appeared across named benchmark and
  proposed families or whether the prepared attacks were learnable at all.
- **Evidence:** One registered seed-11 completion and score audit now exists for
  all eleven Table-III rows. None of the registered fixed operating points
  reproduces its complete printed metric pattern. Supervised feed-forward is
  nevertheless a strong positive control (AUC 97.05%) and can approach the
  paper's DR/FA pair after a retrospectively selected threshold. It and Naive
  Bayes expose material threshold-rule omissions; the other nine score vectors
  remain materially far from the reported DR/FA corner in their registered
  direction.
- **Root cause:** **PARTLY INFERRED, MOSTLY OPEN** — multiple unsupervised and
  recurrent completions rank attacks backward or collapse, while the supervised
  feed-forward result proves the generated attack population contains a strong
  learnable signal. Printed ADASYN and several source ambiguities remain open.
- **Current conclusion + label:** **OBSERVED** — model-family breadth is closed
  for the registered one-seed no-test-ADASYN completions. It is premature to
  add seeds before the predeclared one-factor data/evaluation interpretations
  locate the reasonable divergence branches.
- **Remaining uncertainty / blast radius:** These are not confirmatory
  intervals, do not execute printed ADASYN, and do not establish intent or an
  infinite-space impossibility claim.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/TABLE_III_BREADTH.md`.

### Corrected ROC-point lower bound

- **Former belief/status:** A historical sweep draft treated balanced accuracy,
  `(TPR + 1 - FPR) / 2`, as a general lower bound on ROC-AUC and proposed using
  it to reject branches.
- **Evidence:** A monotone ROC curve can remain at TPR zero until the reported
  FPR, rise vertically to the reported TPR, and then remain there until FPR 1.
  The general bound from one ROC point is therefore
  `AUC >= TPR * (1 - FPR)`, not balanced accuracy. The Table-II bounds range
  from 0.7138 to 0.9216 and do not contradict the paper's AUC cells.
- **Root cause:** **VERIFIED mathematical correction** — the earlier draft
  conflated one operating point's balanced accuracy with area under the full
  ROC curve.
- **Current conclusion + label:** **INVALIDATED** — balanced accuracy is not an
  AUC lower bound. Only the product bound may be used as a necessary condition;
  full saved-score ROC enumeration remains the exact threshold-feasibility
  audit for an executed attempt.
- **Remaining uncertainty / blast radius:** This correction removes one
  proposed rejection argument. It does not affect the independent F1 identity,
  common-prevalence contradiction, or observed experimental results.
- **Source artifact:**
  `docs/plans/2026-07-22-confirmatory-branch-sweep-design.md`.

### SGCC Table-II model and representation breadth

- **Former belief/status:** The paper's Table-II numbers were suspected to be
  far outside what its described reconstruction models could produce, but no
  complete exact-source model-family map existed and a bad SGCC reduction could
  still explain a single failed run.
- **Evidence:** Panther seed-11 runs now cover all eleven models on `last_48`
  and all five proposed models on `first_48` and `binned_mean_48`. Proposed AUC
  is 46.31--54.15 versus 83--93 reported. The same prepared branches give the
  feed-forward control AUC 95.31--96.91 and DR/FA gaps of only 0.80--1.94
  points. Exact enumeration of all deterministic thresholds leaves every
  proposed saved score vector 33.18--50.53 points from its complete reported
  metric row. Within each representation all proposed-model score rankings
  correlate at least 0.957 by Spearman; the SAE/AEA raw MSE pairs and VAE pairs
  are nearly identical by Pearson within their score scales.
- **Root cause:** **INFERRED for these completions** — the printed output-domain
  constraints and standardized targets make the reconstructions track a common
  representation-dependent energy ordering. LSTM, VAE, and attention layers do
  not materially change that ordering in the executed branches.
- **Current conclusion + label:** **OBSERVED** — the complete seed-11 proposed
  family does not reproduce Table II under early, late, or whole-history
  48-wide one-customer readings, and threshold selection cannot rescue the
  saved scores. The user's architecture-level intuition is supported for this
  finite map; the universal claim over all undisclosed implementations is not
  established.
- **Remaining uncertainty / blast radius:** Initialization uncertainty has not
  been measured. Windowed SGCC samples change the sample and split identity;
  alternate missing-value, scaling, latent, recurrent-input, and decoder
  completions remain separate branches. No intent claim follows.
- **Source artifacts:**
  `studies/atk-2022-deep-autoencoder/TABLE_II_BREADTH.md` and
  `studies/atk-2022-deep-autoencoder/results/sgcc_table_2_breadth_seed11_20260818.json`.

### ISET Table-V common-model breadth

- **Former belief/status:** Fixed-model FC-SAE already showed invariant FA, but
  the structural result had not been derived from all five proposed Table-III
  score vectors.
- **Evidence:** For every proposed model, applying its fixed threshold to the
  same held-out benign scores and attacks 1--6 yields one exactly repeated FA:
  58.22, 40.96, 32.62, 25.79, and 58.22% respectively. The paper instead
  reports attack-varying FA ranges of 10--19, 9--15, 8--12, 4.5--8.5, and
  2.5--6.5%. Most reproduced attack-specific DR cells are also far below the
  printed values.
- **Root cause:** **VERIFIED mathematical identity for this interpretation** —
  with fixed model, threshold, and benign identities, FP and TN cannot depend
  on which malicious attack population is paired with them; therefore FA
  cannot vary.
- **Current conclusion + label:** **OBSERVED** — the common-model/common-benign
  interpretation does not reproduce Table V. The paper's pattern requires an
  unstated change in model, threshold, benign identities, or their combination.
- **Remaining uncertainty / blast radius:** “Multiple experiments” can support
  retraining or resplitting interpretations. Those predeclared branches remain
  open and must not be conflated with the common-model structural result.
- **Source artifacts:**
  `studies/atk-2022-deep-autoencoder/TABLE_V_BREADTH.md` and
  `studies/atk-2022-deep-autoencoder/results/iset_table_5_common_model_seed11_20260818.json`.

### ISET proposed-model exact threshold exclusion

- **Former belief/status:** The breadth map showed large fixed-threshold and
  nearest-ROC-point gaps, but did not yet exclude every deterministic threshold
  against the complete seven-metric row.
- **Evidence:** Panther jobs `378199`--`378203` enumerated every score threshold
  for each saved seed-11 proposed-model vector and minimized the maximum
  absolute gap over DR, FA, SP, PR, ACC, F1, and AUC. The exact minima are
  49.96, 48.91, 55.04, 58.48, and 60.11 percentage points in FC-SAE through
  LSTM-AEA order.
- **Root cause:** **OBSERVED** — each registered score vector has fundamentally
  different class ranking/operating characteristics from its printed row; no
  cutoff on that vector can jointly repair the metrics.
- **Current conclusion + label:** **VERIFIED finite exclusion** — threshold
  choice is not an explanation for any complete proposed-model row for these
  five vectors.
- **Remaining uncertainty / blast radius:** Different data interpretations,
  training seeds, and hyperparameters can create different score vectors. The
  exact proof must not be generalized beyond the recorded vectors.
- **Source artifact:**
  `studies/atk-2022-deep-autoencoder/results/iset_table_3_exact_threshold_gaps_seed11_20260818.json`.

### Compact target-transcription omission

- **Former belief/status:** The compact runner was assumed to contain every
  Table-IV and Table-V reported target after model-family breadth completed.
- **Evidence:** FC-VAE half-data job `378184` completed training and scoring but
  failed while looking up an absent Table-IV target key. Static inspection found
  only FC-SAE/LSTM-SAE Table-IV targets and only FC-SAE Table-V targets.
- **Root cause:** **VERIFIED implementation omission** — result metadata was
  incompletely transcribed even though model/data execution was unaffected.
- **Current conclusion + label:** **CORRECTED** in `fcd2d78`; all five proposed
  models now have all 3 Table-IV and all 6 Table-V targets, enforced by a test.
  Completed scores are recovered without retraining and failures stay preserved.
- **Remaining uncertainty / blast radius:** The omission affects historical
  result target annotations and post-score failures only; it does not alter any
  saved model, score vector, prediction, or metric.

### Clean-reader source-only findings before the discovery sandbox

- **Former belief/status:** The concern was initially intuitive: the paper's
  results looked unusually clean, the task appeared possibly too easy to
  support the architectural story, and some written operations seemed
  incapable of behaving as claimed. Earlier project implementations and
  outcomes could not be allowed to decide whether that intuition was a fair
  reading of the source.
- **Evidence:** During Phase 1 of the 2026-08-23 clean-reader rebase, the exact
  12-page PDF was fingerprinted, visually inspected, and text-extracted page by
  page while prior method reconstructions, code, and results remained closed.
  The resulting source-only record transcribes the complete Tables II--V target
  pattern, maps five explanatory claims as `B > A because Z exploits S`, and
  records every material contradiction, omission, static constraint, and
  smallest sandbox question found in that pass.
- **Finding 1 — reported-pattern triage (`OBSERVED`, source-only):** Every
  proposed model is ordered FC-SAE, LSTM-SAE, FC-VAE, LSTM-VAE, LSTM-AEA in
  both main tables for every displayed metric, with FA reversed. The same
  ordering persists at all three data sizes and for DR/FA on all six attacks;
  every data-size accuracy curve is monotone. No run count, seed, dispersion,
  confidence interval, or failed experiment is reported. Several flattering
  metrics are algebraically dependent by the paper's own definitions:
  `SP=100-FA`, `ACC=(DR+SP)/2`, and F1 is derived from DR and PR. The pattern is
  a legitimate audit signal, not evidence of selection, fabrication, or
  intent.
- **Finding 2 — explanatory identification (`OBSERVED`, source-design
  assessment; `M` remains formally open):** The reported comparisons are
  comparisons between complete configurations, not isolated tests of the
  credited components. LSTM versus FC also changes depth, width, activation,
  optimizer, and dropout; VAE versus SAE changes model, objective, anomaly
  score, and threshold; AEA versus VAE is not attention alone. The paper shows
  no matched component ablation, structure destruction, capability witness,
  learned-behavior inspection, or paired uncertainty. Therefore the printed
  tables alone do not identify that recurrence, latent variance, or attention
  caused the reported advantages. They also do not show that those mechanisms
  are absent.
- **Finding 3 — task triviality (`HYPOTHESIS`):** ISET attacks 1--5 introduce
  conspicuous changes in amplitude, zero count, constancy, range, or variance
  that may be detected by zero-parameter or one-feature rules. Attack 6
  reverses time order while preserving the day's multiset and total, and is the
  clearest printed capability witness for temporal sensitivity. Until a
  triviality floor and structure-sensitive controls are run through the same
  evaluation path, high aggregate performance cannot establish that depth,
  recurrence, variational modeling, or attention was necessary.
- **Finding 4 — executable-method status (`VERIFIED` source contradictions and
  `OPEN` completions):** The printed Attack-3 endpoint subtracts a positive
  duration from its start and therefore cannot create the described forward
  interval under an ordinary literal reading. The VAE section says low
  reconstruction probability is anomalous while the shared decision rule says
  greater-than-threshold is anomalous. Equation (9) mixes distributions over
  different variables; a valid positive variance parameterization is omitted;
  the ROC/IQR threshold phrase does not define a unique scalar; the benign-only
  training population cannot directly supply the labeled DR/ROC validation
  later invoked; and customer identity, sample unit/cardinality, and layer
  count remain materially ambiguous. A formal reproduction therefore requires
  visible literal failures and predeclared `I` completions, never silent
  repair.
- **Finding 5 — geometric reconstruction floor (`VERIFIED` under the printed
  preprocessing/output assumptions):** The paper standardizes model targets
  to zero mean and unit variance, while Table I assigns Softmax outputs to the
  FC autoencoders and sigmoid outputs to the recurrent autoencoders. A Softmax
  reconstruction lies in the nonnegative unit-sum simplex and a sigmoid
  reconstruction lies in the unit box. For a general standardized 48-vector
  `x`, every admissible Softmax reconstruction therefore satisfies
  `MSE >= dist(x, simplex)^2/48`; the sigmoid analogue is
  `MSE >= dist(x, [0,1]^48)^2/48`. No optimizer, capacity increase, or extra
  compute can remove that representation-domain floor. This is a structural
  bound on exact reconstruction under those assumptions, **not** a bound on
  DR, FA, AUC, ranking quality, or the published target pattern.
- **Finding 6 — present three-part verdict (`OPEN` / not yet earned):** The new
  clean-reader program has not yet produced a numerical reproduction finding,
  a capability-sensitive formal mechanism finding, or an empirical
  attainability envelope. The strongest present statement is that the source
  reports an exceptionally tidy target pattern, does not isolate the causal
  mechanisms used to explain that pattern, may evaluate a shortcut-dominated
  task, and is not uniquely executable as written. Whether a predeclared
  ordinary completion reproduces the complete target remains open.
- **Root cause:** **INFERRED at the source-design level** — the publication
  attaches causal explanations to aggregate comparisons among jointly changing
  configurations, while its synthetic attacks and evaluation do not force the
  credited capabilities to be necessary and several consequential operations
  are contradictory or omitted.
- **Current conclusion + boundary:** **OBSERVED/VERIFIED SOURCE FINDINGS** —
  suspicion has been converted into explicit, falsifiable questions and one
  genuine reconstruction-domain bound. It has not been converted into an
  accusation, a numerical non-reproduction, a mechanism-absence claim, or an
  attainability verdict.
- **Remaining uncertainty / blast radius:** Phase 2 must use only disposable
  `X` probes to test output-domain behavior, the triviality floor, temporal,
  variance, and attention witnesses, score direction, threshold semantics, and
  one-example execution. Any surviving question must then return to the PDF
  and exact data for a frozen formal contract. Historical results remain
  preserved but cannot retrospectively supply the clean-reader finding.
- **Source artifacts:**
  `studies/atk-2022-deep-autoencoder/CLEAN_READER_ORIENTATION.md`,
  `studies/atk-2022-deep-autoencoder/CLEAN_READER_RECONCILIATION.md`, and
  `docs/plans/2026-08-23-clean-reader-reproduction-rebase.md`.

### Clean-reader Phase-2 disposable discovery sandbox

- **Former belief/status:** The source-only pass suggested that attacks 1--5
  may expose simple statistical shortcuts, reversal may require temporal
  sensitivity, printed decoder domains may impose score geometry, and the VAE
  threshold wording may reverse anomaly orientation. None had yet been
  instantiated independently of the historical implementation and named data.
- **Evidence:** The pre-recorded one-seed toy/synthetic job `381540` executed a
  standalone script on one Panther GPU and completed once with exit `0:0`.
  One-dimensional rules attained AUC 0.993--1.000 on toy attacks 1--5, while
  order-insensitive reversal AUCs remained approximately 0.5. A comparable
  dense AE detected block disruption at AUC 0.950, whereas the seq2seq LSTM was
  near chance at 0.521 and both were near chance on reversal; the LSTM's much
  higher final training loss prevents a capability-absence interpretation.
  Exact unit-box/simplex projection floors were positive and strongly
  population-dependent. Fixed-unit Gaussian reconstruction probability
  strictly decreased with reconstruction error.
- **Root cause:** **OBSERVED within the toy construction** — attacks 1--5
  alter simple marginal statistics, reversal preserves the multiset, bounded
  decoder ranges cannot represent general standardized vectors, and Gaussian
  density is monotone decreasing in squared error. The recurrent witness's
  failure is unresolved between underfitting, witness design, and absent
  incremental capability.
- **Current conclusion + label:** **OBSERVED, EXPLORATORY `X` ONLY** — the
  sandbox supports promoting a triviality-floor control, output-domain audit,
  and explicit VAE score-orientation branches into the clean-reader source
  freeze. The dense/LSTM result rationally increases concern about the claimed
  recurrent explanation because adding recurrence did not reveal a temporal
  advantage and fitted the benign task much worse. A dense network over 48
  ordered coordinates can itself learn time-position relationships, so
  recurrence is an inductive bias rather than exclusive access to temporal
  information. The sandbox did not identify a recurrence mechanism and did
  not test the paper's numerical results or named data.
- **Remaining uncertainty / blast radius:** No toy metric may be substituted
  for a paper result or used to choose a favorable completion. Variance and
  attention witnesses were deliberately not run. Any formal `M` test must be
  frozen later, use the exact evaluation path, show matched fitting success,
  and separate `A`, `B`, `Z`, and `S`. The contradictory VAE score direction
  may be a source typo; preserve and test both directions without inferring an
  implementation choice or intent from the wording alone.
- **Source artifacts:**
  `studies/atk-2022-deep-autoencoder/DISCOVERY_SANDBOX.md`,
  `studies/atk-2022-deep-autoencoder/exploration/phase2_discovery_sandbox.py`,
  `studies/atk-2022-deep-autoencoder/exploration/results/phase2_seed_20260824.json`,
  and Panther job `381540`.

### Clean-reader Phase-3 source freeze

- **Former belief/status:** Phase 1 had identified literal failures and
  omissions but deliberately selected no repair; Phase 2 had identified useful
  questions but could not supply formal interpretations.
- **Evidence:** All 12 pages of the fingerprinted publication were visually
  re-inspected with the sandbox questions in view. Official ISSDA metadata
  independently fixed the named ISET deposit, seven required filenames, byte
  sizes, and MD5s. The source locations were then converted into one literal
  `P` route, preserved failures, a finite alternatives register, and one
  outcome-independent `I` path for the simplest proposed Table-III row.
- **Root cause:** The paper defines the broad data-to-result order and final
  FC-SAE settings but cannot execute Attack 3 or threshold selection literally
  and omits several data, RNG, dropout, and training semantics. A clean-reader
  reproduction therefore needs visible completions even before code fidelity
  can be assessed.
- **Current conclusion + label:** **VERIFIED SOURCE FREEZE; CHECKPOINT 1
  APPROVED** — `CR-ISET-FCSAE-01` is the candidate first anchor. It preserves joint
  pre-split scaling, test-set ADASYN, Softmax reconstruction, and threshold
  0.58 while visibly completing attack timing, identity mapping, exact data,
  architecture details, training, RNG, and metric semantics. The user approved
  Phase-4 fidelity assessment on 2026-08-24 and required a systematic register
  of competing explanations. Formal execution still requires its Phase-5
  pre-run contract.
- **Remaining uncertainty / blast radius:** The approved exact official
  allocation `.tab` is an operational data gate. Historical code/results were
  opened only for the Phase-4 fidelity trace; their outcomes cannot
  retroactively validate or select the freeze.
- **Source artifacts:**
  `studies/atk-2022-deep-autoencoder/CLEAN_READER_SPECIFICATION.md` and
  `docs/decisions/2026-08-24-clean-reader-first-anchor-freeze.md`.

### Clean-reader Phase-4 fidelity correction and Phase-5 data gate

- **Former belief/status:** The five direct reproduction files were candidate
  instrumentation whose fidelity was deliberately unknown; explanation E11
  (our implementation is wrong) had to be addressed before interpreting a new
  numerical result.
- **Evidence:** A complete static trace found that FC-SAE architecture, strict
  days, five attacks, scaling, split mechanism, score direction, threshold, and
  metric formulas were reusable. It also found material mismatches: semantic
  allocation CSV rather than official `.tab`, seed 11, clipped Attack 3,
  attacks from all customers in the held-out set, batch 512,
  `min_delta=1e-4`, no ten-epoch minimum, unresampled scoring by default, no
  eligible-contract guard, and no Softmax projection floor. Only these frozen
  fields were corrected. All prior affected attempts were quarantined rather
  than deleted. The corrected route reloads persisted weights before scoring,
  hashes run artifacts, and has an independent fail-closed anchor audit. The
  full suite passes: 140 study tests plus 38 root tests.
- **Root cause:** The earlier compact route encoded a different historical
  interpretation and exploratory execution defaults. Passing its own tests did
  not make it an implementation of the newly frozen clean-reader contract.
- **Current conclusion + label:** **VERIFIED IMPLEMENTATION FIDELITY WITH
  NAMED-DATA BLOCK** — Phase 4 closes the identified E11 mismatches for the
  declared route at the static/tiny-test level. It produces no `N` result.
  Exact local mirror archives match all six official byte/MD5 identities, but
  the official 196,316-byte allocation `.tab` is absent both locally and on
  Panther and no ISSDA token is configured. Phase 5 therefore stops at the
  exact data gate.
- **Remaining uncertainty / blast radius:** A defect may still be exposed by
  exact preparation or artifact inspection; Phase 4 does not prove that the
  model will train successfully or reproduce any target. The semantic CSV may
  be scientifically equivalent, but it remains an unapproved `I` branch and
  cannot fill this anchor silently. No mechanism or attainability explanation
  is updated by this block.
- **Source artifacts:** `studies/atk-2022-deep-autoencoder/CLEAN_READER_FIDELITY.md`,
  `studies/atk-2022-deep-autoencoder/EXPLANATION_REGISTER.md`, and the five
  direct files under `studies/atk-2022-deep-autoencoder/reproduction/`.

### Clean-reader runtime correction and sandbox/full-data boundary — 2026-08-31

- **Former belief/status:** The running anchor was forecast to finish around
  07:00--08:00 on 2026-08-31 by combining a 14.16-hour historical neighbor-search
  extrapolation with a 3.37-hour historical batch-32 training run. The user
  reasonably questioned whether multi-hour computation was still sandbox work.
- **Evidence:** Phase-2 sandbox job `381540` took 2:25 total, with 60.06 s
  recorded by its script. The later full-data numerical anchor `384390` ran
  from 2026-08-30 13:34:12 to 22:48:39 Qatar time and completed `0:0` after
  9:14:27. Its metadata records 3,890.66 s for exact ADASYN and approximately
  4,183.36 s for profile extraction plus preparation. Its result records
  28,965.90 s of fitting across 28 epochs on a Tesla P100, best epoch 23, and
  11.12 s of Table-III scoring. Training has 1,500,523 profiles; the resampled
  evaluation has 8,884,989 rows.
- **Root cause:** **VERIFIED mismatch in ETA inputs** — the historical neighbor
  benchmark used 14,258,510 references, versus this contract's 5,255,369 original
  test rows. The training comparison also crossed device, seed, and stopping
  semantics. Population size alone does not explain the entire timing gap;
  the remaining implementation/hardware throughput contributions are not
  isolated. Treat historical timing as context, not a portable forecast.
- **Current conclusion:** **INVALIDATED ETA; OBSERVED operational completion**.
  The sandbox remained small and quick. This separate `P+I/N` full-data anchor
  was execution depth, not sandbox breadth. The saved result self-reports
  success, but its independent Phase-6 audit and initial numerical finding
  remain pending. No mechanism or attainability conclusion follows from timing.
- **Sources:** `studies/atk-2022-deep-autoencoder/DISCOVERY_SANDBOX.md`; Slurm
  job `384390` accounting and the remote repository's `slurm-384390.out`;
  the semantic-allocation cache's `metadata.json`; and
  `seed_20260824_2f483335536c/result.json` under the existing study's remote
  reproduction-derived results path.

### First audited clean-reader numerical finding — 2026-08-31

- **Former belief/status:** The sole full-data attempt had completed, but its
  self-reported metrics were not yet an independently trusted finding.
- **Evidence:** The frozen audit ran on CPU allocation `384939` and passed:
  every metric/count regenerated with zero discrepancy. The corrected
  supplemental checker passed 65 checks, fully scanning 31 arrays with no
  NaN/Inf, verifying disjoint customer identities and preserved original rows,
  replaying stopping at epoch 28 with best epoch 23, and reloading saved weights
  whose 256 sampled scores differed by at most 1.1921e-7. DR/FA/ACC/AUC/F1 are
  25.48/45.13/40.18/39.40/30.09%, versus 81/15/83/81/81% reported. All seven
  metrics and deltas are in `CLEAN_READER_FINDING.md`.
- **Checker correction:** The first supplemental report failed because its
  last partial original-data chunk was compared with an overlong resampled
  slice. The defect was in this new checker, not in the data or reproduction.
  Its source and failed report are preserved; clamping the slice yielded a
  passing second report. No scientific attempt, score, weight, or data changed.
- **Root cause:** **OPEN for the numerical gap.** On these fixed scores,
  all-threshold balanced ACC is at most 50.00072% in the printed direction and
  60.21% reversed. Mean benign error exceeds malicious error. Trained scores
  correlate 0.999253 with zero-reconstruction input energy; trained ACC improves
  over that rule by only 1.18 points, and the per-row simplex floor has ACC
  40.20% versus 40.18% trained. These are descriptive geometry/scoring clues,
  not an isolated causal mechanism or a global classification bound.
- **Current conclusion:** **OBSERVED `P+I/N` initial non-reproduction** of one
  Table-III ISET FC-SAE completion. **VERIFIED artifact checks** weaken E11 for
  the checked chain. E7 is supported diagnostically; E8's fixed-score
  threshold-only rescue is closed for this vector. E9/E10 remain open across
  other source completions. Formal `M` and `A` findings remain open; no
  recurrent comparison, repeated seed, or performance envelope was added.
- **Boundary:** This is one descriptive seed under visible completions, not
  confirmation, proof of what author code did, or a paper-wide conclusion.
  ADASYN's actual class counts are near-balanced, not exactly equal; its
  synthetic benign identities are preserved. Table-IV target metadata and
  per-attack audit output do not expand the Table-III-only contract.
- **Sources:** `studies/atk-2022-deep-autoencoder/CLEAN_READER_FINDING.md`,
  `results/clean_reader_anchor_20260831/` under that study, and the scoped
  `checks/clean_reader_anchor_artifacts.py`. Phase 6 is complete; Checkpoint 2
  now requires user review before any additional experimental promotion.

### Public explanation and interpretation corrections — 2026-08-31

- **Former presentation:** the public site led with internal workflow language
  and showed an earlier no-test-ADASYN electricity run as current. Some older
  water-study wording described finite searches and empirical observations as
  universal limits.
- **Evidence:** the audited clean-reader result is already preserved, including
  its full ADASYN evaluation and fixed-score diagnostics. Water search output
  records 229 selected test sizes, not every size in its range. Replay-distance
  and capacity experiments do not establish universal classification bounds.
- **Correction:** a separate readable report now connects paper locations to
  the exact code revision, model diagram, settings, all seven metrics, checks,
  possible causes, and limitations. Earlier pages retain their measurements
  and are clearly dated. Water interpretation corrections are appended to that
  study's `EVIDENCE.md`; no raw artifact or scientific run changed.
- **Current scope:** one numerical non-reproduction remains the current Paper 1
  finding. A conditional output-domain performance bound, useful-information
  measurement, and small controls are proposed but not performed. No claim of
  statistical impossibility, zero useful work, or fabrication is added.
- **Reasoning to preserve:** “useful work” means incremental task-relevant
  contribution over a fair simple comparison, not merely changed weights or
  a small training loss. Equivalence needs a justified effect threshold and
  dependence-aware uncertainty. A stretched percentage axis changes no
  probability; a confidence interval for a fixed model or mean does not rule
  out another seed/configuration. Search-time estimates condition on the
  declared search process and are not minimum historical runtimes.

### Conditional limit, with counterevidence to zero useful work — 2026-08-31

- **Former belief/status:** the audited numerical gap and fixed-score cutoff
  limit did not exclude another seed or configuration. Near-perfect
  correlation with input energy suggested little useful contribution but did
  not establish score equivalence. The proposed bound and controls had not run.
- **New evidence (`C/A`):** primary diagnostic revision `1175e8d` ran on CPU
  job `385090`. It calculates each prepared input's minimum and maximum MSE
  over the closed probability simplex, grants attacks the maximum and benign
  rows the minimum, then enumerates every boundary. On all 8,884,989 rows,
  this label-aware relaxation gives maximum balanced ACC 50.92105%, AUC
  45.10918%, and DR 9.24779% at FA <=15%, versus 83/81/81% reported. Target
  rounding, original-only rows, and reversed direction also fail to recover
  the reported operating point. All saved scores are inside their padded ranges.
- **Root cause isolated within scope:** the fixed prepared input geometry,
  Softmax output domain, and MSE scoring are jointly incompatible with the
  target. The relaxation contains every allowed network output, even granting
  a separate reconstruction for identical inputs. Another seed, wider hidden
  layers, optimizer, or training duration cannot change that conditional
  conclusion. It does not identify which source interpretation or unobserved
  author procedure differs. Evaluation is float64 with outward padding, not
  certified interval arithmetic or a claim about an unseen population.
- **Counterevidence (`C/M`, then adaptive `X/M`):** original-row trained-minus-
  zero balanced ACC is +0.89081 points, customer-bootstrap 95% interval
  [0.80454, 0.98117]. Gains over uniform and projection scores are +0.85335
  and +0.02975 points; some per-attack gains exceed one point. Fewer benign
  alarms drive the aggregate gain while attack detection decreases. Near-
  perfect overall correlation does not make scores interchangeable: within
  energy bands, the trained score distinguishes labels better than energy.
  The separately frozen adaptive control (`26a42db`, job `385091`) sampled
  10,000 source days with six attack siblings. Within-band AUC was
  trained/projection/uniform/energy = 65.49/62.18/55.02/49.74%.
- **Current conclusions:** numerical non-reproduction unchanged; conditional
  attainability limit `VERIFIED`; “nothing useful learned” `WEAKENED, NOT
  ESTABLISHED`. The matched architectural mechanism remains `OPEN`. These
  are score comparisons, not a matched trained-versus-untrained causal test.
- **Uncertainty:** the 2,000 customer-cluster resamples hold fitted model,
  scaler, split, and generated attacks fixed and assume exchangeable customer
  clusters; no synthetic-row bootstrap or seed-level interval was used.
  The +/-1-point region was specified before the new measurements but after
  the original result was known. The adaptive within-band differences have
  no confidence interval and are not independent confirmation. Other simple
  rules and other source-supported input/output/score choices remain untested.
- **Compute and stopping:** the full analysis took 112.92 seconds after a
  5.00-second pilot, and the adaptive control took 18.03 seconds. Allocations
  ended `0:0` after 4:16 and 0:53. No training or original data/result changed.
  Both questions are answered within scope; stop rather than launch more
  seeds. The proposed next discussion concerns source assumptions that can
  change the bound. No new experiment follows automatically.
- **Sources:**
  [diagnostic finding](../studies/atk-2022-deep-autoencoder/POST_ANCHOR_FINDING.md),
  [primary frozen contract](../studies/atk-2022-deep-autoencoder/POST_ANCHOR_DIAGNOSTICS.md),
  [adaptive control contract](../studies/atk-2022-deep-autoencoder/ENERGY_BAND_CONTROL.md),
  [results and execution records](../studies/atk-2022-deep-autoencoder/results/post_anchor_20260831/).

### Does the source force the assumptions? — 2026-08-31

- **Former uncertainty:** the fixed-preparation Softmax/MSE bound was strong,
  but a consequential interpretation supplied by us might have created it.
- **Source evidence:** complete paper re-read; printed pp. 4109, 4114, 4115,
  and 4116 visually checked. Softmax for FC-SAE, MSE, high-error direction,
  and threshold 0.58 are explicit. The fitted normalization scope is omitted.
- **New `C/A` diagnostics on explicit input interpretations:** the current
  reference, joint-scalar scaling, and weaker separate-class feature scaling
  cap detection at 0.58 at 29.58%, 29.81%, and 33.96% (rounded upward),
  versus 81%. Alternative statistics use the full pre-split original
  population. All 4,504,602 original attack rows were evaluated.
- **Scope distinction:** attack-only detection at the fixed cutoff cannot
  be improved by adding synthetic benign rows, so that exclusion survives
  benign-only ADASYN on unchanged attacks. The original-row all-cutoff/AUC
  results do not establish a complete resampled bound for new scalings.
- **Counterevidence/control:** the Sigmoid cube raises the original-row ACC
  ceiling from 50.16% to 80.84%. It still excludes the combined target in the
  printed direction, while Sigmoid plus reversal no longer excludes the
  DR>=81%, FA<=15% pair in the label-aware relaxation. Those are changes to
  the stated final FC-SAE, not a reproduced or trained model. The evidence
  does not justify a universal limit on every bounded-output detector.
- **Static correction guard:** SSE=48*MSE and RMSE=sqrt(MSE) preserve rankings
  and all-cutoff ROC regions; changing units alone cannot rescue the previous
  complete-evaluation bound. A genuinely different score remains outside it.
- **Current interpretation:** source confidence in the main output/score
  choices increases. The fixed-cutoff non-attainability result extends to
  two additional explicit normalization readings. Other populations and
  transformations remain open. No new numerical or learned-mechanism finding,
  nor inference about author code or intent, is added.
- **Execution:** local freeze `b76cb02`, 212 pre-run tests passed. CPU job
  `385119` completed `0:0`, pilot 8.20 s, full 56.76 s, allocation 3:18.
  All hashes and identities passed, and the original reference bound was
  reproduced. Original artifacts and public files remain unchanged.
- **User correction to workflow:** discuss results before updating public
  writing or publishing. Local records and commits preserve this round;
  no push or new experiment follows automatically.
- **Sources:**
  [source map and frozen setup](../studies/atk-2022-deep-autoencoder/SOURCE_ASSUMPTION_CHECK.md),
  [discussion finding](../studies/atk-2022-deep-autoencoder/SOURCE_ASSUMPTION_FINDING.md),
  [complete records](../studies/atk-2022-deep-autoencoder/results/source_assumption_20260831/).

### Sigmoid on the full prepared evaluation — 2026-08-31

- **Former uncertainty:** original-row Sigmoid range permitted high detection
  alone but excluded the high-error DR/FA pair. Synthetic benign rows had not
  been included, so the complete-evaluation answer was explicitly open.
- **New exploratory `X/A` evidence:** keep the full prepared inputs unchanged;
  replace only the allowed reconstruction range by the closed Sigmoid cube.
  On 8,884,989 rows, the printed cutoff still fails: minimum FA 29.66640%,
  versus 15%. But allowing a different high-error cutoff gives an upper DR
  of 85.32587% at FA<=15%; balanced-accuracy and AUC ceilings are 85.69966%
  and 90.02601%. The target pair is no longer excluded. Reversed scoring
  remains open too (DR ceiling 93.76498% at FA<=15%).
- **Why the answer changed:** adding the unchanged synthetic benign rows
  changes the false-alarm population. The earlier original-row DR ceiling of
  59.98410% is reproduced within tolerance; it was not a full-evaluation bound.
  The attack population and its fixed-cutoff DR ceiling are unchanged.
- **Concrete controls:** clipped input and constant-half reconstructions use
  no labels or training. At the printed cutoff their full-data BA is 42.59659%
  and 45.72641%; their best-cutoff BA across both directions is 58.23399%
  (rounded). They do not realize the label-aware bound's advantage. These two
  controls cannot establish failure of a properly trained Sigmoid model.
- **Current conclusion:** Sigmoid alone does not rescue the printed-cutoff
  procedure. Sigmoid plus another cutoff is an unresolved alternative, not
  excluded by this bound and not reproduced. No new trained numerical or
  architectural-mechanism result follows. Do not claim all bounded-output
  detectors or all reasonable implementations fail.
- **Execution:** local freeze `9d6c31b`, 223 pre-run tests; CPU job `385137`
  completed `0:0`, pilot 4.97 s, full 39.70 s, total allocation 2:17. No fitted
  model, head swap, rescaling, regeneration, or original artifact changed.
  Both directions, both controls, full precision and failures are preserved.
- **Publication boundary:** the already discussed source findings were
  published first in `dc37bbe` with Pages deployment `33419150100` verified.
  These new outcomes are local pending discussion; no automatic next run or
  public update. The user's question-led plain-language reporting emphasis
  now appears inside the relevant methods blocks.
- **Sources:** [frozen setup](../studies/atk-2022-deep-autoencoder/SIGMOID_SANITY.md),
  [finding](../studies/atk-2022-deep-autoencoder/SIGMOID_SANITY_FINDING.md),
  [all records](../studies/atk-2022-deep-autoencoder/results/sigmoid_sanity_20260831/).

### Small genuinely trained Sigmoid alternative — 2026-08-31

- **Former uncertainty:** the complete-data Sigmoid range bound permits the
  target after a cutoff change, but neither trivial control realizes it. A
  learned Sigmoid model remained untested. The request to prove failure did
  not make failure a premise of the investigation.
- **New exploratory `X/A` evidence:** paired FC-SAE models differ only in
  their final Softmax/Sigmoid activation, with identical initial weights,
  fitting sample, budget, and dropout seeds. Both completed ten epochs / 640
  updates on 2,048 fitting rows. Benign calibration alone selected checkpoints.
  On 12,119 sampled held-out rows, best DR at FA<=15% was 8.64258% Softmax and
  9.74935% Sigmoid. Reversing gives 25.52083% and 25.39063%. The 81% target
  and rounding-relaxed pair fail for every cutoff. Original-only and initial
  untrained views also fail; every result is retained.
- **What is isolated:** no overlooked cutoff rescues these fixed score
  vectors on these rows. This is exact finite threshold enumeration, not a
  sparse threshold search, an all-weights limit, or a general-population CI.
  The quick trained replacement did not realize the permissive range ceiling.
- **Counterevidence to a stronger interpretation:** Sigmoid calibration MSE
  fell 1.61028→1.33863 with a marked improvement at epochs 5–6; its best epoch
  is the last one tested. No long-run plateau is established. High-error AUC
  fell 43.93233→37.70499%, while reversed AUC rose; better reconstruction is
  distinct from the paper's detection objective. Neither zero useful work nor
  inevitable failure under longer training follows.
- **Current conclusion:** the changed-head/changed-cutoff rescue failed for
  this small fitted pair. Other Sigmoid fits, more data, longer budgets, and
  changed procedures remain open. The earlier complete-input Softmax proof
  remains separate and is not extended to Sigmoid. No new eligible numerical
  reproduction or architectural-mechanism verdict is earned.
- **Execution and stop:** `cc9af5e`, 230 pre-run tests; CPU job `385198`
  completed `0:0`. Pilot analysis 9.22 s, paired small analysis 24.81 s,
  allocation 3:52 including imports and inspection. All hashes, identities,
  finite outputs/scores/losses, initial-weight equality, and update checks
  passed. All records remain local; discuss before publication or further work.
- **Sources:** [frozen setup](../studies/atk-2022-deep-autoencoder/SIGMOID_FIT_CHECK.md),
  [finding](../studies/atk-2022-deep-autoencoder/SIGMOID_FIT_FINDING.md),
  [records](../studies/atk-2022-deep-autoencoder/results/sigmoid_fit_20260831/).

### Recurrent feasibility score differences at decision level — 2026-09-01

- **Former status:** LSTM-SAE and LSTM-VAE completed both feasibility epochs
  but remained blocked by the frozen absolute `1e-6` all-score batch-agreement
  gate. Whether the observed floating-point drift changed a classification or
  performance conclusion was unknown.
- **New operational `X` evidence:** jobs `385583` and `385584` reloaded the
  exact saved weights and the same 12,119 score rows, with no training, and
  evaluated batches 256/128/64/32. Largest primary-score differences were
  `2.0113942e-5` for LSTM-SAE and `3.0121445e-5` for LSTM-VAE. At the printed
  cutoffs, all labels and seven metrics were identical across every batch.
- **ROC result:** SAE's AUC, best balanced accuracy, and FA-capped optima were
  identical. VAE AUC drifted by at most `0.0000217922` percentage points;
  best balanced accuracy and the best DR/FA at FA<=15% and <=15.5% remained
  identical. Literal batch-256 cutoff transfer can flip one boundary row, so
  the evidence supports near decision-invariance, not exact invariance for
  every threshold.
- **Current interpretation:** batch arithmetic does not explain or rescue the
  poor scores of these two saved pilots. The finding is sufficient to discuss
  a prospective decision-level feasibility rule, but the original all-score
  gate remains a preserved failure. No promotion follows automatically.
- **Limits:** the weights came from truncated two-epoch pilot fits. Their large
  gap from the paper target is context, not a Table-III numerical reproduction,
  mechanism result, plateau, all-weights limit, or general-population claim.
- **Integrity and stop:** both jobs completed `0:0`; source/input/artifact
  hashes, shapes, and finite checks passed. Total exposure was 227 GPU-seconds.
  Stop before gate replacement, promotion, retraining, FC-VAE, AEA, mechanism
  work, or publication.
- **Verification:** all 252 post-result repository tests (140 study and 112
  root) and strict data verification pass; original implementation and source
  record hashes remain exact.
- **Sources:**
  [frozen contract](../studies/atk-2022-deep-autoencoder/REMAINING_SCORE_RECOVERY.md),
  [finding](../studies/atk-2022-deep-autoencoder/REMAINING_SCORE_RECOVERY_FINDING.md),
  [execution record](../studies/atk-2022-deep-autoencoder/results/recurrent_score_recovery_20260901.json).

### Evidentiary language and LSTM-SAE anchor cost checkpoint — 2026-09-01

- **Language decision:** systematic non-reproduction, mechanism failure, and a
  stable finite attainability envelope may warrant “highly implausible within
  the declared envelope.” A fabrication claim additionally needs forensic
  evidence distinguishing deliberate reporting from omissions, leakage,
  metric/table mistakes, or transcription errors.
- **Existing operational cost:** A16 epochs 218/207 seconds on 32,768 rows and
  batch-256 scoring 3.913664639 seconds on 12,119 rows imply 42.7902 hours for
  ten full-data epochs and 417.1425 hours for 100 under the frozen 1.5-times
  projection. The existing 72-hour gate fails by a wide margin.
- **Smallest discriminating action:** one unchanged-method, single-H200,
  at-most-two-hour cost pilot. It does not alter batch, data, architecture,
  optimizer, seed, or GPU count. It produces operational `X`, never a paper
  metric.
- **Prospective boundary:** only an H200 worst-case 100-epoch projection <=72
  hours, finite updates/reload, <=75% memory, and the explicit adaptive
  decision-stability gates can promote one full LSTM-SAE anchor. Failure stops
  the sequence rather than converting a partial fit into a full result.
- **Sources:**
  [language decision](decisions/2026-09-01-implausibility-and-fabrication-boundary.md),
  [promotion contract](../studies/atk-2022-deep-autoencoder/LSTM_SAE_ANCHOR_PROMOTION.md),
  [cost record](../studies/atk-2022-deep-autoencoder/results/lstm_sae_anchor_cost_20260901.json).

### H200 cost pilot blocked before execution — 2026-09-01

- **Attempt:** exact frozen commit `93ecd0d` was synchronized to Panther and
  job `385602` requested one H200 for at most two hours.
- **Observed infrastructure state:** the account association exposes only QOS
  `gpulimit`; the H200 partition accepts dedicated H200 QOS values. Slurm held
  the job as not permitted. It was canceled with zero elapsed GPU time.
- **Scientific boundary:** no model was built, no epoch ran, and no score or
  experimental artifact exists. This is operational `X`, not `N`, `M`, or `A`
  evidence and not a stronger claim about the paper.
- **Decision:** the H200 promotion gates were not measured. Because the only
  measured A16 projection remains above 72 hours, the conditional full
  LSTM-SAE anchor is not eligible under the frozen plan.
- **Remaining choices:** obtain authorized faster hardware, change the declared
  compute ceiling, or define an explicitly partial run. Each would require a
  new prospective decision.
- **Sources:**
  [operational finding](../studies/atk-2022-deep-autoencoder/H200_COST_FINDING.md),
  [cost record](../studies/atk-2022-deep-autoencoder/results/lstm_sae_anchor_cost_20260901.json),
  [completed plan](plans/2026-09-01-lstm-sae-anchor-promotion.md).

## How to add a learning

Use: former belief/status; evidence; root cause if isolated; current conclusion
with label; remaining uncertainty; and source artifacts. Preserve invalidated
beliefs rather than deleting them.
