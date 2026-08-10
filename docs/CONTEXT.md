# ATK Evidence — Working Memory

**Last updated:** 2026-08-11

## Environment quirks

- Host is Apple M1 Max/macOS; public setup uses root `.venv` while the pre-publication workspace still has a legacy `replication/.venv`.
- Paper 1 neural runs use Keras 3 with the Torch backend and available Apple MPS; the paper does not state its backend, software versions, hardware, epochs, or batch size.
- Official CER/ISET consumption archives are restricted by ISSDA. Exact
  ScienceDB copies are local under `data/raw/cer-sciencedb/`; all six pass the
  official size/MD5 and ZIP gates. Its 6,445-row allocation CSV is not the
  official `.tab` binary but has zero normalized semantic mismatches against a
  second public allocation workbook and complete coverage of all residential
  reading IDs. Decision `2026-07-21-cer-sciencedb-semantic-allocation.md`
  admits it for the named exploratory branch only.
- Built-in macOS `unzip` failed on the multipart SGCC archive; 7-Zip 26.02 verified and extracted it successfully.
- In zsh, lowercase `path` is tied to `PATH`; never use it as a loop variable because system commands disappear for that shell.
- Cluster access is SSH-key-only and normally requires the institution's VPN
  off-site. Host, user, and project paths are deliberately not recorded in this
  public repository; keep them in local configuration.
- User execution policy (2026-07-24, widened 2026-07-24): run **every**
  experiment's preparation, training, and scoring on the cluster's compute nodes,
  never on the local Mac. This is not scoped to Paper 1 — it covers all
  studies and any private workstream. Local work is limited to code,
  documentation, lightweight inspection, and transfer/monitoring. Do not infer
  permission for a local fallback when the compute cluster is temporarily unreachable.
  Results produced locally are ineligible as experimental evidence and must be
  re-run on the compute cluster.
- The initial one-T4 batch-512 LSTM-SAE/LSTM-VAE OOMs and the cancelled V100
  attempt remain resource evidence. A one-T4 batch-32 run is a separately
  declared sensitivity, never a substitute for the primary batch-512 result.
- The primary-batch LSTM-AEA attention call receives two local
  `[128, 1034, 200]` tensors and attempts a 101.96-GiB allocation per rank;
  smaller unspecified batches remain an ambiguity branch rather than a silent
  replacement for the primary batch.
- The production neural runner is one Python program plus a short `sbatch`
  wrapper. Four-rank DDP preserves global sample-mean gradients and records its
  shuffle/random-stream choices; FC-SAE, LSTM-SAE, LSTM-VAE, and supervised
  feed-forward have now completed through that path.
- A Torch-native supervised probe's BCE assertion was an invalid diagnostic:
  the compiled-Keras loss/Adam rerun completed with synchronized finite state
  and identical final parameters across all four ranks.
- Cluster job 348195 completed and verified 12 classical Table II attempts.
  Mean DR was 7.97% NB, 2.10% ARIMA, 61.78% one-class SVM, and 53.51%
  multiclass SVM versus reported 75%, 88%, 91%, and 92%; all registered
  complete metric patterns are `NOT_CLOSE_MATCH` in this exploratory branch.
- The cluster's observed NVIDIA driver `570.133.07` cannot run the CUDA 13 package
  selected by the earlier floating Torch constraint. The cluster is pinned to the
  official PyTorch 2.7.1 CUDA 12.6 build and jobs fail on invisible CUDA.
- Even with identical SGCC/config/package versions and seeds, ADASYN produced
  77,708--77,712 supervised rows as OMP/MKL thread settings changed. The
  four-GPU cluster branch fixes both to 2 and records runtime cardinalities;
  this tiny variation is an ambiguity, not an explanation for large metric gaps.

- Never edit `config/exploratory_reproduction.toml` mid-branch: its byte hash
  is `contract_sha256` in every run fingerprint, so any edit invalidates
  resume-skip for all SGCC Table II attempts. Freeze ISET-phase additions
  separately in `config/exploratory_iset.toml`.
- The 2026-07-21 blanket fidelity audit is **INVALIDATED**. A fresh source-first
  audit on 2026-07-23 found concrete counterexamples: FC-SAE has seven rather
  than the paper-described eight hidden transformations; FC-VAE does not
  instantiate the printed four-plus-four hidden layout; ISET supervised data
  uses heldout-only attacks rather than malicious data for all customers; and
  the stated VAE reconstruction-probability detector is not implemented.
  Existing results are retained but gated by
  `PAPER_TO_CODE_TRACEABILITY.md`.
- Exact ISET preparation completed 2026-07-22 using all 4,225 residential
  meters. Cache SHA-256 is
  `ab88f180feafb7351ef4530cba2e48a3cbc180af268f8b68016aefc50b98a987`;
  Table IV subsets equal 30.603M/45.905M/61.206M scalar readings, strongly
  supporting the interpretation of the paper's 30M/45M/60M labels.
- LSTM-SAE seed 11 completed on four 16-GB V100s with 2:24:26 Slurm
  elapsed (2:24:03 pipeline; 2:20:38 fit): DR 6.78%, FA 2.22%, AUC
  51.89%. Its score is effectively
  input energy under zero reconstruction (correlation
  0.999999999999996), and even an oracle test threshold gives only 55.52%
  balanced accuracy. This is strong exploratory evidence against a mere
  threshold problem, not a final verdict.
- FC-VAE diagnostic job 354018 showed a finite first Adam step on all four
  ranks but extreme rank-local loss/gradient scales; the later failure remains
  unresolved.
- Public use now has four study-root commands: `download_data.py`,
  `prepare_data.py`, `run_experiment.py`, and `analyze_results.py`. Keep
  internal audit/tests behind this small interface rather than building
  another orchestration layer.
- Correction 2026-07-24: those four short commands are only wrappers over a
  21,414-line internal Python tree including tests. They are not the promised
  compact reference implementation. The target is a genuine five-file
  extraction (`download`, `prepare`, `models`, `run`, `analyze`) for one frozen
  source-faithful anchor, with the branch/evidence/DDP machinery retained
  separately as the forensic harness.
- Workflow reset 2026-07-24: `RUNBOOK.md` is now the canonical tutorial for
  every paper. The active Paper 1 route is a fresh PDF-derived `METHOD.md`,
  genuine five-file ISET implementation, tiny sanity run, then one full
  Table-III FC-SAE anchor before any additional infrastructure, publication,
  or exhaustive branch execution.
- Paper 1 source freeze completed 2026-07-24 at
  `studies/atk-2022-deep-autoencoder/METHOD.md` after fresh extraction and
  visual review of all 12 pages of PDF SHA-256 `f3098e...850f`. The declared
  first anchor is `P0-ISET-FCSAE`: all named residential meters; strict
  48-slot days; all-customer six-attack `M`; joint pre-split feature scaling;
  customer-disjoint B1/B2; printed test-set ADASYN; the full
  `48-400-300-200-100-100-200-300-400-48` sigmoid/Softmax FC-SAE; and printed
  threshold 0.58. Batch/epoch/convergence/seed choices are visibly labeled
  execution completions, not paper facts.
- Independent source re-audit 2026-08-11: the exact PDF hash and overall
  `METHOD.md` flow were reconfirmed after all 12 pages were visually inspected
  before opening the old reconstruction. Corrections/additions: Tables II/III
  have six, not seven, benchmark rows; VAE Eq. (9) mixes incompatible
  distributions/variables; VAE variance positivity and Algorithm-5 decoder
  input are undefined; precision prose conflicts with its formula; Table-II
  Naive Bayes F1 is arithmetically inconsistent; and neither Table II nor III
  admits one common prevalence from its DR/FA/PR rows under generous rounding.
  `P0` is explicitly a paper-primary `P+I` executable completion because
  printed Attack 3 is non-executable. The renewed source-freeze checkpoint is
  the next gate; do not audit code or resume compute before it is accepted.
- The fresh source pass independently reconfirmed pivotal non-uniqueness:
  Eq. (3)'s endpoint is impossible; “rows (customers),” all-customer `M`, and
  unseen-test-customer claims conflict; Algorithm 6 cannot calculate DR/FA
  from benign-only `X_TR` and its scalar width/layer loops cannot yield all
  Table-I layouts as written; Fig. 3's distinct latent width is absent; and a
  common Table-V model/common benign set mathematically requires invariant FA
  although the table reports attack-varying FA. CHECKPOINT 1 is now the only
  gate before the compact five-file implementation.
- CHECKPOINT 1 was approved 2026-07-24. The genuine compact implementation now
  exists at `studies/atk-2022-deep-autoencoder/reproduction/`: five direct
  files and 1,617 total lines, with no imports from the forensic `src/` tree.
  On real CER rows, `p0-tiny-v1` completed source verification, preparation,
  two FC-SAE epochs, scoring, metrics, baselines, Table V, and aggregation.
  The runtime FC-SAE has all printed widths
  `400,300,200,100|100,200,300,400` and 450,448 parameters. Tiny metrics are
  fixture-only (ACC 42.18%, AUC 42.23%); all six Table-V FA values are exactly
  65%, confirming the fixed-model/common-benign invariant. The next action is
  a new full P0 cache from raw verified archives, not the historical cache.
- The fresh full compact preparation materialized 2,251,290 strict benign
  profiles, 13,507,740 generated attack profiles, 1,500,520 customer-disjoint
  training profiles, and the exact 14,258,510-row printed `B2+M` population.
  Applying imbalanced-learn's default ADASYN to that population is not an
  ordinary preprocessing wait: with 48 features, sklearn selects brute-force
  neighbors and its first call entails about
  `750,770 × 14,258,510 = 10.7e12` profile-distance comparisons, followed by a
  second minority-only search. Preserve this as an exact-default
  executability result. The interrupted default call consumed 4,724.52 seconds
  wall time and 33,665.16 CPU-seconds (9.35 CPU-hours) without completing its
  first `kneighbors` call or producing `x_test.npy`. Run the
  no-test-resampling interpretation explicitly
  as `I-ADASYN-NONE`, then a separately labeled scalable ADASYN sensitivity;
  never call either one the completed exact-default P0 cache.
- `prepare_data.py` now exposes opt-in paper-interpretation/corrected policies
  without changing historical defaults: four scaling scopes, printed versus
  absent anomaly-test ADASYN, pre-split versus training-only supervised
  ADASYN, and ISET all-customer versus B2-only malicious populations. Fixture
  tests prove corrected test sets contain no synthetic rows and supervised
  original customer/meter identities are disjoint. No full cache has yet been
  rebuilt under these new policies.
- SGCC preparation and both ordinary/DDP runners now expose all six frozen
  resolutions of 1,034 raw days versus 48 model inputs, all four missing-data
  readings, and customer-disjoint versus row-random sample splitting. Window
  IDs retain their source customer; fixture tests prove the customer-disjoint
  branch is disjoint and the row-random branch is not. These semantics are
  included in run fingerprints. Full rolling windows are structurally
  executable but must enter the frozen screening path rather than be mistaken
  for the historical full-vector result.
- Source-v2 recurrent builders execute both input layouts, both state-transfer
  policies, and repeat/first-step/autoregressive decoder schedules for
  LSTM-SAE/LSTM-VAE. A dedicated Algorithm-5 AEA decoder now feeds the prior
  scalar reconstruction back at every time step while recomputing attention
  from the prior decoder state; it executes concatenate/literal-sum merges,
  mirrored/top-only states, both input layouts, and both latent placements.
- The VAE runner now supports all frozen score IDs. Fixed-variance and learned
  decoder-variance-head branches calculate Monte Carlo multivariate Gaussian
  reconstruction density for 1/10/100 draws; raw probability is explicitly
  lower-is-anomaly, while MSE and MSE+KL remain higher-is-anomaly surrogate
  branches. Existing VAE results are still implementation-v1 surrogates.
- The ordinary runner now executes supplied printed constants and all three
  deterministic repairs of “median of IQR of ROC,” independently crossed with
  ISET-transfer/dataset-specific scope and B1-generated-attack,
  B2-validation-carve-out, or no-derivation populations. B2 validation rows
  are removed by identity from final test; SGCC transfer requires a frozen
  ISET threshold artifact. Fixed epochs, holdout/no-refit, holdout/all-B1
  refit, and five-fold-or-maximum-feasible cross-validation/all-B1 refit also
  change actual fit behavior and retain every history/timing. The DDP runner
  now has matching validation/refit/threshold semantics. Stable branch IDs
  resolve through the public runner into data, model, classical, validation,
  threshold, and Table-V arguments; preparation IDs are content-addressed and
  checked against cache metadata.
- ISET preparation now executes all registered Attack-1 factor scopes,
  Attack-2 half-hour/hour-pair granularities, all three minimal repairs of the
  non-executable Attack-3 interval, both hour-to-slot mappings, and all-4,225
  versus deterministic seeded-3,000 residential populations. The printed
  Attack-3 subtraction remains a non-executable evidence node. Attack
  regeneration is also explicit: fixed per data seed, per model seed, or per
  experiment index, with the resolved seed derivation stored in cache metadata.
- CER archive extraction now executes strict 1--48 days, trimming slots 49/50,
  duplicate-slot averaging, and 48-grid interpolation. ISET preparation also
  executes customer-disjoint and row/profile-random splitting; heldout attacks
  are bound by source-profile identity so row-random meters may overlap without
  mixing the actual profile rows. The existing 3.2-GiB cache remains the
  strict-day/customer-disjoint implementation-v1 artifact.
- Exact-ISET execution now uses that same `run_experiment.py` interface:
  `--dataset iset --table 3` trains Table III and derives Table V from the same
  persisted score vector; `--table 4 --sizes ...` trains the nested size cells.
  The 3.2-GiB cache reverified and loaded in 13.65 seconds locally, and an
  actual one-epoch FC-SAE fixture fit completed end to end. That historical
  cache is implementation-v1 and cannot be relabeled for source-v2 branches.
  The next gate is the content-addressed exact-ISET cache build followed by one
  real DDP smoke on the compute cluster, not a replacement matrix.
- The source-first visual map is the self-contained
  `site/papers/atk-2022-deep-autoencoder/index.html`, with a pointer at
  `studies/atk-2022-deep-autoencoder/PAPER_WORKFLOW.md`. It is a readable
  paper-order explanation, not an embedded Mermaid graph or a display of the
  internal branch combinatorics.
- Bounded local Gate-D evidence is
  `studies/atk-2022-deep-autoencoder/results/gate_d_bounded_sanity_20260724.json`:
  137 study plus 10 project tests pass; the real SGCC printed-anchor preflight
  is ready; the exact-ISET branch correctly stops at its missing
  content-addressed cache.
- Direct `--table 5` execution now covers common model/common benign,
  per-attack retraining, per-attack seeded benign resampling, and both,
  crossed with full-heldout/seeded-3,000 sizes. It persists all six score and
  identity sets and honors lower-is-anomalous VAE probability. The historical
  Table-III-coupled derivation remains only the common/fixed structural
  diagnostic.
- Eq. (10) visually prints squared L2 plus KL. Source-v2 FC/LSTM VAE builders
  now execute both `sum_squared_plus_kl` and the common
  `mean_mse_plus_kl` reading. With zero KL, their reconstruction terms differ
  by exactly the input dimensionality. Learned-decoder-variance branches use
  the analogous summed/mean Gaussian data term so the variance head receives
  gradients; that likelihood loss is a documented prose-consistent
  completion, while fixed variance plus summed squared error is the direct
  printed-loss branch.
- Exact-ISET seed 11 implementation-v1 cells (quarantined as reproduction
  evidence): FC-SAE full DR 22.50%, FA 37.13%,
  ACC 42.69%, AUC 42.59% (paper 81/15/83/81); FC-SAE half ACC 42.68%
  (paper 70); FC-VAE full 40.43/53.86/43.28/40.82%
  (paper 88/11/88.5/85). FC-SAE score correlates 0.99945 with input energy;
  even reversed test-label oracle ACC is only 57.72% on primary rows. These
  are strong one-seed exploratory non-reproductions, not a final verdict.

## Working patterns

- Run deterministic tests with `bash scripts/test.sh`; it supports the root environment and the legacy local environment.
- Preserve raw files in place and identify them by checksum; study artifacts belong under `studies/<study-id>/results/`.

## Don't repeat

- Do not substitute 48-day SGCC windows for the paper's 48 half-hour CER profiles when assessing the primary reproduction hypothesis.
- Do not let literature provenance work displace the exact-data, paper-literal reproduction task.
- Do not silently correct the paper in the primary track; corrections belong in a separately labeled controlled analysis.
- Do not report a best/lucky seed as a reproduced result.
- Do not search Keychain metadata, browser storage, shell history, credential files, or broad home-directory locations for dataset access. A broad credential audit likely triggered workplace endpoint protection on 2026-07-21. Restrict work to the project, explicit dataset locations, and user-supplied authorization.

## Open questions

- The design for the exhaustive confirmatory phase is written and visible at
  `docs/plans/2026-07-22-confirmatory-branch-sweep-design.md` (anchored branch
  enumeration, AUC screening funnel, controls, mechanism demonstration). It is
  a draft: freeze checklist + user CHECKPOINT required before any execution.
- Exact membership and order of the three-paper core corpus after Paper 1.
- Reproduction tolerances, finite hyperparameter envelope, seed count, split policy, and computational stopping rule must be frozen before confirmatory runs.
- Exact ScienceDB CER archives and the named semantic allocation branch have
  passed the implemented code gate, preparation, execution preflight, and
  bounded model smoke. Tables III--V now need the compute cluster result cells.
- Public canonical repository: <https://github.com/fjoad/atk-evidence>, default branch `main`.
- The user explicitly authorized an end-to-end exploratory reconstruction of Paper 1 Tables I-V on 2026-07-21, including timed runs and documented author-intent assumptions. It is not retrospectively preregistered confirmatory evidence.
- Exact ISET execution requires seven restricted official files: six consumption archives plus the SME/residential allocation file; login alone does not grant access.
- The post-verdict controlled solution is specified at
  `docs/plans/2026-07-23-paper-1-controlled-solution.md`. It is deliberately
  gated until Tables I--V, confirmatory assessment, and the Paper 1 LaTeX
  verdict are frozen.

## User emphases

- 2026-08-11 baseline-first execution: before testing alternative assumptions,
  run one frozen straight-through Paper-1 anchor end to end. The current runnable
  lane is full ISET, FC-SAE, seed 11, batch 512, original `B2+M`, producing the
  Table III row, full-data Table IV cell/timing, and Table V attack rows from one
  model. It is explicitly `I-ADASYN-NONE`, not completed printed `P0`; the
  multi-trillion-pair default test-ADASYN failure remains visible. Do not start
  an ambiguity sweep before inspecting this result.
- 2026-08-11 non-executable reporting rule: every paper statement that cannot
  exist or run must be visibly identified on the study site and in the LaTeX
  report. Preserve the literal failure, predeclare all materially reasonable
  executable repairs, run them under separate `I` IDs, and show every result
  beside the reported target. A repair is never relabeled as the literal method;
  a matching repair must be reported as readily as a non-match.
- 2026-08-09 reframe: extracting the paper's actual method is the most rigorous
  and reasoning-intensive part of every study. Implementation should then be a
  small transparent transcription. Keep global infrastructure minimal, prove the
  measuring path cheaply, run one watched full anchor, and only then scale. The
  shared Charter and human documentation must preserve this across compaction;
  see `docs/decisions/2026-08-09-paper-first-minimal-instrument.md`.

- The honest hypothesis is that the reported numbers will not be reproducible from the papers as written, but the project must be genuinely open to being wrong.
- Exact paper-described algorithms and procedures are the highest priority; add nothing extra to the primary track.
- 2026-07-23: a prior contract, passing test, or registered ambiguity is never
  sufficient proof of fidelity. Before compute, reconstruct the method from the
  PDF alone and require claim-to-code-to-cache traceability; quarantine results
  immediately when that chain fails.
- 2026-07-23 visual architecture correction: Table I/Section IV-C require all
  encoder hidden widths and the full mirror, which invalidates
  implementation-v1 FC-SAE and FC-VAE. Figs./prose depict latent layers, while
  Algorithms 2/5 directly reuse the terminal encoder state or attention
  context; LSTM-SAE and AEA therefore remain quarantined algorithm-literal
  branches, with distinct-projection branches also required. LSTM-VAE's hidden
  structure is aligned, but latent width and probability score remain
  unresolved.
- 2026-07-23: the user requires three separate families for every material
  issue: the printed method even when statistically wrong, every defensible
  interpretation of ambiguous/contradictory text, and the scientifically
  corrected method. “All” means a documented finite coverage closure against
  every material PDF statement and omission, not an unbounded claim over
  imaginable code.
- 2026-07-23 standing authorization: continue implementing that approved
  `P`/`I`/`C` mandate autonomously through safe structural and fixture-test
  gates. Do not repeatedly ask permission for each ambiguity branch. Stop only
  at a declared compute/freeze checkpoint, an external blocker, or an action
  requiring materially new authority.
- Paper 1 branch-lattice v1 is machine-readable at
  `config/branch_lattice.toml`: 22 model/data families, 22 printed anchors, 899
  interpretive configurations, 22 separate corrected controls, and 2,763
  three-seed screening attempts. Point screening estimate is 558.7 GPU-hours
  plus 57.4 CPU-hours, or 49.7 ideal active GPU-job hours at the three-job cap
  (99.4 h under the 2x runtime factor), excluding queue time. The
  52.57-billion arbitrary Cartesian product is explicitly excluded; every
  option and compatible option pair is verified, and all 36 ambiguity-register
  rows have machine-checked coverage references. Threshold formula and
  threshold scope are independent dimensions, while impossible formula/label
  and printed-constant/dataset-specific combinations are machine-excluded.
  Algorithm 6 now has three
  explicit branches: literal uniform-width search (36 evaluations), per-layer
  coordinate search capable of unequal widths (86), and direct Table-I replay;
  the earlier eight-evaluation budget had no paper basis and was removed.
- 2026-07-24: SGCC has no six attack-type labels, so the seven-class
  “multiclass SVM” reading applies only to ISET. The SGCC family remains binary
  and fails loudly if seven-class labels are requested. This source-bound
  correction was frozen before replacement results; see
  `docs/decisions/2026-07-24-sgcc-multiclass-label-scope.md`.
- Ambiguities may use reasonable assumptions only with complete documentation.
- Evidence must come from rigorous experiments and statistical assessment, aiming for the strongest defensible conclusion.
- Deliver one LaTeX-style rebuttal/reproduction report per paper and a combined report.
- After Paper 1 reconstruction and analysis are finished, separately design
  the method we would actually use to solve electricity-theft detection,
  explain and isolate why the literal method fails, and test whether the new
  method honestly exceeds the published results. Do not begin this early or
  blur it into reproduction evidence.
- Claims should be supported by independently rerunnable proof-quality evidence; data acquisition and setup must be explicit from a fresh public clone.
- Keep cluster orchestration minimal: short
  `sbatch` wrappers around the Python programs, without a separate manifest,
  probe framework, packed-worker layer, or automatic scheduler.
- Keep the public explanation and reference implementation proportionate. The
  exhaustive audit harness may remain large, but it must not be confused with
  the amount of code required to implement one paper-defined experiment.
- 2026-07-24: the user explicitly identified that project effort had become
  inverted—implementation infrastructure and documentation dominated while
  eligible experiments lagged. For every paper, reading/source freeze should
  dominate reasoning and experiments should dominate elapsed time. Stop if
  code/framework growth is delaying the first eligible full result.
- Publish one GitHub Pages site for the repository, with one path and one
  scientific PDF per paper plus a later synthesis. The public site must deploy
  from a dedicated static directory so internal docs, local data, and paper
  PDFs are not exposed.
- 2026-07-22: user confirmed batch-512 stays the frozen primary (stated motive:
  GPU utilization); batch 32 remains the declared sensitivity. LSTM-AEA is
  planned under the batch-32 branch via DDP (local batch 8 fits ~6.4 GiB).
  `run_model_ddp.sbatch` now accepts an optional config plus direct runner
  arguments, but no replacement compute is authorized before Gate C closes.
- the compute cluster 2026-07-23: no queued or running project jobs. Successful Table II
  cells = 20/33 (classical 12, FC-SAE 3, supervised feed-forward 3, LSTM-VAE
  1, LSTM-SAE 1); FC-VAE failed 3 and 10 cells are unrun. At the user's
  direction, the single-T4 batch-32 LSTM-SAE sensitivity was cancelled after
  23:27:30; it is resource evidence only.

## When to update this file

Update inline when a non-obvious environment fact, failed path, user emphasis,
or active decision appears. Keep entries terse and prune beyond roughly 200
lines. Promote durable causal corrections to `EVIDENCE-AND-LEARNINGS.md`.
