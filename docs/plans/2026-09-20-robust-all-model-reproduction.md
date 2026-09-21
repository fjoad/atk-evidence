# Robust poisoning: fresh reproduction of every reported model

**Date:** 2026-09-20

**State:** preparation, three shallow-model pairs, forest controls, and read-only SVM diagnostic complete and audited

## Current execution: feed-forward pilot

User approved the next coverage step. Recheck the paper's six 500-neuron
hidden layers, ReLU/Sigmoid, Adamax, constraint3, no dropout, 50 epochs and
batch 100. Freeze an explicitly repaired binary-cross-entropy completion,
optimizer/initializer/constraint defaults, and the original p00/p30 inputs.

- [x] Freeze source settings, omissions, software versions, GPU preflight,
  metrics, budget and stopping in FEED_FORWARD_PILOT.md.
- [x] Add the neural builder/runner to the direct implementation; verify
  the literal loss defect and repaired learning on constructed examples.
- [ ] Establish an isolated TensorFlow environment and verify one allocated
  GPU with a constructed update before loading research data.
- [ ] Freeze and run only the two approved 50-epoch pilot fits, one seed,
  within one 20-minute GPU job; preserve partial/failed attempts.
- [ ] Audit weights, scores, histories, paired initialization and data,
  record the outcome and next question in the journal/site, and commit.

Environment installation has a separate bounded CPU setup allocation and
does not count as a paper experiment. No full-data run, extra seed, alternate
loss training branch, or SVM parameter search is included.

Local implementation and tests are complete: 327 main-suite tests pass with
10 environment-specific skips; all 8 neural fixtures and 7 AdaBoost fixtures
pass in their isolated environments. The old SVM diagnostic audit still
matches. CPU dependency-setup job 398992 was last observed running at 8:36
while downloading NVIDIA libraries. The QCRI VPN then disconnected, DNS
resolution failed, and direct-IP TCP/SSH checks timed out. Setup's final state
is unknown; do not submit it again without checking Slurm/logs on reconnect.
No research-data feed-forward fit or GPU job has been submitted. Resume the
approved bounded pair only after verifying setup and the GPU preflight.

## Current execution: read-only SVM replay and kernel diagnostic

User approved this follow-up. No model fitting, parameter alternative, or new
data preparation. Independently reconstruct both saved models' train/test
decision scores, then inspect one outcome-independent 512-row training kernel.

- [x] Freeze input/model identities, subset rule, numerical tolerances,
  spectral checks, interpretations, and finite budget in SVM_REPLAY.md.
- [x] Implement a direct read-only check and constructed-fixture tests;
  preserve all original scientific files and fitted outputs.
- [x] Freeze and run one CPU job, 10 minutes/1 CPU/8 GiB/no GPU, zero fits.
- [x] Verify saved diagnostic artifacts, record supportive or adverse results
  and remaining limits, update journal/site/handoff records, and commit.

Outcome: 4665e07, job 398978, completed 0:0 in 32 seconds; zero experimental
fits. Score reconstruction agrees within 1.68e-12 with all labels unchanged.
Fixed 512-row kernel min=-23.45; centered min=-19.97, far below the declared
tolerances. The latter witnesses equality-constrained negative curvature,
not fitted-solution suboptimality or caused performance loss. All artifacts
and original-file nonmutation checks pass. Actual Slurm allocation 2 logical
CPUs despite 1 requested, with one numerical thread; 8 GiB/no GPU.
Stop. Proposed next coverage step is the feed-forward baseline, with its
cross-entropy repair and other completions frozen first. Keep a finite SVM
parameter-sensitivity question open; no further fits are authorized here.

## Current execution: first SVM pair, September 21

User approved the next step. Freeze C=1 and the printed sigmoid kernel with
explicit completions for gamma, coefficient, stopping, and decision scores.
Use the same original p00/p30 prepared arrays, one seed, two fits, one CPU job
capped at 15 minutes/4 CPUs/16 GiB/no GPU. Do not add probability calibration,
kernel changes, corrected preparation, or repeated fits after seeing outcomes.

- [x] Record the source settings and omissions, score semantics, targets,
  software version, diagnostics, and stopping in SVM_PILOT.md.
- [x] Extend the direct implementation and prove stock-library parity,
  corrupted-label use, raw-margin metrics/ties, convergence reporting, and
  exact save/reload on constructed software fixtures.
- [x] Freeze and run the bounded pair on Panther, preserving all attempts.
- [x] Recheck artifacts and historical audits; record the complete outcome,
  next justified question, journal/site draft, and handoff state; commit.

Outcome: e698173, job 398709, completed 0:0 in 2 minutes. Both fits and all
artifact checks passed, but AUC 65.64/63.38 is weak; all cutoffs/reversals
miss the corresponding printed operating corners. Best DR 19.37/33.48 at
FA caps 10.2/25.7%, versus 89.2/73.7. Weak training accuracy remains despite
solver success. Stop this pair. Proposed next step: freeze a read-only
support-vector score replay and fixed-subset sigmoid-kernel diagnostic,
including numerical tolerances and a bounded compute allocation. This has not
run and does not authorize another fit or parameter sweep. See the journal
and results/svm_pilot_20260921.

## Current execution: first AdaBoost pair

User approved the proposed next model. Run one 0%/30% pair on the unchanged
original generalized two-class pilot. This asks whether the next shallow
baseline learns useful discrimination and whether poisoning changes ranking,
default decisions, or both. It is not full-data reproduction.

- [x] Freeze paper settings, historical-library completion, targets, and budget.
- [x] Add AdaBoost to the direct implementation, prove it on constructed
  inputs in its pinned environment, and keep historical forest audits valid.
- [x] Run two fits, one seed, one CPU job: 15 minutes, four CPUs, 16 GiB,
  no GPU. No new preparation, seed search, or alternate algorithm fit.
- [x] Audit all artifacts, compare all metrics and fixed-score cutoffs, update
  the journal/site draft and handoff records, and commit. Do not publish.

See ADABOOST_PILOT.md. Historical scientific revisions remain immutable in
Git; additive implementation updates must validate old source hashes against
each recorded commit, never rewrite old results to match today's files.

Outcome: frozen d47a6de, job 398348, completed 0:0 in 14 seconds. Both
50-tree fits passed persistence, input/output, and metric audits. AUC is
90.70692/83.73956; default DR 81.08597/46.42534. At FA<=29.9%, poisoned
DR can reach 80.45249, so the default gap is not wholesale loss of ranking.
Pre-split dependence and missing source choices remain. Stop this pair.
The next proposed model is SVM (C=1, sigmoid), after freezing gamma,
coefficient, and score choices. See results/adaboost_pilot_20260920 and the
journal. No subsequent model, seed, or alternate AdaBoost algorithm ran.

## Current execution: matched split/resampling check

The user authorized the proposed follow-up. Keep the existing random-forest
fits as reference A. Add B with the same original train/test identities but
ADASYN applied to training only. Add C with whole source days held out and
training-only ADASYN. Compare all three on the same predeclared intersection
of original test rows; also compare A/B on the complete original test set.

- [x] Freeze the evaluation population rule, controlled interventions,
  unchanged model/seed, counts to report, and finite budget.
- [x] Implement and fixture-test identity matching, ancestry checks, and
  source-day exclusion without altering the original preparation or fits.
- [x] Run exactly four new fits (B/C at 0%/30%) in one CPU job capped at
  15 minutes, four CPUs, 16 GiB, no GPU.
- [x] Audit the outputs, record all outcomes and remaining limitations, and
  update the journal before choosing another experiment.

See SPLIT_RESAMPLING_CHECK.md for the pre-outcome contract. This is a small
controlled diagnostic, not full paper reproduction or a seed sweep.

Outcome: frozen e6e0359 passed 304 tests; job 398164 completed 0:0 in 1:48.
All 112 input arrays and saved outputs passed cluster and local audits. On the
same 445 original rows, training-only synthesis lowers AUC by 6.47/8.99
points at p00/p30. Grouped days cause no further collapse (C AUC 93.18/82.97).
This supports a preparation-policy effect, not universal baseline failure.
Stop this diagnostic. Proposed next distinct question: an explicitly specified
AdaBoost pair on the original verified pilot, with library/version choices
resolved before execution. No further model was launched. See the journal
and results/split_control_20260920.

## Current execution: first baseline

The user has authorized the next steps. Implement the 100-tree random forest
and direct training/analysis files, then run one bounded pair on the existing
generalized two-class 0%/30% pilot inputs. The question is whether the faithful
library model learns, saves/reloads correctly, and produces interpretable
metrics on the verified pipeline. This is not full Table III reproduction.

- [x] Record exact estimator defaults, metrics, controls, saved-score checks,
  source identities, and the finite allocation before fitting.
- [x] Implement and fixture-test the model, runner, and analysis.
- [x] Freeze and run one CPU job: two RF fits, one seed, unchanged pilot data;
  at most 15 minutes, four CPUs, 16 GiB, no GPU.
- [x] Verify predictions and artifacts, inspect the result, and update the
  journal and explanation register before choosing further runs.

No seed sweep, data regeneration, full-data run, or additional model family
is included in this first baseline check. A match or stronger performance
must be reported plainly alongside failures.

Outcome: frozen 48e979a, job 397217, completed 0:0 in 16 seconds. The pilot
shows strong random-forest discrimination and a material cutoff effect under
poisoning. A matched split/resampling check is the named next diagnostic;
another seed of the same construction does not resolve that question. See
the journal and the preserved results/rf_pilot_20260920 record.

## Current execution: steps 1 and 2

The user has authorized source/data resolution and implementation of the shared
preparation pipeline. Work now covers:

- [x] Independently verify online dataset identity, the locally available raw
  bytes, documentation, and cited attack definitions.
- [x] Record one explicit initial setup and all material unresolved alternatives
  before preparing research data.
- [x] Implement acquisition/verification and preparation for novelty and
  two-class models with customer/day/attack/poison/synthetic provenance.
- [x] Check parsing, attacks, splitting, scaling, ADASYN, and poisoning on
  hand-checkable local software fixtures, including direct library parity.
- [x] Run a bounded preparation-only check on Panther if authentication is
  available; otherwise report precisely that research-data execution is pending.
- [x] Update the journal, generated website, status, and checks; commit.

No detector fitting, score comparison, seed sweep, or full ADASYN allocation
is included in these two steps. Local tests use constructed software fixtures;
all real-data preparation remains on cluster compute nodes. Password login
subsequently succeeded. Frozen commit 30ce6c4 ran as CPU job 397206 and finished
0:0 in 2:16. Twenty customers and 560 daily profiles produced eight completed
preparation cases; all 224 saved arrays passed independent artifact checks.
See the journal and results/preparation_20260920 under this study.

## Direction

The user requests a fresh investigation of the paper's actual experiments,
covering all baseline and proposed models. Earlier workflows and conclusions
are historical evidence, not prerequisites or assumed outcomes. The priority
is a correct reconstruction of the data and ordinary library implementations
with the reported settings. Do not investigate speculative number-generation
patterns, author intent, or invented-data scenarios in this phase.

This plan replaces the completed September 2 source-only plan as Paper 3's
current direction. The initial turn established the plan; the user subsequently
authorized steps 1-2, whose completed preparation check is recorded above.
Existing results and source freezes remain preserved. Paper 1's
experiments and any work in its separate task are outside this plan.

## Source and complete coverage

The complete ten-page PDF was freshly read and visually inspected on September
20: *Robust Electricity Theft Detection Against Data Poisoning Attacks in
Smart Grids*, DOI `10.1109/TSG.2020.3047864`, SHA-256
`03a372fb5ee129799d43a50e6bec8f86fb108e9f8bdc85350e3ffe3e13d4a4ff`.

| Reported model | Initial implementation route | Specified settings to preserve |
|---|---|---|
| Random forest | scikit-learn | 100 estimators |
| AdaBoost | scikit-learn | 50 estimators |
| SVM | scikit-learn | C=1, sigmoid kernel |
| ARIMA | statsmodels, with forecast definition resolved from the cited method | MSE score, cutoff 0.58; order and fitting unit need resolution |
| Feed-forward | Keras with TensorFlow backend | 6 hidden layers of 500; Adamax; ReLU/Sigmoid; constraint 3; no dropout |
| GRU | Keras with TensorFlow backend | 8 layers of 300; Adam; ReLU/Softmax; constraint 5; dropout 0.2 |
| Attention autoencoder (AEA) | Keras recurrent layers with source-defined attention/decoder | encoder 500/300/200, decoder 200/300/500; SGD; Sigmoid; constraint 1 |
| Ensemble averaging | combine the three deep components as specified | outputs averaged into a fully connected decision layer; training details need resolution |
| Sequential ensemble | AEA -> 8x300 GRU -> 500-neuron dense -> output | Adam; ReLU/Sigmoid; constraint 1; no dropout |

These are intended implementation choices; only Keras is explicitly named by
the paper. Pin package versions and disclose defaults, especially historical
AdaBoost behavior, recurrent conventions, and optimizer parameters. The paper's
attention graph may require a small custom Keras layer; do not force a
different graph solely to use Sequential or an existing layer.

Full coverage means Tables III and IV (seven baselines, generalized and
customer-specific) and V (AEA plus both ensembles), at 0/10/20/30% poisoning.
Testing only the proposed detector does not satisfy this scope.

## Sequence

0. **Check for released code first.** Search the paper, publisher record,
   author publication pages, GitHub, and relevant artifact repositories using
   the exact title and DOI. Record queries, checked links, candidate identity,
   and access limits. If code is located, compare it with the paper before
   running it. A search that finds nothing establishes only that no public
   implementation was located in the checked sources.
1. **Resolve the shared setup.** Verify ISET source bytes, the residential
   population, daily profiles, attack functions and their parameters, scaling,
   split identities, ADASYN placement, and poisoning definitions. The paper
   does not identify its 3,000 customers. Record that omission and a justified,
   outcome-independent selection if exact identities cannot be recovered;
   distinguish source-dataset fidelity from exact-sample fidelity. Do not
   select a subset by how close its scores are to the paper.
2. **Make the setup auditable.** Preserve original customer/day identities,
   attack siblings, synthetic-sample provenance where available, and both true
   and corrupted training labels. Distinguish customer-based poisoning from
   sample-based poisoning and keep test labels unchanged. Each model follows
   its source-prescribed novelty or two-class preparation; controls can later
   evaluate a common untouched population.
3. **Build ordinary implementations.** Use the printed selected settings
   first. Check shapes, layers, labels, scaling, losses, finite updates, tiny
   overfitting, positive controls, and confusion-count calculations. Record
   necessary completions before performance is inspected. Resolve ARIMA and
   attention details using the cited sources before declaring the setup exact.
4. **Measure cost on Panther.** Use a short, recorded compute-node pilot for
   each materially different execution path. Confirm actual GPU, backend,
   batch 100, CPU/RAM, peak memory, and time per epoch/fit. A software failure
   or slow implementation is a problem to diagnose, not a negative paper
   result. Fix implementation bugs while preserving failed attempts.
5. **Reproduce generalized Tables III and V.** Run every listed model at the
   four poisoning levels under frozen setup and initial seeds, beginning with
   the simpler baselines so pipeline defects are found early. Record all seven
   metrics, raw scores, loss curves, confusion counts, model settings, and
   preparation/fit/scoring times. Independent jobs may run concurrently within
   a declared total allocation; retain one GPU per neural fit.
6. **Cover customer-specific Table IV.** Time a small representative batch of
   customers before allocating the full workload. Seven models x four levels
   x 3,000 customers is 84,000 fits per complete repetition before tuning.
   Keep this table in scope, cost it explicitly, and label any interim customer
   sample as partial coverage. Preserve individual-customer metrics and the
   exact averaging rule.
7. **Investigate substantial gaps.** First audit the shared pipeline and
   standard defaults. Then test a short list of credible repairs or omitted
   choices that could explain each gap. Inspect all score thresholds and
   per-attack results. Repeat informative configurations with paired splits
   and poison realizations; define the statistical unit, repetitions,
   uncertainty procedure, total budget, and stopping rule before confirmation.
   Investigate learning or capacity plateaus only when measured trajectories
   support that question. Preserve any reproduced or stronger-than-published
   results as plainly as failures.

## Choose extra runs by the question they resolve

A large mismatch does not automatically trigger three more seeds, a wider
grid, a longer fit, or another model family. A hypothetical 30% result against
90% reported first triggers diagnosis. This is an operating example, not a
measured result for this paper.

1. Inspect preserved inputs, labels, splits, scaling, metric definitions,
   confusion counts, and score direction. Confirm actual training updates and
   loss behavior; compare with simple and positive controls. Look for a shared
   pipeline defect before reproducing it across more expensive runs.
2. Name the competing explanations and choose the cheapest check whose outcomes
   would distinguish them. A fixed-score threshold enumeration may settle a
   cutoff question; a label/gradient fixture may settle a training-code defect.
   A failure common to many models warrants a shared-setup check first.
3. Before an additional experiment, record its question, competing predictions,
   the observation that would change the next decision, maximum cost, and stop
   rule in the journal. Skip a run if neither outcome would change the next
   action or the supported conclusion. Preserve unresolved uncertainty rather
   than extending the search indefinitely.
4. Repetitions serve a named purpose: quantifying run-to-run variability,
   checking a credible instability explanation, estimating a matched effect,
   or testing a predeclared statistical claim. A large gap alone does not
   prove that seed variation is small; conversely, more seeds do not repair a
   wrong setup or make a narrow implementation representative of all methods.

All reported models and poisoning levels remain in scope. Planned coverage is
distinct from automatic retries of a failing cell. Existing setup checks and
operational failures must be understood before scaling the next affected runs.

Documentation follows each material decision in the same change; avoid
turning a routine step into a new framework or a separate lengthy report.

## September 20 documentation and website follow-through

- [x] Complete and save the code-availability search.
- [x] Click through the live GitHub Pages site and record the reader journey,
  current/legacy page boundaries, and problems relevant to the requested rewrite.
- [x] Maintain a dated study journal with source locations, assumptions,
  questions, checks, outcomes, and the next concrete step; include Equation (1).
- [x] Save a website writing brief for an undergraduate taking introductory
  statistics and a plain-language draft explaining the current source issues.
- [x] Build and visually verify a local journal page from the study log,
  preserving dated questions, observations, corrections, and unrun checks.
- [x] Link these records from current status and commit the documentation.

The user's follow-up requests the research-log format both internally and on
the website. Build a local journal page for this paper and link it from the
study index as the first concrete example. The broader site overhaul remains
later work. Keep writing alongside research so the explanation is ready when
its supporting evidence is ready. This work does not deploy the local draft.

Completed records: `studies/takiddin-2021-robust-poisoning/RESEARCH_LOG.md`,
`CODE_AVAILABILITY.md`, and `docs/WEBSITE-JOURNAL-BRIEF.md`. The journal's
generated page is under `site/papers/takiddin-2021-robust-poisoning/` and its
renderer has a `--check` mode in the normal repository test command. The live
site walkthrough and local draft navigation were checked in the browser;
25 existing site/registry checks passed. No model experiment or deployment
occurred.

## Fresh source issues that must be resolved

- **Equation (1), p. 2679:** both binary cross-entropy terms print log(p), so
  the binary expression collapses to -log(p), independent of the label. Record
  the literal algebraic defect. Standard binary cross-entropy with log(1-p) in
  the second term is the immediate plausible repair for the intended model;
  no expensive run is needed merely to demonstrate the printed cancellation.
- **Section IV-B, p. 2682:** the sequential model is trained jointly through
  classification loss. No explicit reconstruction term or benign pretraining
  is specified for the combined model. Do not silently import the standalone
  AEA's training procedure. Record joint training as the source reading and
  any pretraining or combined-loss alternative separately.
- **Section III-A.3, p. 2678:** poisoning percentages concern customers for
  generalized models and samples for customer-specific models. The exact
  replacement/label-flip procedure and relation to ADASYN need completion.
- **Sections III-A and III-D, pp. 2677 and 2680:** ADASYN operates on novelty
  test samples or before the two-class split; the ROC/IQR threshold rule is
  ambiguous. Preserve source operations and diagnose their effects explicitly.
- **Table IV:** the paper explicitly averages customer metrics. Nonlinear
  single-confusion-matrix identities need not hold after metric averaging.
  Reassess earlier source-arithmetic claims under the proper aggregation
  assumptions before using them in the fresh investigation.

## Hardware and time

The paper specifies RTX 2070, 50 epochs, batch 100; roughly one hour for
shallow detectors, 1.5-3 hours for deep models, three hours for averaging, and
four hours for the sequential ensemble (pp. 2680, 2682).

Prefer a single V100-16GB on Panther if accessible. It is a reasonable candidate
with favorable memory and published compute specifications. VRAM capacity alone
does not establish model throughput: an A16 GPU, T4, and V100 are distinct
devices even when each has 16 GB. The standard scikit-learn classical models
will use CPU resources. Measure the actual execution path and preserve printed
activations, losses, precision, and batch size when interpreting runtime.

Treat 50-epoch completion and the reported elapsed time as distinct observables.
Freeze a finite job budget after the pilot and record what completed by the
paper's time. If the full 50 epochs exceed it, report the measured discrepancy;
do not infer that all efficient implementations or all later training must
fail. Any extension aimed at the statistical performance question must have a
named purpose and finite budget before execution.

Primary hardware references consulted on September 20:

- [NVIDIA RTX 2070 specifications](https://www.nvidia.com/content/nvidiaGDC/gb/en_GB/geforce/graphics-cards/rtx-2070.html): 8 GB GDDR6, 448 GB/s.
- [NVIDIA V100 specifications](https://www.nvidia.com/en-in/data-center/v100/): 16/32 GB variants, 900 GB/s for V100, 14 TFLOPS FP32 for PCIe.
- [NVIDIA A16 specifications](https://www.nvidia.com/en-gb/data-center/products/a16-gpu/): four separate 16 GB GPUs, 200 GB/s each.

The initial noninteractive SSH attempt failed authentication. The subsequent
authorized password login succeeded; the sources and matching software were
present, and Slurm listed V100-16GB resources. CPU job 397206 completed the
preparation check. No GPU was allocated or benchmarked for this paper.

## Deliverable

A reported-versus-measured account covering every listed model, with visible
setup assumptions, complete and partial table coverage distinguished, every
attempt retained, and uncertainty appropriate to the data and repetition
structure. The conclusion follows the measurements. Source verification,
pipeline implementation, and the first paired forest pilot are complete.
Full-population preparation and the remaining detector implementations have
not been executed. The pilot does not supply a complete-paper verdict.
