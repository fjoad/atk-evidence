# Robust poisoning: fresh reproduction of every reported model

**Date:** 2026-09-20

**State:** revised research direction and plan; no new experiment launched

## Current execution: steps 1 and 2

The user has authorized source/data resolution and implementation of the shared
preparation pipeline. Work now covers:

- [ ] Independently verify online dataset identity, the locally available raw
  bytes, documentation, and cited attack definitions.
- [ ] Record one explicit initial setup and all material unresolved alternatives
  before preparing research data.
- [ ] Implement acquisition/verification and preparation for novelty and
  two-class models with customer/day/attack/poison/synthetic provenance.
- [ ] Check parsing, attacks, splitting, scaling, ADASYN, and poisoning on
  hand-checkable local software fixtures, including direct library parity.
- [ ] Run a bounded preparation-only check on Panther if authentication is
  available; otherwise report precisely that research-data execution is pending.
- [ ] Update the journal, generated website, status, and checks; commit.

No detector fitting, score comparison, seed sweep, or full ADASYN allocation
is included in these two steps. Local tests use constructed software fixtures;
all real-data preparation remains on cluster compute nodes. A September 20
read-only SSH attempt reached Panther but failed noninteractive authentication.

## Direction

The user requests a fresh investigation of the paper's actual experiments,
covering all baseline and proposed models. Earlier workflows and conclusions
are historical evidence, not prerequisites or assumed outcomes. The priority
is a correct reconstruction of the data and ordinary library implementations
with the reported settings. Do not investigate speculative number-generation
patterns, author intent, or invented-data scenarios in this phase.

This plan replaces the completed September 2 source-only plan as Paper 3's
current direction. The present turn is a source review and plan, not a job
submission. Existing results and source freezes remain preserved. Paper 1's
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

Panther responded to SSH on September 20, but the noninteractive attempt failed
authentication. Existing notes document interactive password login. No current
queue or GPU-availability claim has been verified, and no job was submitted.

## Deliverable

A reported-versus-measured account covering every listed model, with visible
setup assumptions, complete and partial table coverage distinguished, every
attempt retained, and uncertainty appropriate to the data and repetition
structure. The conclusion follows the measurements. Current work is the plan;
implementation and experiment execution are subsequent steps.
