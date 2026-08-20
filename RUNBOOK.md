# ATK Evidence — Paper Audit Runbook

> **Read this file first in every new session and for every new paper.**
>
> This is the canonical end-to-end scientific workflow for the project.
> The source paper is the authority for scientific content; this runbook is the
> authority for sequencing the work. If another project document encourages
> infrastructure, reporting, or exhaustive branching before the first faithful
> experiment runs, follow this file.

## The job in one sentence

Read the paper completely, expose its numerical and causal claims in a small
discovery sandbox, freeze the source, build the smallest faithful instrument,
map competing explanations with cheap diagnostics, and deepen only the
surviving numerical, mechanism, and attainability questions before reporting
three separately bounded findings.

The intended balance is:

- careful reading, sandbox discovery, and discriminating experimental design
  dominate the reasoning;
- cheap diagnostic breadth precedes expensive data preparation and training;
- promoted formal experiments dominate elapsed wall time; and
- framework engineering, orchestration, documentation, and publishing remain
  small supporting tasks.

If code or compute is growing while no competing explanation is being
discriminated, stop and return to this runbook.

## The governing reframe

The durable rationale is recorded in
[`docs/decisions/2026-08-09-paper-first-minimal-instrument.md`](docs/decisions/2026-08-09-paper-first-minimal-instrument.md).
The three-part evidence contract, discovery sandbox, and breadth/depth boundary
are recorded in
[`docs/decisions/2026-08-20-three-part-evidence-frame.md`](docs/decisions/2026-08-20-three-part-evidence-frame.md).

This project is a scientific audit, not a software platform. Paper extraction is
the thought-intensive work: read and visually inspect the complete source,
reconcile its prose, equations, algorithms, figures, and tables, and freeze every
material claim and uncertainty with source locators. Only then write eligible
formal scientific code; the earlier sandbox remains deliberately exploratory.

Implementation should be mostly transcription of that frozen understanding. The
global instrument stays boring; paper-specific meaning remains explicit. Reuse
only stable mechanical operations such as file verification, parsing, standard
metrics, and result serialization. A few duplicated lines are preferable to an
abstraction that hides what a paper says.

The implementation operating test is:

> If removing a component would not prevent or invalidate the next named
> scientific result, that component is not needed yet.

No eligible formal run launches until it names the paper/table cell or causal
claim, `N`/`M`/`A` evidence question, `P`/`I`/`C` implementation track, model,
seed, competing predictions, exact question, and report finding it feeds. An
adaptive `X` sandbox keeps a lighter question/setup/observation record and must
be formally re-specified before promotion.

## Two classifications, not one

Implementation semantics and scientific questions are orthogonal:

| Classification | Values | Meaning |
|---|---|---|
| Implementation track | `P`, `I`, `C`, `X` | paper-literal, reasonably interpreted, controlled, or exploratory execution |
| Evidence question | `N`, `M`, `A` | numerical reproduction, mechanism identification, or attainability |

A corrected ablation can answer a mechanism question without becoming a
repaired reproduction. One printed run may contribute to all three questions,
but each conclusion needs its own evidence and boundary.

## Scientific rules that never change

1. **Paper first.** Read the complete PDF before treating code or prior
   contracts as authority. Extract both numerical targets and the causal claim
   `B > A because Z exploits S`.
2. **Discover cheaply.** Use a disposable sandbox to find discriminating
   questions before building a formal program. Sandbox output is exploratory;
   promoted questions return to the complete source and exact data before
   formal execution.
3. **Printed method first for numerical evidence.** Preserve questionable
   printed operations. Do not silently replace them with better practice.
4. **No hidden choices.** Record every assumption needed to make the paper
   executable before seeing whether it improves the result.
5. **Four implementation tracks.**
   - `P`: the printed procedure, including statistically improper steps;
   - `I`: materially defensible interpretations of omissions or contradictions;
   - `C`: scientifically corrected controls, reported separately; and
   - `X`: adaptive exploratory or externally motivated work that cannot count
     as reproduction or become confirmatory without a new frozen contract.
6. **Three findings.** Numerical, mechanism, and attainability conclusions are
   earned separately. None is inferred from another.
7. **No cherry-picking.** Preserve all seeds, failures, branches, timings, and
   metrics. A lucky match is not a reproduction.
8. **Exact data or an explicit gate.** Never substitute a convenient dataset or
   sample representation without labeling it ineligible for the primary claim.
9. **Bound every conclusion.** State what was tested within a finite declared
   space. Never infer author intent or claim an infinite space was exhausted.
10. **Independent papers.** A result for one paper says nothing conclusive about
    another until that paper is independently reproduced.
11. **Scrutinize the claimed mechanism, not only the reported numbers.** A paper
   usually asserts *why* its method works — that some structure exists and that
   simpler models cannot exploit it. That assertion is testable independently of
   whether any number reproduces, and it is often the more decisive question.
   Reproduction alone is a weak instrument: it invites the reply "you implemented
   it wrong," which is sometimes correct.
12. **Establish the floor before comparing ceilings.** Measure what a
    zero-parameter rule achieves on the task, through the identical evaluation
    path, before interpreting any model comparison. If a rule with no parameters
    performs comparably, the task is trivial and every downstream comparison —
    the paper's and your own — is moot. Order the work so a low-tier failure
    stops you from spending effort on higher tiers.
13. **Diagnostic breadth first, then execution depth.** Run many small, cheap
    checks that each answer one question; analyse them; only then spend training
    compute, and only on directions the analysis made promising. Most questions
    about a dataset are answered by the data, not by a trained model. Submitting
    a large batch of expensive jobs before analysing anything is the failure
    mode: it consumes a shared cluster, returns nothing for hours, and produces
    no learning in between. One costly full run per named model is already
    depth, not the first breadth layer.
14. **Attainability is empirical unless proved otherwise.** Learning and search
    curves may support conditional implausibility. Only a genuine global bound
    supports structural impossibility.

## The minimal per-paper product

Every paper lives at:

```text
studies/<study-id>/
```

Its durable scientific specification is `METHOD.md`. It contains both the
executable source reconstruction and the paper's causal-claim map. Optional
disposable sandbox code may live under an ignored temporary directory or a
clearly labeled `exploration/` area; it is not part of the formal five-file
instrument and cannot produce eligible evidence.

The primary reproduction is implemented in:

```text
studies/<study-id>/reproduction/
  download_data.py
  prepare_data.py
  models.py
  run_experiment.py
  analyze_results.py
```

These must be the real implementation, not wrappers around a large hidden
framework.

Mechanism and attainability diagnostics should normally be short direct scripts
or analysis functions over preserved arrays. Do not expand the five-file route
into a universal experiment engine to support them.

### File responsibilities

`download_data.py`

- acquire openly available files;
- explain the authorized manual route for restricted files;
- verify filenames, sizes, versions, and checksums;
- never store credentials or modify raw files.

`prepare_data.py`

- parse the exact named data;
- implement the paper's transformations in paper order;
- implement attack generation and splits;
- save the prepared arrays plus a small metadata record;
- support a tiny mode and a full mode.

`models.py`

- contain readable model-building functions;
- expose the actual layers, widths, activations, losses, and anomaly scores;
- keep the paper model and corrected models visibly separate;
- avoid factories or abstraction layers that obscure the architecture.

`run_experiment.py`

- load one prepared dataset;
- train one model for one seed/configuration;
- time loading, training, scoring, and total execution;
- save configuration, history, scores, metrics, and failures;
- support a tiny sanity run before full execution.

`analyze_results.py`

- load every preserved attempt;
- regenerate the paper's target tables;
- compare reported and reproduced values;
- summarize seeds and failures without selecting a favorite;
- produce machine-readable tables for the LaTeX report.

### Complexity boundary

- Target at most roughly 1,500–2,000 non-test lines across the five scientific
  files for the first complete paper route.
- A few direct tests or assertions may live separately, but they must test this
  code rather than become a second implementation.
- Use ordinary functions and small data structures.
- Do not create a workflow engine, branch-lattice framework, scheduler,
  manifest service, plugin system, or alternate runner before the first
  eligible full result.
- Cluster support is a short `sbatch` wrapper around `run_experiment.py`.
- If the five-file target cannot be met, stop and document the specific
  scientific reason before adding architecture.

The repository may retain a larger forensic harness for later audits. It is not
the primary implementation and must not block the five-file route.

## End-to-end workflow

### Stage 0 — Start or resume

For a new paper:

1. Choose a stable lowercase study ID.
2. Register it in `studies/registry.toml`.
3. Put the local PDF under ignored `papers/`.
4. Record the PDF filename, page count, DOI, and SHA-256.
5. Create `studies/<study-id>/METHOD.md`.
6. Create a short active plan under `docs/plans/`.

For an existing paper:

1. Read this runbook.
2. Read `docs/STATUS.md` and the active plan.
3. Check `git status`.
4. Resume the first unfinished stage below.

Do not begin by treating old or author code as the method. Begin with the PDF.

### Stage 1 — First end-to-end reading and claim map

Read and visually inspect every page, including equations, algorithms, figure
captions, footnotes, and tables. Before implementation, write a provisional map
of:

- every numerical target;
- every comparison of a proposed method with a baseline;
- every statement of the form `B > A because Z exploits S`;
- what observation would support or weaken each link in that explanation; and
- which claims appear testable with static reasoning or a tiny sandbox.

Also record table-level red flags—perfect rank ordering, uniformly flattering
metrics, implausibly small or absent variance, repeated tidy increments, and
discussion that mirrors every outcome. Treat them as triage signals only. They
become evidence only through a declared check such as metric-identity
recalculation, feasible confusion-matrix enumeration, rounding analysis, or a
probability model whose dependence assumptions are explicit.

This first pass is complete-paper orientation, not a quick abstract summary and
not yet the formal source freeze.

### Stage 1b — Use a disposable discovery sandbox

Build the smallest exploratory environment capable of distinguishing the first
competing explanations. A short NumPy script, notebook, or minimum-library model
is preferred. Do not use the production runner, branch engine, cluster wrapper,
or full dataset unless the immediate question requires it.

Start with cheap questions:

1. Can the written model perform one finite update and overfit a hand-sized
   sample?
2. Do `A`, `B`, and `B-Z` behave differently on simple geometry where `Z` is
   irrelevant, useful, and necessary?
3. Does a zero, mean, one-feature, linear, nearest-neighbor, or random-feature
   rule already solve the toy task?
4. Does adding or destroying `S` change either model?
5. Are per-example scores, rankings, or representations actually different?
6. What wall-time and memory scale should the formal investigation expect?

If author code exists, inspect it only after the paper claim map exists. Record
whether it implements, completes, or contradicts the paper; never let it
silently redefine the publication. If no code exists, the sandbox is the first
minimal reimplementation of the written claim.

Sandbox work is adaptive and may fail. Preserve useful notes, but label every
outcome exploratory. Its purpose is to discover a discriminating formal
question, not to produce an eligible result.

### Stage 1c — Return to the paper and freeze the formal source

Re-read and visually verify every source location implicated by the sandbox.
Build `METHOD.md` directly from the PDF, not from the sandbox or author code.

This stage should consume most of the project's reasoning. Do not optimize for a
quick summary. The required product is an executable reconstruction precise
enough that implementation becomes straightforward and a reviewer can identify
the source of every consequential line of scientific code.

`METHOD.md` must contain:

1. **Numerical targets:** every table/figure/number to reproduce.
2. **Causal claims:** every `B > A because Z exploits S` statement, its source
   locator, six-link decomposition, and candidate discriminating observations.
3. **Data:** exact datasets, populations, sample units, dates, labels, and
   access routes.
4. **Preparation order:** parsing, filtering, imputation, normalization,
   attacks, balancing, splitting, and validation.
5. **Models:** every layer, width, activation, loss, optimizer, and score stated
   by the paper.
6. **Training:** search procedure, epochs/stopping, batch size, seeds, and all
   reported or missing settings.
7. **Evaluation:** thresholds, score orientation, test identity, formulas, and
   reported metrics.
8. **Timing:** exactly what time is claimed and what hardware/protocol is
   reported.
9. **Source table:** one row per material instruction with a page/section/
   equation/algorithm/figure/table locator and one of:
   `EXACT`, `AMBIGUOUS`, `CONTRADICTORY`, or `NON-EXECUTABLE`.
10. **Straight-through experiment:** one declared primary reading that can run
   without choosing settings after seeing outcomes.
11. **Open branches:** each material alternative reading, listed but not yet
    expanded into a combinatorial system.
12. **Promoted questions:** sandbox-derived hypotheses that remain
    source-relevant, with competing predictions and their exploratory origin
    disclosed.

#### From impossible wording to executable evidence

A `NON-EXECUTABLE` or genuinely contradictory instruction is not the end of
the audit and must not be silently skipped:

1. preserve the exact printed wording, locator, and reason it cannot execute;
2. record the literal branch as a failure/non-executable outcome;
3. predeclare the smallest reasonable repair and every other materially
   defensible interpretation before seeing results;
4. execute each repaired interpretation under its own stable `I` identifier;
5. show reported values, literal failure, every repair's results, seeds,
   timings, and eligibility side by side on the study site and in the LaTeX
   report; and
6. report a match if one occurs and a bounded non-match if none occurs.

No repaired branch may be described as what the authors secretly used. It only
answers whether a reasonable executable completion of their text reproduces
the claim.

The durable evidence contract and its boundaries are recorded in
[`docs/decisions/2026-08-11-non-executable-source-ladder.md`](docs/decisions/2026-08-11-non-executable-source-ladder.md).

#### Source-freeze checkpoint

Before writing eligible formal model code, confirm:

- a reader can reconstruct the full experiment from `METHOD.md`;
- every target table maps to a stated data/model/evaluation path;
- every claimed architectural advantage maps to `A`, `B`, `Z`, and `S`;
- each promoted mechanism question has outcomes that distinguish competing
  explanations;
- missing information is visible rather than silently filled;
- the proposed primary reading is declared before results; and
- the user has seen the pivotal contradictions or non-executable steps.

This is the most important checkpoint in the project. A passing test suite
cannot compensate for a wrong reading of the paper.

### Stage 2 — Acquire and inspect the exact data

Implement `download_data.py` first.

For each source:

- record the authoritative URL/DOI and access status;
- download only public or user-authorized material;
- verify hashes before parsing;
- keep raw files immutable and outside Git;
- record any alternate source and exactly what identity it establishes.

Then use a small read-only inspection to establish:

- rows/customers/meters/samples;
- feature/time dimensions;
- labels and class balance;
- missing/duplicate/invalid values;
- whether the paper's described sample unit actually exists.

If the named data or sample representation is unavailable, stop the primary
track. A proxy may be explored only under an explicit ineligible label.

### Stage 3 — Implement preparation before models

Implement `prepare_data.py` in the exact paper order.

First add `--tiny`:

- deterministic small subset;
- same transformations as full mode;
- fast enough to rerun locally in seconds or minutes.

Then add `--full`:

- exact source population;
- fixed seed and identity-preserving splits;
- prepared data stored under ignored `data/derived/<study-id>/`;
- metadata containing source hashes, counts, shapes, choices, and runtime.

Verify attack equations and transformations numerically on hand-checkable
examples. Preserve customer/sample identity so train/test leakage can be
measured even when the printed method permits it.

### Stage 4 — Implement the models literally

Implement `models.py` from the frozen source table.

For every model:

- enumerate runtime layers and compare them with the paper;
- verify the input and output shapes;
- verify output ranges against the preprocessed target domain;
- verify the printed loss and optimizer;
- verify anomaly-score definition and direction;
- make omitted choices explicit function arguments or one declared default.

Do not improve the architecture in the `P` track. Corrected architectures
belong under clearly named `C` functions or configurations.

### Stage 5 — Prove the instrument and run diagnostic breadth

Before cluster work, first prove the formal instrument:

1. data load and shape check;
2. one forward pass;
3. one finite gradient/update;
4. one epoch on the tiny subset;
5. reconstruction/score distribution check;
6. threshold-direction and metric-formula check;
7. a trivial baseline such as zero reconstruction or class prevalence; and
8. a positive control when needed to show that the prepared data contain a
   learnable signal.

Then cover the cheapest discriminating questions across all three evidence
families:

- `N`: static table identities, score direction, exact thresholds on small
  vectors, and one straight-through tiny result;
- `M`: task triviality, structure presence, synthetic capability witnesses,
  untrained-versus-trained behavior, and one-component removals; and
- `A`: small data-size, capacity, epoch, runtime, and memory contrasts that show
  which response axes might plausibly move the target.

Each probe must state the competing explanations and the different observation
each predicts. Run independent cheap probes in parallel when useful, but
inspect the map before adding any expensive job. Running one full seed for every
named model is not this stage.

Save the sanity and diagnostic results with their actual status. Pure software
fixtures are ineligible; adaptive probes remain `X`; source-frozen diagnostics
retain their `P`, `I`, or `C` semantics and may become formal evidence only
under their declared contract. Promote a question only when it remains
source-relevant and the cheap evidence leaves material uncertainty that a full
run can resolve.

#### Runnable-anchor checkpoint

Do not add generalized infrastructure until one model completes:

```text
raw data → prepared data → training → scores → paper metrics → saved result
```

The runtime layer inventory, sample counts, and metric formulas must link back
to `METHOD.md`.

### Stage 6 — Run the first full anchor

Choose the least ambiguous paper table/dataset and the simplest proposed model.
Run one full exploratory seed using the declared primary reading.

Record:

- exact commit;
- source/prepared-data hashes;
- complete configuration and seed;
- actual layer inventory;
- sample counts and split identities;
- load, preparation, fit, scoring, and total wall times;
- raw scores/predictions where practical;
- every paper metric; and
- the corresponding reported values.

This first full result is the critical project milestone. It comes before a
website, polished report, exhaustive branch engine, or large search.

If it fails operationally, preserve the failure and fix only implementation
errors. Do not silently change the paper method to make it run.

### Stage 7 — Inspect the anchor and freeze promoted depth

After the first anchor works:

1. reload and verify every saved score, identity, metric, and timing;
2. compare the trained result with trivial, zero, untrained, and positive
   controls already established;
3. state what the anchor changed for the numerical, mechanism, and
   attainability questions;
4. run any remaining cheap probe that could prevent an unnecessary full run;
5. list the competing explanations that survived diagnostic breadth; and
6. freeze which questions promote to full data, repetitions, or bounded search,
   including their predictions and finish conditions.

Do not promote a model merely because it appears as another paper row. Promote
the numerical row if it is required for the numerical finding; promote a
mechanism contrast if it can discriminate the causal claim; promote a search
axis if the anchor shows that attainability remains materially open.

### Stage 8 — Numerical depth and finite source coverage

For questions promoted to numerical depth:

1. run the remaining required models for the target table;
2. repeat the frozen seeds and partitions;
3. proceed to the next reported table only when its result contributes to the
   declared numerical finding;
4. regenerate tables from saved attempt files;
5. display reported and reproduced values side by side; and
6. report mean, dispersion, every seed, failure, and training time.

Run independent eligible cells in parallel. Keep no more cluster orchestration
than necessary to submit and inspect them.

The numerical track still requires every materially defensible interpretation,
but not every arbitrary Cartesian mixture.

Use this order:

1. run the frozen straight-through `P` anchor;
2. vary one material ambiguity at a time;
3. combine ambiguities only when the paper couples them or a single-variable
   result establishes a scientifically relevant interaction;
4. run `C` controls separately;
5. preserve all branches regardless of whether they help reproduction.

Before a large ambiguity sweep, freeze:

- the finite branch list and why each branch is textually defensible;
- incompatible combinations;
- the screening metric and promotion rule;
- the seed and compute budget;
- the stopping rule; and
- the complete-pattern reproduction tolerance.

An existing forensic branch inventory may be used to check coverage after the
primary route runs. It must not become a prerequisite for the first result.

### Stage 8b — Scrutinize the claimed mechanism

Numerical reproduction asks *do the numbers come out?* This stage asks whether
the experiment identifies the stated reason. For every promoted claim
`B > A because Z exploits S`, test separately whether `S` exists, whether it is
useful, whether `A` lacks the capability, whether `Z` gives it to `B`, whether
the trained `B` uses it, and whether that use causes the advantage. Design every
test so its outcome is not controlled by the expected answer.

Run this stage breadth-first (rule 13). Most of the checks below need no training
at all and belong in one short job; only the survivors justify training runs.
Verify each measurement is reading what you think — a structure measurement taken
on attack-contaminated inputs can read as "no structure" purely because injected
outliers dominate the estimate, which would manufacture a false headline finding.

The instruments are **make the capability necessary**, then **remove the
structure and observe whether anything notices**:

1. **Capability witnesses.** Use controlled synthetic distributions where `Z`
   is irrelevant, useful, and necessary. Verify that the written `A`, `B`, and
   `B-Z` implementations differ when the claimed capability should matter.
2. **Triviality floor.** Evaluate zero-parameter rules through the identical
   preprocessing, scoring, threshold, and metric path as every trained model, so
   any difference is attributable to the model rather than to the evaluation.
   Report them alongside the paper's models, marked as not from the paper.
3. **Structure ablations.** Destroy one claimed structure at a time — spatial
   correlation, temporal order, context length, graph topology — leaving labels
   and evaluation untouched so runs stay comparable. Verify each ablation
   actually destroyed what it targets, and record the measurement.
4. **Component ablations.** Remove each architectural component the paper credits
   and observe whether the metric moves.
5. **Capacity check.** If the paper claims a simpler model *cannot* learn the
   structure, vary that model's capacity. A flat response means it is not
   capacity-limited and the claim fails on its own terms.
6. **Fairness of the comparison.** If your implementation of the paper's own
   baseline substantially outperforms the paper's reported baseline, record it —
   a margin measured against an undertrained baseline is not a margin.
7. **Learned-behavior check.** Compare per-example scores, rankings,
   representations, attention, gates, or other credited internal behavior. A
   component that exists in the graph but is constant or ignored has not
   demonstrated its claimed function. Use effective rank, PCA, CKA, SVCCA, or
   another representation diagnostic only when it answers a predeclared
   capability question; dimensionality reduction alone is not a mechanism
   test.
8. **Ceiling analysis.** Decompose performance by the structure of the task
   itself. A limit imposed by the data or the detection procedure cannot be
   fixed by any architecture, and explains clustering that would otherwise look
   like coincidence.

Report whatever these produce. An ablation that degrades performance confirms the
paper's premise for that component and must be stated as plainly as one that does
not. Compare `A`, `B`, and `B-Z` with matched seeds and partitions. Freeze a
smallest effect of scientific interest and use paired intervals or an
equivalence procedure when claiming no material difference. State which causal
links are supported, contradicted, unidentified, or untested.

### Stage 8c — Map the empirical attainability envelope

Attainability asks whether the described method shows a credible route to the
reported target. It is not inferred from the number of failed branches.

Freeze the finite axes that the diagnostic map showed could plausibly matter:

- paper-supported data and preprocessing completions;
- seeds and independent partitions;
- model capacity and relevant component capacity;
- training-set size, duration, optimizer budget, and stopping behavior;
- threshold-feasible regions for saved scores; and
- cumulative trials, device-hours, and failure handling.

Record the complete-pattern result against each axis, not only the best metric.
Plot learning, capacity, and best-result-versus-compute curves with uncertainty.
Use deliberately optimistic extrapolations as contextual models, never as
global bounds. Publication dates do not constrain unreported project start time
or hardware and cannot establish what search the authors performed.

The attainable conclusion ladder is:

1. target observed inside the declared envelope;
2. target not observed but an improving route remains credible;
3. target far outside a stable or saturating envelope and therefore highly
   implausible within it; or
4. target structurally impossible under stated assumptions because a genuine
   global bound has been proved.

### Stage 9 — Freeze and run confirmatory experiments

Exploratory results do not become confirmatory retrospectively.

Before confirmatory execution, freeze:

- `N`, `M`, or `A` evidence question and the competing predictions;
- target numerical pattern, causal effect, or attainability region and its
  tolerances;
- eligible paper-consistent branches and separately labeled controls;
- hyperparameter and performance-envelope axes;
- datasets and split identities;
- independent statistical unit;
- seeds/repetitions;
- statistical tests and uncertainty estimates;
- compute budget and stopping rule; and
- treatment of failures.

Then run without changing the contract. Report a numerical reproduction only if
the complete principal pattern is stable—not because one metric or seed
matches. Report a mechanism effect only if the capability-sensitive and
paper-data tests agree within their declared uncertainty. Report empirical
implausibility only inside the frozen envelope.

### Stage 10 — Analyze and issue three bounded findings

`analyze_results.py` must produce:

- the paper's tables from preserved runs;
- reported versus reproduced metrics;
- seed-level and aggregate statistics;
- failure and eligibility counts;
- timing summaries;
- sensitivity across material `I` branches;
- `C` controls in a separate section; and
- machine-readable inputs for the report.

Mechanism analysis must additionally produce:

- the `A`/`B`/`Z`/`S` claim map;
- capability-witness outcomes;
- triviality and fairness baselines;
- structure and component ablations;
- matched effect distributions and uncertainty; and
- supported, contradicted, unidentified, and untested causal links.

Attainability analysis must additionally produce:

- the declared search axes and attempted coverage;
- learning, capacity, and compute-response curves;
- failure and saturation behavior;
- observed and optimistically projected target gaps; and
- the exact boundary between empirical and structural conclusions.

Allowed numerical language:

> The reported result pattern was / was not reproduced within the declared
> datasets, paper-consistent implementations, hyperparameter envelope, seeds,
> partitions, tolerances, and compute budget.

Do not claim fabrication, intent, or impossibility beyond the tested space.

Allowed mechanism language:

> The claim that `Z` gives `B` an advantage over `A` by exploiting `S` was
> supported / contradicted / not identified within the declared witnesses,
> ablations, matched runs, effect threshold, and uncertainty procedure.

Allowed attainability language:

> The reported target was observed inside / remained unresolved outside / was
> highly implausible within the declared empirical envelope. Structural
> impossibility is claimed only under the separately stated proved bound.

No paper receives one undifferentiated verdict that lets a numerical finding
stand in for the mechanism or attainability finding.

### Stage 11 — Write and publish only after results

After the paper's three findings are frozen:

1. complete `reports/<study-id>/main.tex`;
2. compile and visually verify the PDF;
3. publish the method map and report under `site/papers/<study-id>/`;
4. update the cross-paper synthesis only after multiple independent paper
   assessments.

Publication is downstream of evidence. It must never block or displace the
first eligible experiment.

### Stage 12 — Solve the research problem properly

Only after the audit findings are frozen may a separately contracted study ask
how the problem should actually be solved. Build transparent baselines and test
a scientifically preferred method against the same held-out data. This later
solution-design work is distinct from mechanism witnesses and ablations needed
to evaluate the paper's own explanation; those belong inside the audit. A
successful new solution cannot retroactively repair any audit finding.

## Required result record

Keep result persistence simple. One attempt should produce one small JSON record
and, when necessary, one array file:

```text
results/runs/<table>/<model>/<seed>/
  result.json       # status, config, hashes, counts, metrics, timings
  scores.npz        # optional scores, labels, predictions, identities
```

`result.json` must include failures as well as successes. Never overwrite an
attempt after inspecting its outcome.

## Stop conditions

Stop and correct course when:

- production or eligible model code begins before the source-freeze checkpoint;
- a disposable sandbox grows into a framework or is presented as eligible
  evidence;
- a full run is proposed without a cheap diagnostic that explains why depth is
  needed;
- a proxy dataset is being treated as the named data;
- a passing test validates a contract rather than the PDF statement;
- the first full result is blocked by a generalized framework;
- wrappers conceal the real scientific implementation;
- code grows beyond the soft complexity boundary without a scientific reason;
- corrected methodology leaks into the printed track;
- branch count grows before a straight-through anchor runs;
- model-family coverage is called diagnostic breadth despite requiring costly
  full runs;
- numerical non-reproduction is used as mechanism or attainability evidence
  without a discriminating test;
- a fitted response curve is described as a universal bound;
- cluster tooling becomes more than a short resource wrapper;
- documentation or publishing is consuming time needed for experiments; or
- only favorable seeds/results are being retained.

## Fresh-agent completion checklist

For any new paper, a fresh agent should be able to follow this list:

- [ ] Register the paper and fingerprint the complete PDF.
- [ ] Read every page and map numerical targets plus every `B > A because Z
      exploits S` claim.
- [ ] Use a small discovery sandbox to find capability-discriminating questions.
- [ ] Return to the paper and write `METHOD.md` with source locators, causal
      claims, and promoted questions.
- [ ] Pass the source-freeze checkpoint.
- [ ] Acquire and checksum the exact named data.
- [ ] Implement the five real reproduction files.
- [ ] Pass tiny data/model/metric sanity checks and the diagnostic breadth map.
- [ ] Run one eligible full anchor and report its timing immediately.
- [ ] Promote only source-relevant uncertainties to depth.
- [ ] Complete the required numerical tables and material `I` branches over
      predeclared seeds.
- [ ] Test the claimed mechanism with capability witnesses, fair baselines,
      structure/component ablations, and matched comparisons.
- [ ] Map the declared empirical attainability envelope.
- [ ] Freeze and execute separate `N`, `M`, and `A` confirmatory contracts.
- [ ] Generate three bounded findings from preserved attempts.
- [ ] Complete, verify, and publish the LaTeX report.
- [ ] Only then begin a corrected solution or the next paper.

When uncertain, choose the smallest source-relevant action that most clearly
distinguishes the remaining explanations without weakening the evidence.
