# ATK Evidence — Paper Reproduction Runbook

> **Read this file first in every new session and for every new paper.**
>
> This is the canonical end-to-end implementation tutorial for the project.
> The source paper is the authority for scientific content; this runbook is the
> authority for sequencing the work. If another project document encourages
> infrastructure, reporting, or exhaustive branching before the first faithful
> experiment runs, follow this file.

## The job in one sentence

Read the paper completely, write down exactly what experiment it describes,
implement that experiment in five readable Python files, run it, preserve every
result, test the material alternative readings, and only then write and publish
the scientific report.

The intended balance is:

- careful reading and experimental design dominate the reasoning;
- data preparation and model training dominate elapsed wall time; and
- framework engineering, orchestration, documentation, and publishing remain
  small supporting tasks.

If the codebase is growing while no eligible full experiment has run, stop and
return to this runbook.

## The governing reframe

The durable rationale is recorded in
[`docs/decisions/2026-08-09-paper-first-minimal-instrument.md`](docs/decisions/2026-08-09-paper-first-minimal-instrument.md).

This project is a scientific audit, not a software platform. Paper extraction is
the thought-intensive work: read and visually inspect the complete source,
reconcile its prose, equations, algorithms, figures, and tables, and freeze every
material claim and uncertainty with source locators. Only then write code.

Implementation should be mostly transcription of that frozen understanding. The
global instrument stays boring; paper-specific meaning remains explicit. Reuse
only stable mechanical operations such as file verification, parsing, standard
metrics, and result serialization. A few duplicated lines are preferable to an
abstraction that hides what a paper says.

The operating test is:

> If removing a component would not prevent or invalidate the next named
> scientific result, that component is not needed yet.

No run launches until it names the paper/table cell, model, `P`/`I`/`C`
interpretation, seed, exact question, and report result it feeds.

## Scientific rules that never change

1. **Paper first.** Reconstruct the method directly from the complete PDF before
   reading an old implementation as an authority.
2. **Printed method first.** Preserve questionable printed operations. Do not
   silently replace them with better practice.
3. **No hidden choices.** Record every assumption needed to make the paper
   executable before seeing whether it improves the result.
4. **Three result families.**
   - `P`: the printed procedure, including statistically improper steps;
   - `I`: materially defensible interpretations of omissions or contradictions;
   - `C`: scientifically corrected controls, reported separately.
5. **No cherry-picking.** Preserve all seeds, failures, branches, timings, and
   metrics. A lucky match is not a reproduction.
6. **Exact data or an explicit gate.** Never substitute a convenient dataset or
   sample representation without labeling it ineligible for the primary claim.
7. **Bound every conclusion.** State what was tested within a finite declared
   space. Never infer author intent or claim an infinite space was exhausted.
8. **Independent papers.** A result for one paper says nothing conclusive about
   another until that paper is independently reproduced.
9. **Scrutinize the claimed mechanism, not only the reported numbers.** A paper
   usually asserts *why* its method works — that some structure exists and that
   simpler models cannot exploit it. That assertion is testable independently of
   whether any number reproduces, and it is often the more decisive question.
   Reproduction alone is a weak instrument: it invites the reply "you implemented
   it wrong," which is sometimes correct.
10. **Establish the floor before comparing ceilings.** Measure what a
    zero-parameter rule achieves on the task, through the identical evaluation
    path, before interpreting any model comparison. If a rule with no parameters
    performs comparably, the task is trivial and every downstream comparison —
    the paper's and your own — is moot. Order the work so a low-tier failure
    stops you from spending effort on higher tiers.
11. **Breadth first, then depth.** Run many small, cheap checks that each answer
    one question; analyse them; only then spend training compute, and only on
    directions the analysis made promising. Most questions about a dataset are
    answered by the data, not by a trained model. Submitting a large batch of
    expensive jobs before analysing anything is the failure mode: it consumes a
    shared cluster, returns nothing for hours, and produces no learning in
    between. A cheap check that turns out to be wrong is also far better caught
    in two minutes than after a long sweep has been built on it.

## The minimal per-paper product

Every paper lives at:

```text
studies/<study-id>/
```

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

Do not begin by reading old code. Begin with the PDF and `METHOD.md`.

### Stage 1 — Read the paper end to end

Read and visually inspect every page, including equations, algorithms, figure
captions, footnotes, and tables. Build `METHOD.md` directly from the PDF.

This stage should consume most of the project's reasoning. Do not optimize for a
quick summary. The required product is an executable reconstruction precise
enough that implementation becomes straightforward and a reviewer can identify
the source of every consequential line of scientific code.

`METHOD.md` must contain:

1. **Target claims:** every table/figure/number to reproduce.
2. **Data:** exact datasets, populations, sample units, dates, labels, and
   access routes.
3. **Preparation order:** parsing, filtering, imputation, normalization,
   attacks, balancing, splitting, and validation.
4. **Models:** every layer, width, activation, loss, optimizer, and score stated
   by the paper.
5. **Training:** search procedure, epochs/stopping, batch size, seeds, and all
   reported or missing settings.
6. **Evaluation:** thresholds, score orientation, test identity, formulas, and
   reported metrics.
7. **Timing:** exactly what time is claimed and what hardware/protocol is
   reported.
8. **Source table:** one row per material instruction with a page/section/
   equation/algorithm/figure/table locator and one of:
   `EXACT`, `AMBIGUOUS`, `CONTRADICTORY`, or `NON-EXECUTABLE`.
9. **Straight-through experiment:** one declared primary reading that can run
   without choosing settings after seeing outcomes.
10. **Open branches:** each material alternative reading, listed but not yet
    expanded into a combinatorial system.

#### Source-freeze checkpoint

Before writing model code, confirm:

- a reader can reconstruct the full experiment from `METHOD.md`;
- every target table maps to a stated data/model/evaluation path;
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

### Stage 5 — Run sanity experiments immediately

Before cluster work, run:

1. data load and shape check;
2. one forward pass;
3. one finite gradient/update;
4. one epoch on the tiny subset;
5. reconstruction/score distribution check;
6. threshold-direction and metric-formula check;
7. a trivial baseline such as zero reconstruction or class prevalence;
8. a positive control when needed to show that the prepared data contain a
   learnable signal.

Save the sanity result, but label it as fixture/exploratory evidence.

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

### Stage 7 — Reproduce the reported tables

After the first anchor works:

1. run the remaining models for that table;
2. repeat the predeclared seeds;
3. proceed to the next reported table;
4. regenerate tables from saved attempt files;
5. display reported and reproduced values side by side;
6. report mean, dispersion, every seed, failures, and training time.

Run cheap independent cells in parallel. Keep no more cluster orchestration
than necessary to submit and inspect those jobs.

### Stage 8 — Test ambiguity branches without explosion

The project still requires every materially defensible interpretation, but not
every arbitrary Cartesian mixture.

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

Reproduction asks *do the numbers come out?* This stage asks *is the stated
reason true?* Run it whenever a paper justifies its architecture by claiming that
particular structure exists in the data and that simpler models cannot capture
it. Design every test so its outcome is not controlled by the expected answer.

Run this stage breadth-first (rule 11). Most of the checks below need no training
at all and belong in one short job; only the survivors justify training runs.
Verify each measurement is reading what you think — a structure measurement taken
on attack-contaminated inputs can read as "no structure" purely because injected
outliers dominate the estimate, which would manufacture a false headline finding.

The instrument is **remove the structure and observe whether anything notices**:

1. **Triviality floor.** Evaluate zero-parameter rules through the identical
   preprocessing, scoring, threshold, and metric path as every trained model, so
   any difference is attributable to the model rather than to the evaluation.
   Report them alongside the paper's models, marked as not from the paper.
2. **Structure ablations.** Destroy one claimed structure at a time — spatial
   correlation, temporal order, context length, graph topology — leaving labels
   and evaluation untouched so runs stay comparable. Verify each ablation
   actually destroyed what it targets, and record the measurement.
3. **Component ablations.** Remove each architectural component the paper credits
   and observe whether the metric moves.
4. **Capacity check.** If the paper claims a simpler model *cannot* learn the
   structure, vary that model's capacity. A flat response means it is not
   capacity-limited and the claim fails on its own terms.
5. **Fairness of the comparison.** If your implementation of the paper's own
   baseline substantially outperforms the paper's reported baseline, record it —
   a margin measured against an undertrained baseline is not a margin.
6. **Ceiling analysis.** Decompose performance by the structure of the task
   itself. A limit imposed by the data or the detection procedure cannot be
   fixed by any architecture, and explains clustering that would otherwise look
   like coincidence.

Report whatever these produce. An ablation that degrades performance confirms the
paper's premise for that component and must be stated as plainly as one that does
not. State results as distributions across seeds; a single run is not evidence
when run-to-run spread is comparable to the effect being claimed.

### Stage 9 — Freeze and run confirmatory experiments

Exploratory results do not become confirmatory retrospectively.

Before confirmatory execution, freeze:

- target numerical pattern and tolerances;
- eligible paper-consistent branches;
- hyperparameter envelope;
- datasets and split identities;
- seeds/repetitions;
- statistical tests and uncertainty estimates;
- compute budget and stopping rule; and
- treatment of failures.

Then run without changing the contract. Report a reproduction only if the
complete principal pattern is stable—not because one metric or seed matches.

### Stage 10 — Analyze and issue the bounded verdict

`analyze_results.py` must produce:

- the paper's tables from preserved runs;
- reported versus reproduced metrics;
- seed-level and aggregate statistics;
- failure and eligibility counts;
- timing summaries;
- sensitivity across material `I` branches;
- `C` controls in a separate section; and
- machine-readable inputs for the report.

Allowed verdict language:

> The reported result pattern was / was not reproduced within the declared
> datasets, paper-consistent implementations, hyperparameter envelope, seeds,
> partitions, tolerances, and compute budget.

Do not claim fabrication, intent, or impossibility beyond the tested space.

### Stage 11 — Write and publish only after results

After the paper-level verdict is frozen:

1. complete `reports/<study-id>/main.tex`;
2. compile and visually verify the PDF;
3. publish the method map and report under `site/papers/<study-id>/`;
4. update the cross-paper synthesis only after multiple independent verdicts.

Publication is downstream of evidence. It must never block or displace the
first eligible experiment.

### Stage 12 — Solve the research problem properly

Only after the reproduction verdict is frozen may a separately contracted
study ask how the problem should actually be solved. Build transparent
baselines, isolate failure mechanisms, and test a scientifically preferred
method against the same held-out data. This work cannot alter the reproduction
verdict.

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

- model code begins before the source-freeze checkpoint;
- a proxy dataset is being treated as the named data;
- a passing test validates a contract rather than the PDF statement;
- the first full result is blocked by a generalized framework;
- wrappers conceal the real scientific implementation;
- code grows beyond the soft complexity boundary without a scientific reason;
- corrected methodology leaks into the printed track;
- branch count grows before a straight-through anchor runs;
- cluster tooling becomes more than a short resource wrapper;
- documentation or publishing is consuming time needed for experiments; or
- only favorable seeds/results are being retained.

## Fresh-agent completion checklist

For any new paper, a fresh agent should be able to follow this list:

- [ ] Register the paper and fingerprint the complete PDF.
- [ ] Read every page and write `METHOD.md` with source locators.
- [ ] Pass the source-freeze checkpoint.
- [ ] Acquire and checksum the exact named data.
- [ ] Implement the five real reproduction files.
- [ ] Pass tiny data/model/metric sanity checks.
- [ ] Run one eligible full anchor and report its timing immediately.
- [ ] Complete the reported tables over predeclared seeds.
- [ ] Test material `I` branches and separate `C` controls.
- [ ] Freeze and execute the confirmatory contract.
- [ ] Generate the bounded verdict from preserved attempts.
- [ ] Complete, verify, and publish the LaTeX report.
- [ ] Only then begin a corrected solution or the next paper.

When uncertain, choose the action that gets a source-faithful result running
sooner without weakening the evidence.
