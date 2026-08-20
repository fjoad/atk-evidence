# ATK Evidence

**→ [atk-evidence site: findings, reports, and method maps](https://fjoad.github.io/atk-evidence/)**

ATK Evidence is a paper-by-paper audit of published results that do not look
right. It rebuilds the printed method from scratch and asks three questions:
did the reported number reproduce, did the claimed mechanism actually explain
the advantage, and does the target lie inside a credible empirical performance
envelope?

Most reproduction work asks only *"do the numbers come out?"* That question is
necessary but too weak. A number can be correct while the experiment fails to
identify the explanation attached to it; a numerical miss can also be real
without proving that every possible implementation must fail. This project
therefore separates numerical, mechanistic, and attainability findings and
earns each one with the evidence it requires.

## Operating philosophy

The difficult part is reconstructing the paper correctly and discovering what
would actually discriminate its story from simpler alternatives. After one
end-to-end read, the researcher uses a disposable sandbox—tiny synthetic data,
trivial rules, minimal versions of the compared systems, and output
inspection—to find decisive questions cheaply. The researcher then returns to
the complete source and freezes every material claim, operation, omission, and
reasonable interpretation before formal model code is written. Sandbox results
remain exploratory; they can motivate a test but cannot become confirmation
after the fact.

The software is deliberately small. Each paper uses five direct scientific files
for download, preparation, models, execution, and analysis. Shared code performs
mechanical work only; the meaning of the paper remains visible in its study. The
project does not build a general experiment platform before producing eligible
results.

Execution is breadth-first before it is deep. The first computational pass uses
many cheap, question-specific probes across competing explanations. One costly
full run for every named architecture is already depth, not diagnostic breadth.
Only unresolved questions with a clear route to changing a report finding are
promoted to expensive execution.

The full rationale and working rules are in the
[`paper-first minimal-instrument decision`](docs/decisions/2026-08-09-paper-first-minimal-instrument.md)
and the
[`three-part evidence-frame decision`](docs/decisions/2026-08-20-three-part-evidence-frame.md).

## Conflict of interest

Faaiz Joad, a maintainer of this project, is the second author of the 2025 water
paper audited as study 2. That study's published findings are therefore held to
pre-registered ambiguity branches, published corrections of the audit's own
errors, and conclusions bounded strictly to what the artefact supports. The
project does not assert how any reported numbers arose.

## Papers in the pipeline

| # | Paper | Year | Published state |
|---|---|---|---|
| 1 | Deep Autoencoder-Based Anomaly Detection of Electricity Theft Cyberattacks in Smart Grids | 2022 | `in progress` |
| 2 | Graph Transfer Learning-Based Attack Detection in Cyber-Physical Water Distribution Systems | 2025 | `no consistent protocol` |

Each paper has a study directory (`studies/<id>/`), a report
(`reports/<id>/main.tex`), and an evidence page on the site. Registry labels
record publication state and existing numerical or internal-consistency
findings; the report contract requires the three findings described here.

## The pipeline

1. **Read and map the claims.** Record both numerical targets and causal claims
   such as “`B` beats `A` because component `Z` exploits structure `S`.”
2. **Discover in a sandbox.** Use toy witnesses, trivial rules, minimal systems,
   static checks, and output inspection to find discriminating questions.
3. **Freeze the source reading.** Return to the complete paper and predeclare
   the executable method, reasonable interpretations, predictions, budgets, and
   stopping rules.
4. **Build the minimal instrument.** Acquire the exact data, implement the
   paper-literal route, and pass deterministic and one-step checks.
5. **Run diagnostic breadth.** Test coherence, triviality, claimed structures,
   component necessity, comparison fairness, evaluation soundness, and possible
   ceilings with the cheapest informative probes first.
6. **Promote surviving questions to depth.** Run only the numerical branches,
   mechanism controls, or performance-envelope axes that remain material.
7. **Confirm and report three findings.** Preserve all attempts and state
   numerical, mechanism, and attainability conclusions separately.

Full protocol: [`RUNBOOK.md`](RUNBOOK.md).

## Research rule

For every paper:

1. orient with a complete read, then use a disposable discovery sandbox;
2. return to the paper and freeze the printed method and causal claim map;
3. execute questionable printed steps rather than silently correcting them;
4. predeclare every material interpretation before formal outcomes are seen;
5. keep exploratory work, reasonable repairs, and controlled analyses visibly
   separate from paper-literal evidence;
6. prefer cheap discriminating tests before expensive model-family coverage;
7. preserve every seed, failure, timing, configuration, and raw result; and
8. issue bounded numerical, mechanism, and attainability findings without
   inferring intent or claiming an infinite space was exhausted.

The working hypotheses are paper-specific and genuinely falsifiable. A stable
numerical reproduction, a component that passes capability-sensitive causal
tests, or a target that falls inside the observed envelope must each be
reported plainly. Suspicion chooses what to inspect; it does not determine the
result.

## Studies

| Study | Paper | State |
|---|---|---|
| [`atk-2022-deep-autoencoder`](studies/atk-2022-deep-autoencoder/) | Takiddin et al., “Deep Autoencoder-Based Anomaly Detection of Electricity Theft Cyberattacks in Smart Grids” | Active source reconstruction and reproduction |
| [`tlstgt-2025-water`](studies/tlstgt-2025-water/) | Ahasan et al., “Graph Transfer Learning-Based Attack Detection in Cyber-Physical Water Distribution Systems” | Artifact-level verdict recorded; execution frozen while Paper 1 is active |
| Paper 3 | To be registered independently | Not started |

Open the standalone
[`Paper 1 method map`](site/papers/atk-2022-deep-autoencoder/index.html)
to see, in one document, what the first paper says happens from raw data to
Tables I–V.

## Repository layout

```text
studies/<study-id>/       code, claims, configurations, and results for one paper
reports/<study-id>/       LaTeX source for that paper's scientific report
reports/synthesis/        later cross-paper report
site/                     public project site and readable method maps
docs/                     project protocol, status, decisions, and evidence ledger
data/ and papers/         local-only inputs; never committed
```

## Start here

```bash
git clone https://github.com/fjoad/atk-evidence.git
cd atk-evidence
bash scripts/bootstrap.sh
```

Then read [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) and the target
study's README. Raw datasets and publication PDFs are not redistributed.
Acquisition instructions, access conditions, provenance, and checksums are
recorded per study.

Agents and researchers implementing a paper should begin with the single
end-to-end [`RUNBOOK.md`](RUNBOOK.md). It defines the initial paper read,
discovery sandbox, source freeze, five-file instrument, diagnostic breadth,
promoted numerical/mechanism/attainability work, confirmation, and reporting
sequence.

## About the implementation size

The current Paper 1 directory has two different concerns:

- four short researcher-facing commands for download, preparation, execution,
  and analysis; and
- a much larger internal forensic harness that enumerates ambiguous readings,
  verifies source-to-code fidelity, records immutable attempts, and supports
  cluster execution.

The genuine five-file reference track—download, prepare, models, run, and
analyze—now exists independently of the forensic `src/` tree. The larger harness
remains historical audit evidence, but it is neither an authority nor a
dependency of the compact route and is not the amount of code needed to
implement the paper.

## Reports and website

Each completed paper will have a conventional LaTeX report under
`reports/<study-id>/` with separate numerical, mechanism, and attainability
findings. GitHub Pages can host the project landing page, the method maps, and
rendered report PDFs at one project URL:

```text
https://fjoad.github.io/atk-evidence/
```

The repository contains one Pages site with a separate path for each paper;
separate repositories would be required only if each paper needed its own
independent GitHub Pages project.

See [`reports/README.md`](reports/README.md) for the report build and publishing
convention, [`docs/VISION.md`](docs/VISION.md) for the research thesis, and
[`docs/STATUS.md`](docs/STATUS.md) for the current state.
