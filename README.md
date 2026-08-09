# ATK Evidence

**→ [atk-evidence site: verdicts, reports, and method maps](https://fjoad.github.io/atk-evidence/)**

ATK Evidence is an audit pipeline for published results that do not look right.
Papers enter, get rebuilt from scratch against the printed method, and leave
with a verdict — including harsh ones, where the evidence supports them.

Most reproduction work asks *"do the numbers come out?"* That question is too
weak: it invites the reply *"you implemented it wrong"*, and sometimes that
reply is correct. This project instead audits a paper the way a reviewer would
who doubts every item in it and checks each one personally.

## Operating philosophy

The difficult part is reconstructing the paper correctly. Every page, equation,
algorithm, figure, table, and material omission is turned into a source-located
executable specification before model code is written. If that reading is wrong,
everything downstream is evidence about an experiment the paper never claimed.

The software is deliberately small. Each paper uses five direct scientific files
for download, preparation, models, execution, and analysis. Shared code performs
mechanical work only; the meaning of the paper remains visible in its study. The
project does not build a general experiment platform before producing eligible
results.

The full rationale and working rules are in the
[`paper-first minimal-instrument decision`](docs/decisions/2026-08-09-paper-first-minimal-instrument.md).

## Conflict of interest

Faaiz Joad, a maintainer of this project, is the second author of the 2025 water
paper audited as study 2. That study's verdict is therefore held to
pre-registered ambiguity branches, published corrections of the audit's own
errors, and conclusions bounded strictly to what the artefact supports. The
project does not assert how any reported numbers arose.

## Papers in the pipeline

| # | Paper | Year | Verdict |
|---|---|---|---|
| 1 | Deep Autoencoder-Based Anomaly Detection of Electricity Theft Cyberattacks in Smart Grids | 2022 | `in progress` |
| 2 | Graph Transfer Learning-Based Attack Detection in Cyber-Physical Water Distribution Systems | 2025 | `no consistent protocol` |

Each paper has a study directory (`studies/<id>/`), a report
(`reports/<id>/main.tex`), and a verdict page on the site. The verdict
vocabulary and its evidentiary bars are defined in `studies/registry.toml`.

## The pipeline

Tiers are ordered so a failure low down makes everything above it moot.

0. **Does the paper cohere with itself?** No computation.
1. **Is the problem non-trivial?** Measure a zero-parameter rule through the
   identical evaluation path first. If it performs comparably, every downstream
   comparison is moot — including ours.
2. **Is each claimed component doing work?** Destroy one claimed structure at a
   time and see whether anything notices.
3. **Are the comparisons fair?** A margin over an undertrained baseline is not a
   margin.
4. **Is the evaluation sound?** Per-class detectability, seed variance,
   threshold sensitivity.
5. **Only now, the headline claims** — with intervals, not point estimates.

Full protocol: [`RUNBOOK.md`](RUNBOOK.md).

## Research rule

For every paper:

1. reconstruct the printed method before running experiments;
2. execute questionable printed steps rather than silently correcting them;
3. branch every material ambiguity before looking at outcomes;
4. keep scientifically corrected controls separate from reproduction evidence;
5. preserve every seed, failure, timing, configuration, and raw result; and
6. state only what was or was not reproduced inside the tested finite space.

The working hypothesis is that the selected papers' complete numerical result
patterns will not reproduce reliably under faithful, documented
implementations. It is a hypothesis, not a verdict or an allegation. A stable
reproduction is a valid and welcome falsification.

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
end-to-end [`RUNBOOK.md`](RUNBOOK.md). It defines the PDF-first method freeze,
five-file reproduction, sanity runs, full experiments, ambiguity tests,
confirmatory assessment, and reporting sequence.

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
`reports/<study-id>/`. GitHub Pages can host the project landing page, the
method maps, and rendered report PDFs at one project URL:

```text
https://fjoad.github.io/atk-evidence/
```

The repository contains one Pages site with a separate path for each paper;
separate repositories would be required only if each paper needed its own
independent GitHub Pages project.

See [`reports/README.md`](reports/README.md) for the report build and publishing
convention, [`docs/VISION.md`](docs/VISION.md) for the research thesis, and
[`docs/STATUS.md`](docs/STATUS.md) for the current state.
