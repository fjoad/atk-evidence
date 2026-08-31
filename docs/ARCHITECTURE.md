# Where the research lives

This repository keeps the public explanation, the paper-specific implementation,
and the evidence record separate so that a reader can trace every claim.

```text
paper and official data
        ↓
source reading and explicit assumptions
        ↓
five direct implementation files
        ↓
saved run: configuration, scores, metrics, failures, timing
        ↓
artifact checks and analysis
        ↓
README, website, and scientific report
```

## Public reading path

- [README.md](../README.md) gives the result and the simplest explanation.
- [site/index.html](../site/index.html) is the public landing page.
- Each paper page presents its question, methods, results, explanations, and
  limitations.
- Rendered PDFs under `site/reports/` are dated reports. Earlier reports are
  labeled as such when a later result supersedes them.

## Paper directories

Every paper has a stable directory under `studies/` and an entry in
`studies/registry.toml`.

The current Paper 1 implementation is deliberately direct:

```text
studies/atk-2022-deep-autoencoder/reproduction/
  download_data.py
  prepare_data.py
  models.py
  run_experiment.py
  analyze_results.py
```

The source reading and assumptions sit beside those files. Small result JSON
files are versioned; large arrays, weights, raw datasets, and copyrighted PDFs
remain outside Git. Older `src/` code and study-root wrappers are retained as
historical audit material but do not define the current experiment.

## Durable records

- `docs/STATUS.md`: what is true now and the next decision.
- `docs/CONTEXT.md`: compact facts needed after a handoff.
- `docs/EVIDENCE-AND-LEARNINGS.md`: why an earlier conclusion changed.
- `docs/decisions/`: choices that changed evidence or interpretation.
- `docs/plans/`: current and historical execution plans.
- A study's source specification: what the paper says and every necessary
  completion.
- A study's finding: the result, checks, diagnostics, and conclusion boundary.
- A study's explanation register: competing causes and tests that distinguish
  them.

## Labels used inside the record

Internal scientific records may label an implementation as printed,
interpreted, controlled, or exploratory, and a question as numerical,
mechanistic, or attainability-related. Public pages translate those labels into
ordinary language. Labels never replace an explanation.

## Rule for changing this structure

Add a shared abstraction only after repeated paper work demonstrates a real
mechanical need. Do not build a general experiment platform in anticipation of
reuse. A new paper gets its own source reading, evidence contract, code, and
finding before it enters cross-paper synthesis.
