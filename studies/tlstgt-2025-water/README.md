# Study 2 — Graph Transfer Learning for Water-Network Attack Detection

Audit of Ahasan, Joad, Atat, Thompson, Serpedin, Takiddin, *"Graph Transfer
Learning-Based Attack Detection in Cyber-Physical Water Distribution Systems"*,
EUSIPCO 2025.

> **Conflict of interest.** Faaiz Joad, a maintainer of this project, is the
> second author of the audited paper. Every claim here is therefore held to
> pre-registered ambiguity branches, published corrections of this audit's own
> errors, and verdicts bounded strictly to what the artefact supports. The audit
> does not assert how the reported numbers arose.

**Verdict: `no-consistent-protocol`.** An exhaustive pre-registered search over
67,326 measurement protocols per cell finds none that reproduces the reported
table; 20 of 27 cells are unreachable under every protocol searched.

## What this project is *not*

It is not "implement the paper and compare numbers." That is too weak — it
invites *"you implemented it wrong"*, and during this audit that reply was
correct twice. Both instances are documented in `EVIDENCE.md` as retractions
rather than quietly amended.

The audit instead asks what a reviewer would ask who doubts every item in the
paper: does it cohere with itself, is the task non-trivial at all, does each
claimed component do work, are the comparisons fair, is the evaluation sound —
and only then, does the headline pattern reproduce.

## Read in this order

| File | Contents |
|---|---|
| [`DATA.md`](DATA.md) | inputs, checksums, provenance, how to point the code at your copy |
| [`EVIDENCE.md`](EVIDENCE.md) | durable causal record — every finding, correction, and retraction |
| [`AMBIGUITY_REGISTER.md`](AMBIGUITY_REGISTER.md) | ambiguity axes, pre-registered before results were seen |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | the paper's method transcribed line by line |
| [`PAPER_SPEC.md`](PAPER_SPEC.md) | fidelity audit: what the paper says vs what we implemented |
| [`../../reports/tlstgt-2025-water/main.tex`](../../reports/tlstgt-2025-water/main.tex) | the report |

## Running it

```bash
export WATER_DATA=/path/to/data        # see DATA.md
python src/test_detect.py              # detector regression tests
python src/sanity.py --size 31         # breadth checks; trains nothing
python src/forensics.py                # forensics on the reported table
python src/protocol_search.py          # exhaustive protocol search
python src/run.py --help               # the full model comparison
```

`run.py` exposes every pre-registered ambiguity axis as a flag: `--delta-scale`,
`--train`, `--thresh`, `--transformer`, `--errfit`, `--adj`, `--window`,
`--ablate`, `--batch-s`. Defaults are the paper's literal reading.

**Reporting rule:** always state which configuration produced a number. Training
on the paper's literal 50/50 split and training on benign windows only differ by
roughly 25 F1 points, so an unattributed figure is meaningless.

## Files

`src/data.py` inputs, Figure-1 graphs, attack synthesis · `src/models.py` the
nine detectors · `src/detect.py` residual → Mahalanobis → threshold → metrics ·
`src/run.py` orchestration · `src/sanity.py` zero-parameter checks ·
`src/depth_auc.py`, `src/depth_replay_capacity.py` aimed probes ·
`src/forensics.py`, `src/protocol_search.py` analysis of the reported table ·
`src/test_detect.py` regression tests · `results/` raw outputs.
