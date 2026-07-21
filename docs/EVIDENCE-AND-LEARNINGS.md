# ATK Evidence — Evidence and Causal Learnings

**Last updated:** 2026-07-21

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
- **Supporting evidence:** No confirmatory experiment has yet been completed. Static inconsistencies and omissions motivate testing but do not establish the hypothesis.
- **Root cause:** Not yet investigated experimentally.
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
- **Disconfirming or supporting evidence:** SGCC was acquired from the author-linked repository and checksum-verified. Official ISSDA metadata marks the CER consumption archives restricted; no authorized copy was found locally.
- **Root cause:** CER access requires an approved institutional account rather than an anonymous public download.
- **Current conclusion + label:** **VERIFIED** for current local availability; CER acquisition is **OPEN**.
- **Remaining uncertainty / blast radius:** Exact CER reproduction cannot run until authorized archives are supplied and match official MD5 values.
- **Source artifacts:** `studies/atk-2022-deep-autoencoder/DATA_SOURCES.md`, `data/raw/sgcc-verified/data.csv` (local, ignored by Git).

## How to add a learning

Use: former belief/status; evidence; root cause if isolated; current conclusion
with label; remaining uncertainty; and source artifacts. Preserve invalidated
beliefs rather than deleting them.
