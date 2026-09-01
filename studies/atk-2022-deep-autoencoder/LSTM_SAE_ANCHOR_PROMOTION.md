# LSTM-SAE anchor promotion and hardware-cost contract

**Date:** 2026-09-01

**Status:** approved for one operational H200 cost pilot. The full anchor is
conditional on the frozen promotion gates below.

**Local freeze gate:** all 256 repository tests (140 study and 116 root),
strict data verification, Python compilation, shell syntax, and whitespace
checks pass. The historical paper-facing implementation hashes remain exact.

**Evidence type:** adaptive operational `X` for costing. A later full anchor,
if promoted, is the primary `P+I/N` completion `CR-ISET-LSTMSAE-01` already
defined in `REMAINING_PAPER_CONTRACT.md`.

## Existing evidence and cost

The preserved A16 feasibility attempt used the exact approved architecture,
seed, batch 32, 32,768 fitting rows, and two epochs. Its log records 218 and
207 seconds per epoch. The audited score recovery records 3.913664639 seconds
for 12,119 rows at batch 256. Applying the already frozen projection formula,

`1.5 * (epochs * slowest_pilot_epoch * 1,500,523 / 32,768 + score_time * 8,884,989 / 12,119)`,

gives approximately 42.7902 hours for ten epochs and 417.1425 hours for 100
epochs. The possible full run therefore does not fit the 72-hour gate on the
measured A16. This is a conservative operational projection, not an observed
full-data runtime.

## Adaptive prospective score-stability rule

The original absolute `1e-6` all-score gate remains a preserved failure. It is
not rewritten. For this new cost decision, score stability is judged by the
scientific decisions the score feeds. Across batches 256, 128, 64, and 32 on
the exact saved 12,119-row selection, require:

- finite scores and two or more memory-safe batches;
- zero label changes at the printed threshold 0.61;
- maximum one changed label when a batch-256 FA-capped threshold is transferred
  unchanged to another batch;
- independently optimized DR and FA at both FA<=15% and FA<=15.5% differing
  by no more than one malicious or benign row respectively; and
- AUC differing by no more than 0.001 percentage points.

These limits were chosen after the A16 score-recovery result and are therefore
adaptive operational criteria, not confirmatory evidence about score
determinism. They govern only whether numerical batch noise blocks a later
full fit. The raw score differences remain recorded regardless.

## H200 cost pilot

Run exactly one immutable pilot on one H200 GPU:

- exact data cache and source selection from attempt `5f53ca7217aa`;
- root seed `20260824` and the same initialization/shuffle streams;
- unchanged `CR-ISET-LSTMSAE-01` architecture and decoder completion;
- batch 32, 32,768 fitting rows, two epochs;
- the exact 12,119 scoring rows and batches 256/128/64/32;
- strict deterministic CUDA, 16 CPUs, 96 GiB RAM, `gpu-H200`, and at most two
  hours; and
- immutable config, weights, history, scores, result/failure, and hashes.

The pilot must verify all consumed data, selection, implementation, and source
record identities. It may not load the A16 fitted weights; it must recreate the
same seeded two-epoch timing workload from initialization.

## Full-anchor promotion

Promotion requires all of the following:

1. two complete finite epochs and non-identical initial/fitted weights;
2. exact saved-weight reload;
3. all adaptive decision-stability gates above;
4. peak resident and GPU memory no more than 75% of their allocations; and
5. the existing 1.5-times conservative projection, including full scoring, is
   no greater than 72 hours for the possible 100-epoch run.

If any gate fails, preserve the cost result and stop. Do not increase batch,
GPU count, memory, timeout, or change model/data/training semantics. If all
pass, freeze a separate full-anchor implementation and wrapper, pass the full
repository/data gates, and launch one watched 72-hour attempt. No result from
the cost pilot is numerical paper evidence.
