# Robust-poisoning paper source and arithmetic audit

**Date:** 2026-09-02

**Status:** Complete; stopped at the source-only checkpoint for discussion

**Evidence question:** source reconstruction and cheap numerical (`N`)
consistency checks. No trained-model finding is authorized.

**Implementation semantics:** paper-source transcription (`P`) plus a
controlled static audit (`C`).

## Goal

Establish an independent, source-located specification for Takiddin et al.,
“Robust Electricity Theft Detection Against Data Poisoning Attacks in Smart
Grids,” preserve every printed result in Tables II–V, and test the internal
arithmetic of Tables III–V before any data preparation or model training.

The paper's own runtime statements are part of the claim. Any later numerical
attempt must first freeze the stated epochs, batch size, accelerator, and
training-time envelope. Modern or additional hardware may not be introduced as
an outcome-driven rescue.

## Current authorization

This phase may:

- register the paper and its exact local PDF fingerprint;
- write `METHOD.md`, including the numerical targets and `A/B/Z/S` claim map;
- transcribe Tables II–V into reviewable records;
- predeclare and run deterministic arithmetic checks on the printed numbers;
- record the bounded source-only result; and
- inspect whether the named data are already available, without preparing them.

This phase may not:

- prepare, resample, or score a dataset;
- implement or train a detector;
- submit a cluster job;
- reuse a Paper-1 result as evidence about this paper;
- infer an omitted procedure from whichever choice approaches the target; or
- make a cross-paper or author-intent conclusion.

## Source identity

- Local PDF: `papers/his/Robust_Electricity_Theft_Detection_Against_Data_Poisoning_Attacks_in_Smart_Grids.pdf`
- Pages: 10
- DOI: `10.1109/TSG.2020.3047864`
- SHA-256: `03a372fb5ee129799d43a50e6bec8f86fb108e9f8bdc85350e3ffe3e13d4a4ff`
- Published: *IEEE Transactions on Smart Grid* 12(3), May 2021

## Paper-time rule

The source is unusually specific about its main hardware and budget:

- deep detectors: 50 epochs, batch size 100;
- accelerator: NVIDIA GeForce RTX 2070;
- shallow-detector training: approximately one hour;
- deep-detector training: approximately 1.5–3 hours;
- ensemble-averaging training: approximately three hours;
- sequential-ensemble training: approximately four hours; and
- online testing: approximately two seconds per reported decision.

A later paper-time attempt must use an RTX 2070 if reasonably available. If it
is unavailable, a substitute may be admitted only before results through a
documented, conservative throughput calibration against a period-appropriate
device. The fitting boundary, inclusion or exclusion of hyperparameter search,
preparation, and scoring, one-device limit, seed, and stopping rule must be
frozen first. A result outside the paper-time budget cannot answer the bounded
paper-time question.

If a future paper omits hardware, select a favorable contemporaneous envelope
from primary records before execution. Never use H200/A100/multiple GPUs or a
retry merely because the declared result missed.

## Steps

### 1. Source freeze

- [x] Inspect all ten rendered pages, including Tables I–V, Figures 1–3, and
  Algorithm 1.
- [x] Record bibliographic identity and PDF checksum.
- [x] Complete `METHOD.md` with source locators and material omissions.
- [x] Record the causal claims as `B > A because Z exploits S`.

### 2. Printed table freeze

- [x] Transcribe Tables II–V independently from the rendered pages.
- [x] Verify model, metric, and poisoning-level coverage.
- [x] Freeze the transcription before executing the arithmetic checker.

### 3. Static arithmetic audit

- [x] Apply the predeclared one-decimal rounding intervals.
- [x] Check `SP = 100 - FA`.
- [x] Check `F1 = harmonic_mean(DR, PR)`.
- [x] Check balanced-test `ACC = (DR + SP)/2`.
- [x] Check balanced-test `PR = DR/(DR + FA)`.
- [x] Compare prevalence ranges implied independently by precision and
  accuracy, without assuming balance.
- [x] Recalculate prose degradation and model-order claims from the printed
  tables.
- [x] Preserve both matches and failures in a machine-readable record.

### CHECKPOINT: source-only discussion

Stop after the source finding. The user must inspect the transcription,
ambiguities, arithmetic result, proposed first anchor, and its measured or
projected RTX-2070 cost before authorizing any data preparation or training.

### 4. Later work, not yet authorized

- exact-data identity and 3,000-customer selection;
- disposable poison/mechanism sandbox;
- five-file paper-facing implementation;
- one cheap full anchor;
- matched generalized/customer-specific and sequential/averaging mechanism
  tests; and
- finite numerical and attainability depth.

## Verification

- [x] Every consequential source statement has a page/section/table locator.
- [x] Printed, interpreted, controlled, and exploratory work remain separate.
- [x] Tables III–V contain all 476 printed metric cells exactly once.
- [x] Static audit has deterministic tests and a machine-readable result.
- [x] No experimental preparation, training, or scoring occurred locally.
- [x] Repository test suite passes.
- [x] `docs/STATUS.md` and `docs/CONTEXT.md` name this checkpoint.
- [x] Changes are committed before any later execution contract is written.

## Finish condition

This phase finishes when the source specification, causal map, exact printed
tables, and bounded static finding are committed and the project is stopped for
discussion. A runnable model is deliberately not part of this finish condition.
