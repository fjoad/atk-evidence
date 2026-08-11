# ATK Evidence — Current Status

**Last updated:** 2026-08-11

**Branch:** `main`

**Active plan:**
[`plans/2026-08-09-paper-1-minimal-finish.md`](plans/2026-08-09-paper-1-minimal-finish.md)

## Current project state

- The paper-first minimal-instrument reframe is accepted and recorded in
  [`decisions/2026-08-09-paper-first-minimal-instrument.md`](decisions/2026-08-09-paper-first-minimal-instrument.md).
- Paper 1 is the only active execution target. Paper 2's existing artifact-level
  audit remains preserved but frozen; cross-paper synthesis has not started.
- Panther job `373789` completed successfully in 53:12 from commit `c8c136f`
  on one 16-GB V100. It passed source/preparation gates, trained the frozen
  batch-512 FC-SAE for 74 epochs (best epoch 69), and saved Tables III/full-IV/V
  artifacts. Panther CPU job `373799` measured the selected exact ADASYN
  neighbor implementation, and score-audit job `373800` completed. Premature
  repeated-seed jobs `373803` and `373804` were cancelled before execution
  after the breadth-first correction. One-factor linear-output job `373805`
  is running on one 16-GB V100. It had already been submitted with an eight-CPU
  request; future wrappers request only the GPU so CPU shape cannot delay them.
- Experimental preparation, training, and scoring must run on cluster compute
  nodes. Local work is limited to source reconstruction, code, documentation,
  lightweight inspection, transfer, and monitoring.
- The renewed source freeze is accepted. The five direct files have now been
  traced against the corrected `METHOD.md`; the unfinished score-audit work was
  completed without importing the historical `src/` route.
- The next experiment is frozen to the single baseline in
  [`../studies/atk-2022-deep-autoencoder/reproduction/BASELINE.md`](../studies/atk-2022-deep-autoencoder/reproduction/BASELINE.md):
  full ISET, FC-SAE, seed 11, batch 512, original `B2+M`, Tables III/full-IV/V.
  No ambiguity sweep is authorized before this anchor is inspected.

## Paper 1: verified foundation

- The exact 12-page PDF was independently re-audited on 2026-08-11: SHA-256
  verified, text extracted, and every rendered page visually inspected before
  the prior reconstruction was opened. The corrected source-located executable
  reconstruction is
  [`../studies/atk-2022-deep-autoencoder/METHOD.md`](../studies/atk-2022-deep-autoencoder/METHOD.md).
- The re-audit confirmed the main flow and prior pivotal contradictions, fixed
  the benchmark count (six, not seven), and added omitted VAE-derivation,
  decoder, precision-definition, F1, and common-prevalence inconsistencies.
- `P0-ISET-FCSAE` is now labeled precisely as a paper-primary `P+I` executable
  completion. The printed Attack-3 subtraction remains a non-executable source
  outcome rather than being silently called literal.
- The accepted non-executable-source contract requires the literal failure and
  every predeclared reasonable repair to be executed and reported side by side
  with the published target; see
  [`decisions/2026-08-11-non-executable-source-ladder.md`](decisions/2026-08-11-non-executable-source-ladder.md).
- Exact CER/ISET consumption archives are verified. The allocation CSV is the
  explicitly labeled semantic-equivalence branch, not the official `.tab`
  serialization.
- The five direct reproduction files exist under
  `studies/atk-2022-deep-autoencoder/reproduction/` and do not import the
  historical forensic `src/` implementation.
- Full preparation produced 2,251,290 benign profiles, 13,507,740 attacked
  profiles, 1,500,520 B1 training profiles, and the 14,258,510-row `B2+M` test
  population.
- Printed-position default ADASYN is not complete: its exact default full-scale
  neighbor query entails roughly 10.7 trillion first-pass distances. A 16-core
  same-machine benchmark estimates 14.16 wall-hours for both exact neighbor
  searches alone, before synthesis and persistence. This is expensive but
  feasible as an overnight job and gives no basis for claiming that the authors
  could not have run ADASYN. Do not relabel the no-resampling interpretation as
  the printed result.

## Paper 1: current experimental evidence

The new compact batch-512 anchor completed:

- method: `I-ADASYN-NONE-ISET-FC-SAE`, seed 11, batch 512;
- reproduced DR/FA/ACC/AUC/F1:
  26.18 / 58.22 / 33.98 / 31.04 / 40.46%;
- reported DR/FA/ACC/AUC/F1: 81 / 15 / 83 / 81 / 81%;
- fit/total time: 45:07 / 47:41 inside the pipeline; and
- Table-V FA is exactly 57.9152% for all six attacks, as required by the frozen
  common-model/common-benign interpretation but unlike the paper's varying FA.

This is a completed exploratory no-resampling anchor, not printed-ADASYN `P0`
and not yet a confirmatory verdict. Its score-distribution/reload audit is now
complete: an oracle threshold in the paper direction reaches only 50.00%
balanced ACC; reversing direction reaches 66.26%; and the trained score ranking
is 0.99946-correlated with the zero-reconstruction control. The committed
result records are
[`../studies/atk-2022-deep-autoencoder/results/compact_route_fc_sae_seed11_batch512_20260811.json`](../studies/atk-2022-deep-autoencoder/results/compact_route_fc_sae_seed11_batch512_20260811.json)
and
[`../studies/atk-2022-deep-autoencoder/results/iset_fc_sae_seed11_score_audit_20260811.json`](../studies/atk-2022-deep-autoencoder/results/iset_fc_sae_seed11_score_audit_20260811.json).

One full compact-route cluster result exists:

- table/model: Table III, FC-SAE;
- method: `I-ADASYN-NONE-ISET-FC-SAE`;
- seed/batch: 11 / 32;
- population: exact original pre-ADASYN `B2+M` rows;
- epochs: 29;
- reproduced DR/FA/ACC/AUC/F1:
  26.44 / 58.51 / 33.97 / 31.03 / 40.78%; and
- reported DR/FA/ACC/AUC: 81 / 15 / 83 / 81%.

This is one exploratory batch-32 sensitivity, not a reproduction verdict and not
the batch-512 primary branch. Its full score arrays, weights, history, predictions,
and hashes are local. Its score/eligibility audit is unfinished.

## Exact next action

1. Complete one seed of the separately labeled linear-output FC-SAE control in
   Panther job `373805`; it tests the standardized-input/Softmax-output
   incompatibility and nothing else. On completion, validate and compare its
   saved scores before submitting another job.
2. Continue breadth-first, one watched result at a time: LSTM-SAE, FC-VAE,
   LSTM-VAE, LSTM-AEA; the six benchmark rows; then the one-factor population,
   split, scaling, threshold, and Attack-3 interpretations. Existing historical
   results count only if they pass the renewed source/provenance/score gates.
3. Record the complete breadth map. Only then freeze the finite surviving
   branches and deepen them with repeated seeds and confirmatory intervals.

## Not on the critical path

- the historical 921-configuration branch matrix;
- new scheduling, manifest, DDP, or workflow infrastructure;
- Paper 2 execution;
- website or LaTeX polishing;
- the corrected preferred detector; and
- cross-paper conclusions.
