# ATK Evidence — Current Status

**Last updated:** 2026-08-09

**Branch:** `main`

**Active plan:**
[`plans/2026-08-09-paper-1-minimal-finish.md`](plans/2026-08-09-paper-1-minimal-finish.md)

## Current project state

- The paper-first minimal-instrument reframe is accepted and recorded in
  [`decisions/2026-08-09-paper-first-minimal-instrument.md`](decisions/2026-08-09-paper-first-minimal-instrument.md).
- Paper 1 is the only active execution target. Paper 2's existing artifact-level
  audit remains preserved but frozen; cross-paper synthesis has not started.
- No local or cluster experimental job is running.
- Experimental preparation, training, and scoring must run on cluster compute
  nodes. Local work is limited to source reconstruction, code, documentation,
  lightweight inspection, transfer, and monitoring.
- The working tree contains an unfinished, uncommitted Paper 1 score-audit change
  in `reproduction/analyze_results.py`. It must be finished and verified or
  discarded before another result is interpreted. It is not part of the current
  documentation commit.

## Paper 1: verified foundation

- The 12-page PDF was freshly read and visually inspected; the source-located
  executable reconstruction is
  [`../studies/atk-2022-deep-autoencoder/METHOD.md`](../studies/atk-2022-deep-autoencoder/METHOD.md).
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
  neighbor query entails roughly 10.7 trillion distances. Preserve this as an
  executability finding. Do not relabel the no-resampling interpretation as the
  printed result.

## Paper 1: current experimental evidence

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

Do not launch another model yet.

1. Resolve the unfinished `analyze_results.py` score-audit change.
2. Run the compact Paper 1 Step-0/eligibility gate on the cluster.
3. Inspect the full seed-11 score arrays, directions, baselines, attack breakdown,
   identities, and persistence.
4. Classify that attempt explicitly.
5. Discuss and freeze the minimal batch-512 primary resource probe, then run one
   watched full primary anchor before parallel seeds.

## Not on the critical path

- the historical 921-configuration branch matrix;
- new scheduling, manifest, DDP, or workflow infrastructure;
- Paper 2 execution;
- website or LaTeX polishing;
- the corrected preferred detector; and
- cross-paper conclusions.
