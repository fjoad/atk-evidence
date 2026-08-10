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
- No local or cluster experimental job is running.
- Experimental preparation, training, and scoring must run on cluster compute
  nodes. Local work is limited to source reconstruction, code, documentation,
  lightweight inspection, transfer, and monitoring.
- The working tree contains an unfinished, uncommitted Paper 1 score-audit change
  in `reproduction/analyze_results.py`. It must be finished and verified or
  discarded before another result is interpreted. It is not part of the current
  documentation commit.
- Experimental execution remains frozen at a renewed source-freeze checkpoint.
  No local or cluster job should start until the user accepts the corrected
  source reconstruction and the five-file route is audited against it.

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

Do not launch or interpret another model yet.

1. Present the pivotal 2026-08-11 source corrections and obtain the renewed
   source-freeze checkpoint approval.
2. Audit the five direct reproduction files line by line against the corrected
   `METHOD.md`; remove or relabel any source mismatch without adding
   infrastructure.
3. Resolve the unfinished `analyze_results.py` score-audit change.
4. Run the compact Paper 1 Step-0/eligibility gate on the cluster.
5. Inspect and classify the existing full seed-11 attempt before choosing the
   batch-512 primary probe.

## Not on the critical path

- the historical 921-configuration branch matrix;
- new scheduling, manifest, DDP, or workflow infrastructure;
- Paper 2 execution;
- website or LaTeX polishing;
- the corrected preferred detector; and
- cross-paper conclusions.
