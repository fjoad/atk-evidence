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

1. Commit and push the audited compact baseline.
2. Pull that exact commit on Panther and submit the one short
   `run_baseline.sbatch` wrapper.
3. Inspect the saved architecture, loss history, score distributions, zero and
   untrained controls, hashes, metrics, and Table-V FA invariant before any
   second seed or interpretation branch.
4. Classify the batch-512 attempt independently of the old batch-32 sensitivity.

## Not on the critical path

- the historical 921-configuration branch matrix;
- new scheduling, manifest, DDP, or workflow infrastructure;
- Paper 2 execution;
- website or LaTeX polishing;
- the corrected preferred detector; and
- cross-paper conclusions.
