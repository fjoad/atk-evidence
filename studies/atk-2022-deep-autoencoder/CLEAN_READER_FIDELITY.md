# Clean-reader five-file fidelity assessment

**Date:** 2026-08-24

**Status:** Phase 4 complete; corrected route and quarantine verified by 178
deterministic tests (140 study tests plus 38 root tests)

**Anchor:** `CR-ISET-FCSAE-01`

**Evidence classification:** numerical (`N`), paper-literal/reasonable-reader
implementation (`P/I`). This document contains a static implementation audit,
not an experimental result.

## 1. Question and authority

Does the direct five-file route execute every consequential item in
[`CLEAN_READER_SPECIFICATION.md`](CLEAN_READER_SPECIFICATION.md) without using
historical results to choose an interpretation?

The paper and approved clean-reader freeze are the scientific authority. The
existing implementation is reusable only where it agrees with that freeze.
Historical output is never evidence that a mismatching route is correct.

## 2. File-level disposition

| Direct file | Matching parts | Material mismatch or open check | Disposition |
|---|---|---|---|
| `reproduction/download_data.py` | Exact six official archive identities; exact official allocation `.tab` identity; non-overwriting checksum gate | General verifier silently selects the semantic ScienceDB branch when the official `.tab` is absent | Reuse checksum/download code; the clean-reader route must explicitly require `official-tab-v1` |
| `reproduction/prepare_data.py` | strict 48-slot days; all residential meters; attacks 1, 2, 4, 5, and 6; joint feature-wise `B+M` scaling; customer-disjoint 2:1 benign split; test ADASYN with five neighbors; provenance arrays | hard-coded ScienceDB allocation CSV; seed 11; clipped Attack 3; test attacks from all customers rather than `B2`; cache identity names the historical procedure | Correct only those frozen fields; preserve the former route as an explicitly ineligible historical interpretation |
| `reproduction/models.py` | FC-SAE widths, eight sigmoid hidden layers, dropout after all eight, Softmax reconstruction, MSE, Adam `1e-3`, Glorot kernels, zero biases, parameter count | Backend deterministic-operation state is not recorded by this file | Reuse FC-SAE unchanged; record/enforce runtime determinism in the runner where supported |
| `reproduction/run_experiment.py` | per-row mean MSE; high-score anomaly rule; strict `>0.58`; printed confusion/metric formulas; continuous-score AUC; immutable configuration/result directories; weights, history, scores, predictions, timings, runtime | defaults are seed 11, batch 512, `min_delta=1e-4`, and original unresampled test; no ten-epoch minimum; result is labeled exploratory; no contract guard; zero-reconstruction diagnostic is not the Softmax projection floor | Add one guarded clean-reader contract with the frozen training/test values and an exact Softmax-domain floor; retain generic historical execution only under an explicit exploratory contract |
| `reproduction/analyze_results.py` | loads every success/failure; groups configurations without selecting a seed; audits saved score direction, thresholds, distributions, complete metric-vector proximity, and per-attack behavior | historical eligibility inference cannot recognize the new anchor; audit does not independently check every frozen contract field or recompute the saved result from reloaded weights | Add clean-reader eligibility recognition and a strict anchor contract/artifact audit; do not merge the anchor with historical groups |

## 3. Consequential source-to-code trace

The last column preserves the pre-correction status; every `MISMATCH` listed
below was corrected without using an experimental outcome.

| Frozen instruction | Static trace | Pre-correction status |
|---|---|---|
| exact official ISSDA V1 archives and allocation `.tab` | the six local mirror archives exactly match the official bytes and MD5 values; preparation selected the semantic allocation CSV | `MISMATCH`; corrected archive identity is usable, but the exact `.tab` remains `BLOCKED` locally because it is absent |
| all officially residential meters; complete interval | default population is all 4,225 labels and all six archives are parsed in full mode | `MATCH` after official source selection |
| strict chronological 48-slot customer-days | `strict_profiles` requires exactly unique slots 1–48 and drops all other days | `MATCH` |
| root seed `20260824` with separated attack streams | separated deterministic streams exist; CLI default is 11 | `MISMATCH` |
| Attack 1 one multiplier per customer matrix | one alpha is indexed by meter across all its retained days | `MATCH` |
| Attacks 2 and 5 one multiplier per reading | one continuous draw per array coordinate | `MATCH` |
| Attack 3 duration first, then a valid within-day start | implementation draws start first, draws duration independently, then clips at hour 24 | `MISMATCH` |
| Attack 4 daily arithmetic mean | every row is replaced by its 48-coordinate mean | `MATCH` |
| Attack 6 within-day reversal | each row is reversed across its 48 coordinates | `MATCH` |
| joint feature-wise population standardization before split | scaler accumulates benign and all six attack populations with `ddof=0` | `MATCH` |
| disjoint meter-level `B1/B2`; attacks in test only from `B2` | benign split is meter-disjoint, but all-customer attacks enter the test set | `MISMATCH` |
| ADASYN inside test, `k=5`, seed `20260824` | algorithm and default neighbors match; seed follows the mismatching root default | `MISMATCH` until seed corrected; computational feasibility remains `OPEN` |
| FC-SAE architecture and optimizer | direct layer inventory agrees exactly with the freeze | `MATCH` |
| batch 32, maximum 100, minimum 10, patience 5, `min_delta=1e-6`, restore best | maximum, patience, and restoration match; batch, delta, and minimum do not | `MISMATCH` |
| mean row MSE; anomaly iff `score>0.58` | scoring and strict comparison agree | `MATCH` |
| all seven printed metrics | formulas agree with the frozen paper reading | `MATCH` |
| exact Softmax output-domain floor beside the anchor | only a zero-reconstruction diagnostic is currently saved | `MISMATCH` |
| preserve full attempt and stop after one | immutable attempts preserve major artifacts; no clean-reader one-attempt guard exists | `MISMATCH` |

## 4. Historical-result quarantine

Every existing ISET run prepared by this route is ineligible for
`CR-ISET-FCSAE-01` because it uses at least one materially different field:

- the semantic allocation CSV instead of the exact official `.tab`;
- seed 11 instead of `20260824`;
- clipped Attack 3 instead of the frozen duration-first completion;
- all-customer malicious test rows instead of attacks derived only from `B2`;
- no test ADASYN in the completed FC-SAE anchor; and/or
- batch 512 and `min_delta=1e-4` instead of batch 32 and `1e-6`.

Those attempts remain valid records of their named exploratory procedures.
They are not deleted, renamed as failures, or imported into the clean-reader
finding. Their outcomes did not select any Phase-4 correction.

## 5. Minimal correction boundary

Phase 4 may only:

1. expose and checksum-gate the exact official source branch;
2. implement the already frozen Attack-3 and `B2` malicious-population choices;
3. add a guarded `CR-ISET-FCSAE-01` preparation/training contract;
4. implement the frozen minimum-epoch stopping behavior and output-domain
   floor;
5. make the new eligibility and artifact checks explicit; and
6. add deterministic tiny fixtures for those operations.

It may not tune a model, choose a new data branch, repair the paper further,
run a second seed, add a mechanism control, or use historical proximity to the
target as a fidelity criterion.

## 6. Exit gate

Phase 4 passed on 2026-08-24. The five-file route now fails closed on the data,
attack, population, seed, model, training, scoring, and one-attempt contract;
eligible scores are produced by reloading persisted best weights; the exact
Softmax projection floor and artifact hashes are saved; and
`analyze_results.py --audit-clean-reader-anchor` independently regenerates the
metric vector and checks the complete artifact.

Phase 5 then checked both the local workspace and Panther. The six consumption
archives are available with exact official byte sizes and MD5s, but
`SME and Residential allocations.tab` is absent in both places and no
`ISSDA_API_TOKEN` is configured. This is a genuine named-data gate, not a model
result or evidence against the paper. The frozen semantic CSV remains
ineligible unless the user approves a new visible `I` branch.
