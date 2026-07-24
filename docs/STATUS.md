# ATK Evidence — Project Status

**Last updated:** 2026-07-24
**Current branch:** `codex/paper1-exploratory-reproduction`

## Component status

| Component | Status | Location | Notes |
|---|---|---|---|
| Vision and Charter scaffold | Done | `docs/`, `AGENTS.md` | Vision approved 2026-07-20 |
| Canonical paper runbook | Done | `RUNBOOK.md` | Every paper now follows PDF freeze → exact data → five real files → sanity → first full anchor → tables/seeds → ambiguity controls → confirmation → report |
| Public repository and onboarding | Done | <https://github.com/fjoad/atk-evidence> | Public `main`, bootstrap/acquisition/verification scripts, CI |
| Multi-study registry | Done | `studies/registry.toml` | Stable domain-neutral study namespace |
| Paper 1 source/code fidelity audit | Historical forensic layer preserved | `studies/atk-2022-deep-autoencoder/PAPER_TO_CODE_TRACEABILITY.md` | Useful coverage evidence, but no longer the implementation critical path |
| Paper 1 branch coverage | Preserved; execution deferred until after compact anchor | `studies/atk-2022-deep-autoencoder/BRANCH_LATTICE_SUMMARY.md` | The 921-configuration inventory is a later coverage checklist, not a prerequisite for the first eligible result |
| Paper 1 data acquisition and preparation | Full compact route complete through the exact pre-ADASYN population | `data/`, `studies/atk-2022-deep-autoencoder/DATA_SOURCES.md` | 2,251,290 benign profiles, 13,507,740 attacked profiles, the customer split, and the 14,258,510-row `B2+M` test population are materialized; imbalanced-learn's exact default test-set ADASYN enters a brute-force 10.7-trillion-distance query and remains an executability branch, not a silently repaired cache |
| Paper 1 exploratory reproduction | Reset to compact ISET-first route | `docs/plans/2026-07-24-paper-1-minimal-reimplementation.md` | Existing artifacts remain retained; fresh source freeze and five-file route precede replacement runs |
| Cluster execution infrastructure | Connected; required for every experiment | local configuration (not published) | Execution runs on cluster compute nodes; no local experimental fallback is permitted |
| Paper 1 confirmatory experiments | Not started | `studies/atk-2022-deep-autoencoder/` | Exploratory results do not replace a later frozen confirmatory contract |
| Paper 1 LaTeX report | Scientific scaffold complete; findings/verdict pending | `reports/atk-2022-deep-autoencoder/main.tex` | Local Tectonic installation currently crashes before TeX compilation; no PDF is published |
| Public project site | Scaffold complete; not yet deployed | `site/`, `.github/workflows/pages.yml` | Multi-paper landing page and self-contained Paper 1 method map; deployment begins after merge to `main` and Pages is enabled |
| Five-file Paper 1 reference track | Tiny route complete; full original-population FC-SAE next | `studies/atk-2022-deep-autoencoder/reproduction/` | Five direct files remain the scientific route; the runner now separates the exact ADASYN test view from the pre-ADASYN `B2+M` interpretation so neither can collide or be silently substituted |
| Paper 1 controlled solution | Deferred/gated | `docs/plans/2026-07-23-paper-1-controlled-solution.md` | Design an honest detector that can exceed the paper only after the reproduction verdict is frozen |
| Cross-paper synthesis | Not started | `reports/synthesis/` | Begins after independent paper-level verdicts |

## Branch state

| Branch | Purpose | Status |
|---|---|---|
| `main` | Project bootstrap and canonical state | Published baseline |
| `codex/paper1-exploratory-reproduction` | Paper 1 implementation and exploratory Tables I-V evidence | Current |

## In-flight branches

`codex/paper1-exploratory-reproduction`

## Recent decisions

| Date | Decision | Why |
|---|---|---|
| 2026-07-20 | Paper-literal reproduction is the primary track | The research question concerns whether the published results follow from the method as described |
| 2026-07-20 | Ambiguities become documented reasonable branches | Missing details must not be silently filled in or optimized post hoc |
| 2026-07-20 | Controlled/corrected experiments are secondary and separately labeled | Method improvement cannot answer the primary reproducibility question |
| 2026-07-20 | Raw data and paper PDFs remain outside Git | Preserve access, licensing, and repository-size boundaries while recording checksums |
| 2026-07-21 | Publish as domain-neutral `atk-evidence` with per-study isolation | Allow later papers and domains without importing Study 1 assumptions |
| 2026-07-21 | Every non-redistributed dataset needs acquisition instructions and verification code | Independent researchers must be able to recreate the complete input gate |

## Existing artifacts and evidentiary status

- SGCC raw data: verified and checksummed locally.
- CER/ISET: all six ScienceDB consumption archives are local under
  `data/raw/cer-sciencedb/`, match the official filenames, sizes, and MD5s, and
  pass ZIP integrity. The converted allocation CSV has 6,445 unique mappings
  and matches a second public allocation workbook across all five semantic
  columns. It is admitted only as the named exploratory semantic-equivalence
  branch; it is not represented as the official checksum-gated `.tab` binary.
- Metric arithmetic audit: retained as a static paper audit, not an experimental
  reproduction verdict.
- SGCC 48-day attack pilot: exploratory proxy only. It does not use the CER
  half-hour profiles and cannot support or refute the primary hypothesis.
- Interrupted ten-seed proxy run: no confirmatory status; do not resume as part
  of the paper-literal track.
- the cluster is the authorized execution target for the remaining exact-SGCC
  Table II cells. Code must run from the exact pushed commit, the raw input must
  pass the frozen checksum, and model/seed attempts must remain immutable.
- Four-V100 sanity probes show materially different recurrent costs: LSTM-SAE
  about 14.1 minutes/epoch, LSTM-VAE about 1.25 minutes/epoch, and supervised
  LSTM about 20.3 minutes/epoch. The literal LSTM-AEA attention branch requests
  about 101.96 GiB per rank at local batch 128 and fails the primary resource
  gate. These are resource/timing observations, not Table II metrics.
- The initial supervised-LSTM CUDA assertion under a Torch-native optimizer was
  invalidated as model evidence by a clean compiled-Keras loss/optimizer rerun
  with finite loss, gradients, parameters, and optimizer state on all ranks.
- Twelve immutable classical Table II attempts (four models by three seeds)
  passed manifest and fingerprint verification. Every registered classical
  branch is currently `NOT_CLOSE_MATCH`.
- Table II currently has 20/33 successful cells: 12 classical, three FC-SAE,
  three supervised feed-forward, one LSTM-VAE, and one LSTM-SAE. FC-VAE has
  three retained failures and ten neural cells remain unrun. This is partial
  exploratory evidence, not a paper-level verdict.
- LSTM-SAE seed 11 is not remotely close: DR 6.78%, FA 2.22%, and AUC 51.89%
  versus the reported 86%, 12%, and 85%. Even an oracle test-label threshold
  reaches only 55.52% balanced accuracy on the paper-primary test rows.
  Its reconstruction score correlates 0.999999999999996 with the squared
  standardized input (zero reconstruction) and essentially matches FC-SAE's
  score. See `results/table_2_neural_score_sanity.json`.
- The exact ISET preparation completed in about 2 h 40 m: 4,225 residential
  meters, 2,251,290 benign profiles, 4,504,626 six-attack profiles, and all
  paper-positioned ADASYN partitions. Its Table IV subsets contain 30,603,120,
  45,904,656, and 61,206,240 scalar readings, closely matching the paper's
  30M/45M/60M labels. Cache SHA-256:
  `ab88f180feafb7351ef4530cba2e48a3cbc180af268f8b68016aefc50b98a987`.
- The exact-ISET execution adapter is available through the existing
  `run_experiment.py` interface. A real-cache preflight reverified the 3.2-GiB
  SHA-256 and loaded all partitions in 13.65 seconds. A one-epoch FC-SAE
  fixture smoke completed model construction, fit, Table III scoring, and
  Table V derivation. Table V reuses fixed row identities and the exact same
  Table III score vector and threshold; it does not retrain six models.
- First exact-ISET seed-11 results are artifact-verified executions of
  implementation v1, but are **not currently eligible as paper-reproduction
  evidence**. The fresh audit found that FC-SAE instantiates seven hidden
  transformations rather than the printed four-plus-four layout, FC-VAE has
  both an architecture mismatch and no reconstruction-probability score, and
  the ISET supervised cache does not use the paper's all-customer malicious
  population. The retained v1 values were: FC-SAE full
  DR/FA/ACC/AUC 22.50/37.13/42.69/42.59% versus reported
  81/15/83/81%; its half-data ACC is 42.68% versus 70%. FC-VAE full gives
  40.43/53.86/43.28/40.82% versus 88/11/88.5/85%. FC-SAE full improves
  balanced ACC over half by only 0.007 percentage points. See the eligibility
  map in `PAPER_TO_CODE_TRACEABILITY.md` before using any of these numbers.
- The source-first paper flow is now a self-contained readable document at
  `site/papers/atk-2022-deep-autoencoder/index.html`, linked from
  `studies/atk-2022-deep-autoencoder/PAPER_WORKFLOW.md`. It follows the SGCC
  and ISET lanes through Tables II--V without exposing the internal branch
  matrix as the primary explanation.
- Gate-D local sanity is recorded in
  `results/gate_d_bounded_sanity_20260724.json`: 137 study and 10 project tests
  pass; source-neural one-step and classical/data/runner fixture cells pass;
  the real SGCC printed anchor preflights in 9.55 seconds. The ISET printed
  anchor fails closed because its content-addressed source-v2 cache does not
  yet exist. This prevents accidental reuse of the historical
  implementation-v1 cache.

## What to work on next

Current critical path:

1. Build the new full P0 cache directly from the verified CER archives; do not
   relabel or import the historical implementation-v1 cache. The route is
   complete through `B2+M`; preserve the exact-default ADASYN feasibility
   failure and complete its scalable sensitivity separately.
2. Run the full ISET FC-SAE seed-11 model on the exact pre-ADASYN `B2+M`
   population on the cluster and immediately report
   reproduced metrics plus load/preparation/training/scoring/total time.
3. Complete the separately labeled ADASYN sensitivity, verify eligibility,
   then run seeds 22 and 33 and expand to the remaining
   Table III models.
4. Complete Tables IV--V, then SGCC Table II interpretations.
5. Use the existing forensic branch inventory to check material interpretation
   coverage after anchors exist.
6. Freeze and execute confirmatory runs.
7. Complete the report and controlled solution only after the verdict.

No new publication, generalized orchestration, or branch-lattice work belongs
before step 2.

**Execution correction (2026-07-24):** a local FC-SAE attempt was interrupted
after ten epochs/3,204.82 seconds wall when the user reconfirmed that all
experiments must run on the cluster. It produced no model, score, or result record
and is ineligible. The local default-ADASYN preparation was also interrupted
after 4,724.52 seconds wall inside its first brute-force neighbor query. No
experimental process remains active locally.
