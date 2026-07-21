# ATK Evidence — Project Status

**Last updated:** 2026-07-21
**Current branch:** `main`

## Component status

| Component | Status | Location | Notes |
|---|---|---|---|
| Vision and Charter scaffold | Done | `docs/`, `AGENTS.md` | Vision approved 2026-07-20 |
| Public repository and onboarding | Done | <https://github.com/fjoad/atk-evidence> | Public `main`, bootstrap/acquisition/verification scripts, CI |
| Multi-study registry | Done | `studies/registry.toml` | Stable domain-neutral study namespace |
| Paper 1 source audit | In progress | `studies/atk-2022-deep-autoencoder/` | Initial extraction exists; must be reorganized around the paper-literal contract |
| Paper 1 data acquisition | In progress | `data/`, `studies/atk-2022-deep-autoencoder/DATA_SOURCES.md` | SGCC verified; CER consumption archives require authorized ISSDA access |
| Paper 1 reproduction contract | Not started | `docs/plans/2026-07-20-paper-1-reproduction-contract.md` | Draft plan awaiting review |
| Paper 1 confirmatory experiments | Not started | `studies/atk-2022-deep-autoencoder/` | Do not start before contract and exact-data gate |
| Paper 1 LaTeX report | Not started | `reports/atk-2022-deep-autoencoder/` | Reproduction/rebuttal report |
| Cross-paper synthesis | Not started | `reports/synthesis/` | Begins after independent paper-level verdicts |

## Branch state

| Branch | Purpose | Status |
|---|---|---|
| `main` | Project bootstrap and canonical state | Current |

## In-flight branches

_None active._

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
- CER/ISET: official manifest and documentation acquired; restricted consumption
  archives are not yet available locally.
- Metric arithmetic audit: retained as a static paper audit, not an experimental
  reproduction verdict.
- SGCC 48-day attack pilot: exploratory proxy only. It does not use the CER
  half-hour profiles and cannot support or refute the primary hypothesis.
- Interrupted ten-seed proxy run: no confirmatory status; do not resume as part
  of the paper-literal track.

## What to work on next

1. ~~Approve the project vision and attach Charter.~~
2. ~~Initialize the repository and preserve the current baseline.~~
3. ~~Publish ATK Evidence with a multi-study layout and reproducible onboarding.~~
4. **Review and freeze the Paper 1 reproduction contract: reported targets,
   literal algorithm, ambiguity branches, search envelope, tolerances, seeds,
   and stopping rules.** **(next)**
5. Obtain the authorized CER/ISET consumption archives and verify official MD5s.
6. Implement and validate the paper-literal pipeline only.
7. Execute preregistered pilot and confirmatory runs.
8. Produce the Paper 1 LaTeX report and verdict.
9. Register Paper 2 without importing Paper 1's verdict.
