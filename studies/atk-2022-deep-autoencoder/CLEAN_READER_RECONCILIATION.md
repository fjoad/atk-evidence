# Clean-reader rebase — Phase 0 reconciliation

**Date:** 2026-08-23

**Status:** Phase 0 classification complete; no artifact admitted as the new
clean-reader anchor

**Governing plan:**
[`../../docs/plans/2026-08-23-clean-reader-reproduction-rebase.md`](../../docs/plans/2026-08-23-clean-reader-reproduction-rebase.md)

## Purpose

Preserve and classify the work that existed before the clean-reader rebase.
This is an inventory and evidentiary boundary, not a new interpretation of the
paper and not a retroactive preregistration of prior work.

## No-new-run gate

No preparation, training, scoring, ambiguity branch, or repeated-seed run was
launched during this phase. Existing external outputs may be harvested later
without authorizing additional execution. Their current remote state remains
`OPEN` because the last recorded cluster access required an interactive VPN
login.

## Inventory snapshot

- Target PDF: 12 pages, 2,994,509 bytes, SHA-256
  `f3098e0c27ee19b27bea026aedc3d10e5dbb0c46f5cd01ed5bd5c05b7dcf850f`.
- Local study results: 36 files including `.gitkeep`, approximately 1.1 MiB.
- Minimal reproduction directory: five scientific Python files plus direct
  documentation and one resource wrapper, approximately 376 KiB total.
- Historical forensic `src/` directory: approximately 2.2 MiB.
- Local data tree: approximately 11 GiB, containing raw named sources,
  provenance documents, and prior derived caches.
- Local paper tree: approximately 8.3 MiB. Only the target PDF identified above
  is source authority for the orientation pass.

Counts and sizes describe the 2026-08-23 local workspace. They are not
scientific evidence.

## Provisional artifact classification

These classifications govern what may be consulted and what may later be
admitted. They do not change the labels carried by the artifacts themselves.

| Artifact family | Examples | Provisional question / semantics | Rebase status | Clean-reader rule |
|---|---|---|---|---|
| Target publication | Local fingerprinted PDF | Source authority | `VERIFIED` identity; content requires new source-only pass | Sole scientific authority during Phase 1 |
| Prior source reconstructions | `METHOD.md`, `PAPER_WORKFLOW.md`, reported CSVs, traceability and literal contracts | Mostly `N`; prior `P/I` readings | Preserved, prior-exposed, not authoritative for Phase 1 | Close during the paper-only pass; compare only after the new orientation record is saved |
| Raw named data and provenance | CER archives/allocation, SGCC archives, official CER documentation | Candidate formal `N`; semantics unresolved until source freeze | Preserved and immutable; identity/provenance candidates | Do not inspect experimentally or admit a derived sample unit before Phase 3 |
| Prior derived caches | ISET and SGCC `.npz` plus metadata | Candidate `P/I-N` | Preserved; eligibility unresolved | Admit only after Phase 4 binds hashes, sample identities, and preparation semantics to the new freeze |
| Five-file reproduction | `reproduction/download_data.py`, `prepare_data.py`, `models.py`, `run_experiment.py`, `analyze_results.py` | Candidate `P/I-N` instrument | Preserved; neither authority nor rejected | Keep closed during Phases 1–3; assess line by line in Phase 4 |
| Historical root wrappers and `src/` harness | Root commands, branch lattice, paper-literal runners, DDP and cluster utilities | Mixed historical `P/I/C/X`; multiple questions | Preserved forensic/operational history | Cannot become the primary clean-reader implementation; consult only after the source freeze for coverage or provenance |
| Static reported-number audits | `reported_metrics_audit.*` and associated evidence notes | Candidate source-level `N` | Preserved; must be independently regenerated from the new transcription | May confirm an arithmetic contradiction only after Phase 1 records the printed definitions and cells |
| Straight-through and model-family attempts | Compact anchor, Table-II/III/V breadth summaries, saved score audits | Mostly exploratory `X-N`; some may later be candidate `I-N` | Preserved without confirmation or paper-level verdict | May suggest questions after Phase 1; an existing attempt is admitted only if Phase 4 proves exact agreement with the approved freeze |
| Threshold and score-vector analyses | Exact threshold gaps, oracle-direction and score-sanity summaries | Exploratory `X-N`; possible contextual `M/A` | Preserved finite statements about recorded vectors | Cannot generalize to different scores, implementations, or an attainability envelope |
| Controlled and positive-control work | Linear-output control, supervised controls, trivial/zero-score comparisons | `C-M` or exploratory `X-M/A` | Preserved diagnostic observations | Cannot repair or replace the primary numerical reproduction; may motivate a promoted mechanism question later |
| Runtime and operational evidence | ADASYN estimate, thread sensitivity, Panther probes, preserved failures | Exploratory `X-A` or operational evidence | Preserved; remote harvest remains `OPEN` | May inform a later compute envelope but cannot establish what another party could have executed |
| Software fixtures and tests | Unit tests, tiny sanity outputs, compilation checks | No direct `N/M/A` finding | Preserved software evidence | Demonstrate instrument properties only; never substitute for source fidelity or a scientific result |
| Reports, site pages, status, and evidence ledger | LaTeX, public method maps, summaries, plans | Downstream narrative/coordination | Preserved; not scientific authority | Update only from admitted source and result artifacts, never use them to decide the new paper reading |

## Prior-exposure boundary

The researcher and agents have already seen earlier source reconstructions,
code, and outcomes. Phase 1 therefore cannot honestly be called cognitively
blind. It is instead **procedurally source-isolated**:

1. only the fingerprinted target PDF and its rendered pages may be consulted;
2. the prior reconstruction, implementation, results, reports, and evidence
   ledger remain closed until the new orientation record is saved;
3. every claim in that record must carry a PDF locator; and
4. agreement with a prior belief does not validate it until the later explicit
   comparison.

An independently blinded human reconstruction would be stronger and remains an
available future control. It is not silently claimed here.

## Phase 0 disposition

- **VERIFIED:** the target PDF identity and local artifact inventory.
- **OBSERVED:** extensive prior source, implementation, data, run, control,
  failure, and reporting artifacts exist.
- **OPEN:** whether any prior attempt exactly matches the clean-reader
  specification that will be frozen at Checkpoint 1.
- **OPEN:** the unharvested remote state last observed for existing Table-IV
  jobs.
- **NOT AUTHORIZED:** any new scientific execution before Checkpoint 1.

## Exit decision

Phase 0 is complete. Existing work is preserved and fenced off from the new
source interpretation. Proceed to Phase 1 using only the target PDF. No prior
artifact has been admitted, rejected, or relabeled as a result of this
inventory.
