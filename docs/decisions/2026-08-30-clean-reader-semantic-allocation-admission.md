# Clean-reader semantic allocation admission

**Date:** 2026-08-30

**Status:** Approved by the user for the single `CR-ISET-FCSAE-01` anchor

## Context

The original clean-reader anchor required the byte-identical restricted
Dataverse serialization `SME and Residential allocations.tab`. Official
metadata later established that this `.tab` is an archival ingest of an
original XLSX, not a scientifically distinct allocation dataset.

Two public representations independently establish the allocation mapping:

- the ScienceDB CSV has a frozen MD5, SHA-256, byte size, and 6,445 unique
  meter assignments; and
- a commit-frozen public GitHub workbook agrees with it across all five
  semantic allocation columns after the predeclared normalization of
  inapplicable workbook zeroes and CSV blanks.

The six consumption archives remain byte-identical to the official ISSDA
objects. Every residential reading meter is present in the verified mapping.
The remaining difference is serialization provenance only.

On 2026-08-30 the user explicitly approved proceeding with the public data
after this distinction and the branch name were presented.

## Decision

Admit `sciencedb-csv-semantic-equivalence-v1` as a visible reasonable
interpretation (`I`) of the allocation serialization for the one frozen
`CR-ISET-FCSAE-01` numerical (`N`) anchor.

This decision changes only the allocation source branch. It does not change:

- the six exact consumption archive identities;
- the 4,225-meter residential population selected by `Code = 1`;
- profile parsing, attacks, scaling, split, test ADASYN, model, training,
  scoring, threshold, metrics, seed, compute budget, or stopping rule; or
- the requirement to stop after one preserved attempt or operational failure.

The run and audit must fail closed unless they record the exact branch name,
the frozen CSV hashes and cardinalities, and verified status for every source
file. Reports must call this semantic allocation identity, never a
byte-identical copy of the official `.tab`.

## Consequences

- The formerly blocked official-serialization branch remains preserved as a
  data-gate failure.
- Phase 5 may proceed under the newly approved `I` branch.
- Any future byte-identical `.tab` result remains a distinct provenance branch
  and cannot overwrite this attempt.
- No second seed, model, threshold, correction, mechanism test, or
  attainability search is authorized before the anchor is fully audited and
  Checkpoint 2 is reviewed.
