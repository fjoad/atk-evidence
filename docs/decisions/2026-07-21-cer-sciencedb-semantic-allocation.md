# CER ScienceDB acquisition and semantic allocation branch

**Date:** 2026-07-21

## Context

The official ISSDA record restricts anonymous access. ScienceDB DOI
`10.57760/sciencedb.17619` publicly exposes the six CER consumption archives
and a converted allocation CSV. All six archives are byte-identical to the
official ISSDA objects by filename, size, and MD5. The allocation CSV is not
binary-identical to the official `.tab` file required by the initial gate.

## Decision

For the authorized **exploratory** Paper 1 run:

1. Admit the six ScienceDB archives as exact CER inputs because each passes the
   official MD5 and ZIP-integrity checks.
2. Admit `SME_and_Residential_allocations.csv` under the explicit branch name
   `sciencedb-csv-semantic-equivalence-v1`, not as an official `.tab` binary.
3. Preserve the distinction in every manifest and report. If the official
   `.tab` is later acquired, it supersedes this branch for provenance but must
   be checked for any residential-ID difference before results are combined.
4. Keep all raw inputs local and ignored by Git; publish only checksums,
   provenance, verification summaries, and code.

## Evidence supporting semantic equivalence

- The CSV parses to 6,445 unique meter IDs with no conflicting assignments:
  4,225 residential, 485 SME, and 1,735 other.
- Its first five semantic columns match a second public CER allocation workbook
  for all 6,445 rows after normalizing blank cells represented as `0` in the
  workbook and empty cells in the CSV.
- A complete scan of 157,992,996 readings found every residential and SME ID,
  no reading without an allocation, and only ten absent allocation IDs, all in
  the unused `other` category.
- The only undocumented time suffixes above 50 affect two `other` meters and no
  residential meter. The primary residential selection therefore removes them
  before profile validation.

Machine-readable evidence is in
`studies/atk-2022-deep-autoencoder/results/cer_sciencedb_acquisition.json`.

## Alternatives considered

- **Wait indefinitely for the official `.tab`:** rejected for the exploratory
  run because the exact residential mapping is independently cross-checked and
  file serialization is not a scientific variable in the paper.
- **Rename the CSV and pretend it is official:** rejected because it would
  erase a real provenance distinction and fail the official MD5.

## Consequences

- The implementation must expose this as a named allocation branch and record
  the CSV checksum and semantic digest.
- It must filter residential IDs before rejecting malformed non-residential
  time codes.
- Confirmatory reporting must state that archive identity is cryptographic,
  while allocation identity is semantic rather than binary.
