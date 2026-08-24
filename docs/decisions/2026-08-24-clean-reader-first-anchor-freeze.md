# Clean-reader first-anchor freeze

**Date:** 2026-08-24

**Status:** Accepted as the candidate evidence contract for Checkpoint 1;
formal execution remains unapproved

## Decision

Use
[`CLEAN_READER_SPECIFICATION.md`](../../studies/atk-2022-deep-autoencoder/CLEAN_READER_SPECIFICATION.md)
as the sole candidate source contract for the new clean-reader route.

The first anchor is `CR-ISET-FCSAE-01`: one seed of the Table-III ISET FC-SAE
row. Preserve the literal failures of Attack 3 and threshold derivation, then
use only the predeclared minimal `I` completions in the specification. Require
the official ISSDA Version-1 consumption archives and allocation `.tab` at the
data gate. Stop after one fully preserved attempt or literal operational
failure.

## Why this anchor

- FC-SAE is the simplest proposed model and avoids repairing VAE, recurrent,
  and attention semantics before the basic measuring path is trusted.
- ISET supplies source-defined 48-value daily profiles and six explicit attack
  functions, while the SGCC representation needed by the 48-input models is
  not stated.
- A single row is enough to test the complete paper-to-data-to-score chain; it
  is not enough for a paper-level reproduction or mechanism verdict.
- Requiring the exact allocation artifact prevents a source-independent data
  substitution from silently becoming the new primary route.

## Consequences

- Phase 4 may inspect the existing five-file implementation only after the
  user approves Checkpoint 1.
- Matching historical attempts may be admitted only if every consequential
  field agrees with the approved freeze; otherwise they remain exploratory or
  quarantined.
- No named-data run, extra model, branch, seed, threshold tuning, or mechanism
  control is authorized before its later checkpoint.
- If the exact allocation `.tab` is unavailable, the first anchor records a
  data-gate failure unless the user explicitly approves a separate semantic-
  serialization `I` branch.

## Rejected alternatives for the first anchor

- starting with a VAE, recurrent model, or attention model;
- beginning with SGCC despite the unstated mapping to 48 inputs;
- admitting the semantically checked allocation CSV without review;
- selecting an Attack-3 repair, scaler, threshold, or training rule after
  observing which one approaches the reported row; and
- expanding immediately to the paper’s complete model/table matrix.
