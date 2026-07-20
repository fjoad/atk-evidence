# Decision: Paper-Literal Reproduction and Repository Evidence Policy

**Date:** 2026-07-20

## Decision

The primary experiment for every target paper is a paper-literal reproduction
with predeclared ambiguity branches; raw datasets and publication PDFs remain
outside Git while their provenance and checksums are committed.

## What was considered

1. **Start with corrected methodology:** rejected because it answers whether a
   better experiment works, not whether the published result is reproducible.
2. **Choose one best interpretation of omissions:** rejected because it permits
   undocumented analyst discretion and post-hoc favorable choices.
3. **Chosen — literal branches plus versioned provenance:** implement explicit
   steps first, enumerate reasonable ambiguities, and keep protected inputs
   local with identifying manifests.

## Why this choice

The user's primary hypothesis concerns the causal connection between what the
paper says was done and the numbers it reports. The evidence must therefore
make every added assumption visible and must not confuse method improvement
with reproduction. Dataset licensing and size also make raw-input exclusion
from Git the appropriate boundary.

## Impact

The work requires more up-front specification and may produce several branches
for one ambiguous statement. In exchange, every result has a clear scope, raw
inputs can be independently verified, and later reports can distinguish failure
to reproduce from failure of an improved alternative.

