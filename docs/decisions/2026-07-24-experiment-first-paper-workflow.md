# Decision: experiment-first paper workflow

**Date:** 2026-07-24
**Status:** Accepted

## Context

Paper 1 accumulated a 21,414-line internal Python tree including tests before a
complete source-faithful result matrix existed. Several expensive results were
later invalidated or quarantined because the PDF had not been frozen into a
claim-to-code specification before implementation. The user identified that
implementation and documentation time had overtaken eligible experimental
runtime.

## Decision

[`../../RUNBOOK.md`](../../RUNBOOK.md) is the canonical execution order for
every paper:

1. complete PDF-derived method freeze;
2. exact data;
3. genuine five-file implementation;
4. tiny sanity checks;
5. first full eligible anchor;
6. remaining tables and seeds;
7. material ambiguity branches and corrected controls;
8. confirmatory contract;
9. report and publication.

The existing Paper 1 branch engine remains preserved as a forensic coverage
tool, but it is removed from the critical path to the compact reproduction.

## Consequences

- No new infrastructure or publication work precedes the first eligible full
  anchor.
- Material interpretations are still required, but a straight-through anchor
  runs first and alternatives are expanded without arbitrary Cartesian
  combinations.
- The five public files contain the real scientific implementation rather than
  wrappers over the forensic harness.
- Existing artifacts retain their original eligibility; this decision does not
  upgrade, delete, or rewrite them.
