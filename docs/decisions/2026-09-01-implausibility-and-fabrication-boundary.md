# Separate empirical implausibility from fabrication

**Date:** 2026-09-01

## Decision

This project may conclude that a reported result is **highly implausible within
a declared empirical envelope** when all of the following have been measured:

1. a faithful primary implementation and the finite set of materially
   reasonable source-supported completions miss the complete reported pattern;
2. simple corrections, score directions, thresholds, seeds, and ordinary
   optimization budgets do not close the gap;
3. learning, capacity, data-size, and cumulative-compute curves are stable or
   saturating far below the target;
4. simple and positive controls show whether the task and measuring path are
   learnable;
5. mechanism witnesses and matched ablations do not display the capability
   claimed to explain the paper's advantage; and
6. any stronger structural bound is stated only under its exact assumptions.

That conclusion concerns the relationship between the reported result and the
described method. It is not automatically a conclusion about author intent.

“Fabricated,” “invented,” or “made up” requires additional forensic evidence
that distinguishes deliberate reporting from an undocumented implementation,
data leakage, a metric or table-generation error, a transcription error, or
another consequential procedure omitted from the paper. Examples include raw
artifacts incompatible with the table, impossible internal metric identities,
contradictory code or logs, or direct provenance evidence. Repeated
non-reproduction alone may make fabrication plausible, but does not uniquely
identify it.

## Why

A reasonable researcher should not search forever. Finite predeclared breadth,
one or more faithful full anchors, discriminating mechanism tests, and a stable
attainability envelope can justify a strong bounded conclusion. The boundary
prevents two opposite errors: treating every failed run as proof of misconduct,
or refusing to say “highly implausible” after ordinary explanations have been
systematically tested and the target remains far outside the evidence.
