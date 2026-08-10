# Decision: non-executable source evidence ladder

**Date:** 2026-08-11

**Status:** Accepted

**Scope:** Every audited paper and every public report

## Context

A paper may print a method step that is internally contradictory, undefined, or
literally impossible to execute. Quietly repairing it erases a source finding.
Stopping after identifying the defect leaves the underlying numerical claim
untested. Selecting only the repaired reading that later scores closest to the
published value creates hindsight bias.

The working hypothesis may be that reasonable completions will also fail to
reproduce the published result. That hypothesis is testable, but it cannot
choose interpretations, seeds, stopping rules, or which results are shown.

## Decision

Every material non-executable source node produces this evidence ladder:

1. **Literal source result.** Preserve the exact wording, source locator, and
   proof or explanation of why it cannot execute. Record this as a result, not
   as an omitted run.
2. **Predeclared repairs.** Before seeing outcomes, name the smallest reasonable
   executable completion and every other materially distinct defensible
   interpretation. Give each a stable `I` identifier.
3. **Complete execution.** Run every declared repair over every declared seed,
   preserving failures and timings as well as successful outputs.
4. **Side-by-side reporting.** The study site and LaTeX/PDF report must show the
   published target, literal failure, configuration of each repair, all run
   outcomes, and the resulting metric distributions together.
5. **Bounded conclusion.** If a repair matches, report the match plainly. If
   none matches, conclude only that the result was not reproduced under the
   finite predeclared family. Never infer an unobserved author implementation or
   claim that no conceivable implementation could match.

## Boundaries

- A repair is an interpretation track (`I`), not a paper-literal track (`P`).
- A scientifically preferred method is a corrected control (`C`) and answers a
  different question.
- Static arithmetic contradictions remain source findings; experimental output
  does not repair a published table identity.
- No branch may be added, removed, or promoted because its preliminary result
  is favorable.

This ladder is part of the evidence contract and therefore must survive chat
compaction and agent changes.
