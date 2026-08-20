# [Step Name] — Evidence Plan

**Goal:** [What will be true when complete.]

**Status:** [Draft / Approved / In progress / Complete]

**Evidence question:** [Discovery / Numerical (`N`) / Mechanism (`M`) /
Attainability (`A`)]

**Implementation semantics:** [Exploratory (`X`) / Paper-literal (`P`) /
Reasonable interpretation (`I`) / Controlled analysis (`C`)]

## Context

[Why this step happens now and what it unblocks.]

## Claim and decision

**Paper claim or causal link:** [Source location and exact proposition; for
example, `B` outperforms `A` because component `Z` exploits structure `S`.]

**Question:** [One precise question this plan will answer.]

**Competing explanations:**

- [Explanation 1]
- [Explanation 2]

**Discriminating predictions:**

| Outcome | Supports or weakens | Why |
|---|---|---|
| [Observable result] | [Explanation] | [Reason] |

**Report consequence:** [Which of the numerical, mechanism, or attainability
findings can change, and how.]

## Promotion from diagnostic breadth

**Cheap checks already completed:** [Static audit, toy witness, trivial rule,
minimal ablation, output inspection, or other sandbox/formal probe.]

**Why deeper execution is still needed:** [Material uncertainty that the
proposed work can resolve.]

**Cheaper remaining discriminator:** [None, or name it and run it first.]

## File structure

**Create:**

- `[path]` — [purpose]

**Modify:**

- `[path]` — [change]

## Steps

### Task 1: [Name]

- [ ] [Action]
- [ ] Verify: [check]

## CHECKPOINT: user review

[What must be reviewed before continuing.]

### Task 2: [Name]

- [ ] [Action]

## Verification

- [ ] Evidence-question and implementation-semantics labels are recorded.
- [ ] Exact data, independent unit, seeds/partitions, statistics, tolerance,
      budget, and stopping rule are frozen where applicable.
- [ ] All attempts and failures are preserved; no favorable-run selection.
- [ ] Conclusion is bounded to the declared finite space.
- [ ] [Specific check]
- [ ] Full test suite passes
- [ ] STATUS and evidence records updated
- [ ] Changes committed

## Finish condition

[Name the exact artifact or finding that makes this step complete. More code,
more models, or more branches are not a finish condition by themselves.]
