# Project Flow

Canonical project facts live in shared files readable by Claude and Codex.
`AGENTS.md` is the bootstrap; `CLAUDE.md` imports it. These rules automate the
same contract for Claude and must not become a second source of project truth.

## Start every session

1. Read `docs/STATUS.md` for the current step and next action.
2. Read `docs/CONTEXT.md` for compact memory and don't-repeats.
3. Read `docs/EVIDENCE-AND-LEARNINGS.md` when interpreting evidence,
   corrected claims, or disputed findings.
4. Read `docs/ARCHITECTURE.md` and the active plan in `docs/plans/`.
5. If the current step lacks a plan, create one before implementation.

## Project document map

- `docs/VISION.md` — thesis, scope, falsification posture, and success criteria.
- `docs/STATUS.md` — current state and next work.
- `docs/CONTEXT.md` — prunable working memory across compaction.
- `docs/EVIDENCE-AND-LEARNINGS.md` — durable causal evidence.
- `docs/ARCHITECTURE.md` — artifact and evidence flow.
- `docs/decisions/` — deliberate choices and rationale.
- `docs/plans/` — approved execution contracts and checkpoints.

Follow the ordered work in STATUS. Do not skip the paper-literal reproduction
contract to resume exploratory experiments.

