# Context Discipline

Maintain `docs/CONTEXT.md` inline so important state survives compaction. After
`/compact`, run `/charter-recover` to load CONTEXT, STATUS, and the active plan.

## Write when

- A non-obvious environment or data-access fact appears.
- A command or code pattern solves a non-trivial problem.
- A path fails and should not be retried.
- The user emphasizes or corrects a scientific priority.
- A mid-stream choice is not yet large enough for a decision record.

## Do not write

- Current project state: use STATUS.
- Architecture: use ARCHITECTURE.
- Significant choices: use `docs/decisions/`.
- Durable causal corrections: use `docs/EVIDENCE-AND-LEARNINGS.md`, leaving only
  a terse active pointer in CONTEXT.
- Trivial logs or ephemeral outputs.

Keep entries to one or two lines, update the date, and prune around 200 lines.
Never read raw Claude transcript JSONL for recovery; use `/charter-recover` or
`/charter-replay`.

