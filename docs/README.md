# Documentation guide

Start with the public [README](../README.md). It states the current result in
ordinary language.

## Current research

| Question | Read |
|---|---|
| What happened? | [Paper 1 clean-reader finding](../studies/atk-2022-deep-autoencoder/CLEAN_READER_FINDING.md) |
| Where are we now? | [STATUS](STATUS.md) |
| What happens next? | [Paper 3 all-model reproduction plan](plans/2026-09-20-robust-all-model-reproduction.md) |
| What did the Paper 3 source audit find? | [Source-audit finding](../studies/takiddin-2021-robust-poisoning/SOURCE_AUDIT_FINDING.md) |
| What explanations remain for Paper 3? | [Paper 3 explanation register](../studies/takiddin-2021-robust-poisoning/EXPLANATION_REGISTER.md) |
| What must survive a handoff? | [CONTEXT](CONTEXT.md) |
| Why did a conclusion change? | [Evidence and learnings](EVIDENCE-AND-LEARNINGS.md) |

## Research method

- [Runbook](../RUNBOOK.md): how to audit one paper from reading to report.
- [Vision](VISION.md): the questions and scientific posture.
- [Architecture](ARCHITECTURE.md): where claims, code, runs, and reports live.
- [Evidence frame](decisions/2026-08-20-three-part-evidence-frame.md): the
  detailed definitions and conclusion ladder.
- [Getting started](GETTING_STARTED.md): environment and data access.

## History

Files under `docs/plans/` and `docs/decisions/` are provenance. A dated file
may explain why a choice was made, but it is not automatically an instruction
to repeat that work. Follow only the active plan named in STATUS.

The July plans describe earlier implementation and execution paths. They remain
available because deleting them would erase changes of interpretation and
failed approaches. Paper 1's clean-reader and paper-time plans now describe a
completed audit stage; the Paper 3 all-model reproduction plan is current.

Agent-specific files contain no unique scientific facts. [AGENTS.md](../AGENTS.md)
is the short working guide; there are no charter recovery commands.
