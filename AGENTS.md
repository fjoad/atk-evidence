# ATK Evidence — Working Guide

## What this repository does

This project reads research papers, implements the methods they describe, and
compares measured results with published ones. When a result differs, the work
asks what explains the difference before spending more compute.

The public explanation starts with the paper, code, experiment, and result.
Internal labels exist to keep evidence separate; they are not the voice of the
README or website.

## Start here

1. Read [RUNBOOK.md](RUNBOOK.md) for the research procedure.
2. Read [docs/STATUS.md](docs/STATUS.md) for the current result and next decision.
3. Read [docs/CONTEXT.md](docs/CONTEXT.md) for facts that must survive a handoff.
4. Read the active plan named in STATUS.
5. When interpreting a result, also read the relevant study finding and
   explanation register.

The paper is the authority for what it claims. The runbook is the authority for
the order of the audit. If the paper is incomplete or contradictory, preserve
that fact and make every executable interpretation visible.

## Scientific rules

- Read the complete paper before implementing a reported experiment.
- Connect each consequential instruction to a page, equation, figure, table, or
  source note.
- Use small exploratory scripts to discover useful questions. Do not present
  their results as a reproduction.
- Implement the written method before improving it. Corrections and controls
  remain separate.
- Record every necessary assumption before seeing whether it helps.
- Preserve failed, interrupted, and unfavorable attempts.
- Establish simple and zero-parameter comparisons before crediting a complex
  architecture.
- Ask whether an added component supplies the capability claimed for it; a
  headline metric alone does not answer that question.
- Use cheap checks before long runs. Another expensive seed or paper row must
  answer a named uncertainty.
- Freeze the data, code, metric, seeds, uncertainty method, budget, and stopping
  rule before confirmatory depth.
- Treat related rows, repeated customer-days, attacks derived from one profile,
  and synthetic examples as dependent unless the analysis establishes
  otherwise.
- State exactly what was tested. A fixed-score cutoff limit is not a limit on
  scores from another model. An empirical plateau is not a mathematical proof.
- Report a match or evidence supporting the paper as plainly as a failure.
- Do not infer author intent, undocumented code, or fabrication from
  non-reproduction alone.
- Do not transfer a result from one paper to another.

The detailed separation between numerical reproduction, mechanism, and
attainability is preserved in
[the evidence-frame decision](docs/decisions/2026-08-20-three-part-evidence-frame.md).

## Current scientific boundary

The discussed source-assumption findings are published at `dc37bbe`. The
subsequent user-approved no-training Sigmoid investigation is also complete:
[saved local finding](studies/atk-2022-deep-autoencoder/SIGMOID_SANITY_FINDING.md).
Stop for discussion before publishing its outcomes or running another check.
No training or broad search is authorized. The
[bounded follow-up plan](docs/plans/2026-08-31-source-findings-and-sigmoid-sanity.md)
records publication, the frozen setup, every outcome, and this stop.

Paper 1 is the only active experiment. Its first clean-reader FC-SAE run is
complete and audited. The result did not reproduce Table III under the declared
implementation and one seed. The completed follow-up excludes the target for
any weights under the fixed prepared inputs, Softmax output, and MSE score.
Useful score differences exist; the claimed architectural mechanism and other
source interpretations remain open. See
[the follow-up finding](studies/atk-2022-deep-autoencoder/POST_ANCHOR_FINDING.md)
and [current status](docs/STATUS.md).

The Checkpoint-2 diagnostic rounds and Sigmoid follow-up are complete. Their
CPU allocations ended successfully; no model was trained. Any further work
needs a named remaining question, recorded setup, and approval. Do not start a full
training run, seed sweep, model family, or broader search. Experimental scoring
remains on cluster compute nodes; local runs are software fixtures only.
Preserve the original run and both diagnostic contracts unchanged.

## Repository and evidence

- Keep raw datasets and source PDFs local and unmodified.
- Never commit credentials, restricted archives, or machine-specific secrets.
- Each paper belongs in `studies/<study-id>/` and is registered in
  `studies/registry.toml`.
- The five direct files under a study's `reproduction/` directory are the
  active paper-facing implementation. Older forensic machinery is historical
  evidence, not the default route.
- `docs/STATUS.md` is current state; `docs/CONTEXT.md` is compact handoff
  memory; `docs/EVIDENCE-AND-LEARNINGS.md` preserves changed conclusions.
- Historical plans and decisions remain for provenance. They do not override
  the current plan.

## Working practice

For a multi-file change, save a short plan, complete the change, verify it,
update current status, and commit it. User checkpoints still apply to
scientific experiments. Documentation work may report completed evidence but
must not strengthen the conclusion beyond that evidence.

Use:

```bash
bash scripts/bootstrap.sh
bash scripts/test.sh
.venv/bin/python scripts/verify_data.py --strict
```

The verification command may stop when restricted data are unavailable. Never
substitute a proxy and describe it as the named dataset.
