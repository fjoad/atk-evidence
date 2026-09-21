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

The user-approved feed-forward pair is implemented and locally fixture-tested,
but no research-data neural fit or GPU job has been submitted. The QCRI VPN
disconnected during CPU environment setup job 398992; its final status is
unknown. On reconnect inspect that existing job/log before retrying setup.
FEED_FORWARD_PILOT.md fixes the six 500-neuron hidden layers and standard BCE
repair, 50 epochs/batch 100, original p00/p30 inputs, and one V100-16GB pair
inside 20 minutes with seven-minute fit guards. GPU preflight and fixtures
must pass before real inputs; no CPU fallback. Resume this already-approved
pair when access returns, but do not add seeds or alternative settings.
The implementation and assumptions are preserved for the pending code freeze.
All prior results remain unchanged. Nothing is published.

### Earlier read-only SVM checkpoint

The approved read-only SVM follow-up is complete (4665e07, job 398978),
with zero experimental fits. All 13,392 manually reconstructed/native scores
agree within 1.68e-12 and all labels match. The fixed 512-row sigmoid kernel
has large negative directions, including under the dual equality constraint
(centered minimum -19.97 versus tolerance 1.48e-8). Thus the usual concave-dual
guarantee is unavailable here; fitted-point suboptimality, performance cause,
and failure of other settings are NOT established. Original artifacts remain
unchanged and all diagnostic audits pass. See the
[record](studies/takiddin-2021-robust-poisoning/results/svm_replay_20260921/README.md).
Stop this diagnostic. Next proposed coverage step is a separately specified
feed-forward pilot with the standard cross-entropy repair explicit; SVM
parameter sensitivity remains open. No new fit is authorized by this summary.
Website changes remain local drafts; no deployment occurred.

### Earlier SVM checkpoint

The user-approved first sigmoid SVM pair is complete (e698173, job 398709).
Its original-pilot AUC is 65.64/63.38 at p00/p30; at the corresponding printed
FA caps 10.2/25.7%, best saved-score DR is 19.37/33.48 versus 89.2/73.7.
Neither any cutoff nor favorable score reversal rescues these fitted models
on these rows. Both solvers and artifact audits passed; training accuracy is
weak too. Other parameters/full data are not excluded. Stop this pair. The
next proposed question is bounded read-only score replay and sigmoid-kernel
inspection on a fixed subset, not extra seeds/fits. No kernel spectrum has
been measured. See the [SVM record](studies/takiddin-2021-robust-poisoning/results/svm_pilot_20260921/README.md).
Preserve raw scores, original inputs, all fitted models, contracts, and frozen
Git revisions. Journal/site updates remain local drafts, not deployed.

### Earlier AdaBoost checkpoint

The user-approved first AdaBoost pair is complete (d47a6de, job 398348).
Its historical SAMME.R completion gives AUC 90.71/83.74 at p00/p30 on the
original pilot. Default DR drops to 46.43% under poisoning, but saved-score
DR reaches 80.45% within FA<=29.9%. Useful ranking survives; the complete
printed pattern is not reproduced by this small pilot. All artifact checks
passed. Stop this pair; next proposed model is a separately specified SVM
pair, not automatic AdaBoost seeds or alternative settings. See the
[record](studies/takiddin-2021-robust-poisoning/results/adaboost_pilot_20260920/README.md).
The direct files now support both baselines. Preserve historical scientific
revisions in Git and verify old source hashes against their recorded commits;
never rewrite old evidence to match the latest implementation. Local website
drafts are not published. Paper 1 is unchanged by this work.

### Earlier Paper 3 checkpoints

Paper 3's September 20 all-model investigation is active under the plan in
STATUS. Its preparation, first forest pair, and user-approved matched controls
are complete. The four-fit control (frozen e6e0359, job 398164) found that
training-only resampling lowers AUC on matched rows, while whole-source-day
grouping does not cause a further collapse. C retains AUC 93.18/82.97 at
0%/30% poisoning. All saved-artifact checks passed. This is one 20-customer
pilot, not full-paper reproduction or evidence of intent. Stop this diagnostic;
the proposed next distinct question is a separately specified AdaBoost pair.
See the [control record](studies/takiddin-2021-robust-poisoning/results/split_control_20260920/README.md).
No additional model or seed is authorized by this summary alone. The website
changes are local drafts, not deployed. Preserve the original preparation,
baseline, control contract, code, and outputs.

### Separate Paper 1 checkpoints

The source-assumption findings are published at `dc37bbe`; subsequent Sigmoid
checks are saved locally. The approved
[small paired fit](docs/plans/2026-08-31-small-sigmoid-fit.md) is complete.
Both models finished ten epochs. Sigmoid's best detection at FA<=15% was
9.74935% (25.39063% reversed), versus 81%. This excludes cutoff rescue for
those fitted models and sampled rows, not all Sigmoid configurations. Its
calibration loss still improved; no long-run plateau is established. Stop
for discussion before another experiment or publication. See
[the finding](studies/atk-2022-deep-autoencoder/SIGMOID_FIT_FINDING.md).

Paper 1's first clean-reader FC-SAE run is
complete and audited. The result did not reproduce Table III under the declared
implementation and one seed. The completed follow-up excludes the target for
any weights under the fixed prepared inputs, Softmax output, and MSE score.
Useful score differences exist; the claimed architectural mechanism and other
source interpretations remain open. See
[the follow-up finding](studies/atk-2022-deep-autoencoder/POST_ANCHOR_FINDING.md)
and [current status](docs/STATUS.md).

All diagnostic allocations, including the small paired fit, are complete.
Any further work needs a named
remaining question, recorded setup, and approval. Do not start a full
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
