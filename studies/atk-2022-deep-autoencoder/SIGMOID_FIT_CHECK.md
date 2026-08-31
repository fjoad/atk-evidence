# Small paired training check: does Sigmoid realize the opening?

Date: 2026-08-31

Status: setup recorded before new model code or scoring; user asked to test
whether the Sigmoid-plus-changed-cutoff alternative also fails.

## Question and limits

The preceding complete-evaluation range bound does **not** exclude the
detection/false-alarm pair after changing the output and cutoff. The user's
desired failure is not a premise. This exploratory `X/A` check asks whether
one small, properly trained Sigmoid FC-SAE reaches the target on a fixed
held-out sample, alongside a paired Softmax control.

- A successful Sigmoid outcome is evidence against the proposed failure.
- If every cutoff on the fitted scores fails, that closes the cutoff rescue
  for those fixed models and sampled rows only.
- A short failed fit cannot prove that all Sigmoid weights, training budgets,
  data samples, or configurations fail. No learning curve extrapolation or
  seed-level confidence claim is allowed.

Primary target: DR >=81%, FA <=15%; relaxed rounding target: DR >=80.5%,
FA <=15.5%. The full seven-metric row and the two-metric target remain distinct.

## Fixed source and model

Reuse the unchanged `reproduction/models.py` definition, SHA-256
`3515415082b26bb91cb5367effbd1eba4324bf250ec47e799f7fccb3e6df83f0`:
48 → 400 → 300 → 200 → 100 → 100 → 200 → 300 → 400 → 48;
eight sigmoid hidden layers, dropout 0.4 after each, Adam 0.001, MSE.
The direct builder supplies the template; clone its configuration, changing
only the final activation between Softmax and Sigmoid. Both clones receive
identical initial weight arrays and corresponding fixed dropout seeds.
Validate both with the unchanged runtime checker: 450,448 parameters.
Neither model uses saved trained weights. This is not a post-hoc head swap.

Source locations were already frozen and visually verified: Table I and
Section IV-C p. 4115; MSE p. 4109; Table III p. 4116. Sigmoid changes the
explicit final FC-SAE head, so this is a control, not a literal reproduction.
No change to the five reproduction files is needed.

## Data and deterministic selection

Keep the saved preparation, joint feature scaling, attack generation, and
customer-disjoint B1/B2 split unchanged. Source result SHA-256:
`ae07b42ef6c84242ca9b39db8b8828694d6d4df6859abdee090fc0a613a69154`.
Verify all consumed arrays against that record/metadata before selection.

Use NumPy generator seed 20260831. Draw 3,072 distinct B1 training-array rows:
the first 2,048 are for fitting, the remaining 1,024 for benign calibration.
They are row-disjoint, not claimed to be customer-disjoint within B1. B2
evaluation customers remain disjoint from B1 under the frozen preparation.
No confidence interval treats those rows as independent subjects.

Draw 1,024 B2 source days without replacement, sort their indices, and retain
all six attack siblings. Also draw synthetic benign rows without replacement,
with count `round(1024 * full_synthetic_count / full_B2_day_count)`, preserving
the full evaluation's original/synthetic proportions approximately. Retain
an original-row-only view. Do not regenerate or refit preprocessing.

Save selection indices and scores outside Git, with hashes in the result.
No test labels may affect model fitting, checkpoint selection, or calibration.

## Fit and measurements

Model seed 20260831, batch 32 (same as the original anchor), ten epochs,
fixed sampled row order with `shuffle=False`. Fit only benign inputs to
themselves. Both heads use the same inputs, epoch cap, learning rate,
initial weights, and dropout seed schedule.

Select the checkpoint with the lowest benign calibration MSE; never select
on test detection, AUC, or an oracle cutoff. Record every epoch's fit and
calibration loss, update count, timing, and any early budget stop. Save initial
and selected checkpoint hashes and verify weights changed after fitting.

For each head, score before fitting and after restoring its calibration-best
weights. Record both evaluation views and both score directions:

- all seven metrics at 0.58;
- a label-blind calibrated cutoff: 85th percentile of benign calibration error
  for the high-error rule, using `method="higher"`; 15th percentile for the
  low-error control, using `method="lower"`, with strict inequalities;
- exhaustive cutoff diagnostics on each fixed test score vector: best
  balanced accuracy, AUC, DR at FA<=15% and <=15.5%, and target-pair flags;
- score finiteness, output range, and Softmax row-sum checks.

Oracle test-cutoff diagnostics give the fixed model every cutoff advantage;
they are not validation-selected thresholds or an eligible reproduction.
Preserve every head/direction/stage, including any matching outcome.

## Budget and promotion

Local work is limited to hand-sized software fixtures. Experimental training
and scoring require one CPU compute allocation: four cores, 8 GiB, ten minutes,
no GPU. This is not full-data retraining.

Pilot: 128 fitting rows, 64 benign calibration rows, 8 source days with all
siblings and proportional synthetic benign rows; one epoch per head. Use
the same construction and checks. Promote one small run only if both heads
make finite weight updates, the initial weight hashes match, and data/output
checks pass. Estimate the full run using each head's median of the last three
pilot batch times × 640 updates, plus measured non-batch fit overhead, scaled
scoring throughput, and verification cost; require estimate below 240 seconds.
This avoids projecting first-update startup cost across every later update.
Do not enlarge the allocation or
revise the sample after seeing results.

Per-head fitting cap: 90 seconds, checked after each batch (one-batch timing
granularity). Overall process alarm: 300 seconds; scheduler cap: ten minutes.
If a cap interrupts fitting, preserve partial results and label that head
budget-limited; do not compare it as an equally completed ten-epoch fit.
No automatic retries, new seeds, learning-rate search, or longer run.

## Stop and reporting

Preserve the pilot, single small run, weights, sampled indices, score arrays,
summary JSON, source/code/contract hashes, and failures. Update the internal
finding, explanation register, STATUS, and CONTEXT. Discuss before publishing.
The existing website and numerical reproduction remain unchanged.

Stop after this bounded pair. Report exactly whether the sampled trained
models reach the target, and whether their every-cutoff check excludes it.
Do not turn either a failed fit or an inconsistent complete metric row into
a proof that the two-metric target is universally impossible.
