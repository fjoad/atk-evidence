# ATK Evidence — Current Status

**Last updated:** 2026-08-11

**Branch:** `main`

**Active plan:**
[`plans/2026-08-09-paper-1-minimal-finish.md`](plans/2026-08-09-paper-1-minimal-finish.md)

## Current project state

- The paper-first minimal-instrument reframe is accepted and recorded in
  [`decisions/2026-08-09-paper-first-minimal-instrument.md`](decisions/2026-08-09-paper-first-minimal-instrument.md).
- Paper 1 is the only active execution target. Paper 2's existing artifact-level
  audit remains preserved but frozen; cross-paper synthesis has not started.
- Panther job `373789` completed successfully in 53:12 from commit `c8c136f`
  on one 16-GB V100. It passed source/preparation gates, trained the frozen
  batch-512 FC-SAE for 74 epochs (best epoch 69), and saved Tables III/full-IV/V
  artifacts. Panther CPU job `373799` measured the selected exact ADASYN
  neighbor implementation, and score-audit job `373800` completed. Premature
  repeated-seed jobs `373803` and `373804` were cancelled before execution
  after the breadth-first correction. One-factor linear-output job `373805`
  completed successfully in 1:05:43 on one 16-GB V100; audit job `373824`
  completed in 24 seconds. Naive Bayes (`373833`), ARIMA (`373836`), one-class
  SVM (`373837`), supervised feed-forward (`373838`), multiclass SVM (`373840`),
  and FC-VAE (`373842`) are complete. The first supervised-LSTM attempt
  (`373839`) trained but failed during oversized-batch scoring; the first three
  proposed recurrent attempts (`373841`, `373843`, `373844`) failed before
  training in diagnostic layer inventory. Those defects were fixed in
  `c735dd9`, and corrected supervised-LSTM job `374310` completed. A second
  pre-training sanity-probe batch leak caused jobs `374311`--`374313` to fail
  before training; it is fixed in `4469a53`. Jobs `374388`--`374390` were then
  rejected by the immutable-attempt guard before execution because their
  configuration still named score batch 512. Explicit score batch 256 makes
  the preserved retry identity distinct. LSTM-SAE job `374391` completed and
  score-audit job `374433` closed its threshold question. LSTM-VAE job `374395`
  completed training but exhausted 16-GB GPU memory during scoring; the weights
  are preserved, and fresh-process scoring job `374435` is running at inference
  batch 64 without retraining. LSTM-AEA job `374396` remains healthy. The Slurm
  wrapper only requests `gpu-all` plus `gpu:1` and executes its arguments.
- Experimental preparation, training, and scoring must run on cluster compute
  nodes. Local work is limited to source reconstruction, code, documentation,
  lightweight inspection, transfer, and monitoring.
- The renewed source freeze is accepted. The five direct files have now been
  traced against the corrected `METHOD.md`; the unfinished score-audit work was
  completed without importing the historical `src/` route.
- The next experiment is frozen to the single baseline in
  [`../studies/atk-2022-deep-autoencoder/reproduction/BASELINE.md`](../studies/atk-2022-deep-autoencoder/reproduction/BASELINE.md):
  full ISET, FC-SAE, seed 11, batch 512, original `B2+M`, Tables III/full-IV/V.
  No ambiguity sweep is authorized before this anchor is inspected.

## Paper 1: verified foundation

- The exact 12-page PDF was independently re-audited on 2026-08-11: SHA-256
  verified, text extracted, and every rendered page visually inspected before
  the prior reconstruction was opened. The corrected source-located executable
  reconstruction is
  [`../studies/atk-2022-deep-autoencoder/METHOD.md`](../studies/atk-2022-deep-autoencoder/METHOD.md).
- The re-audit confirmed the main flow and prior pivotal contradictions, fixed
  the benchmark count (six, not seven), and added omitted VAE-derivation,
  decoder, precision-definition, F1, and common-prevalence inconsistencies.
- `P0-ISET-FCSAE` is now labeled precisely as a paper-primary `P+I` executable
  completion. The printed Attack-3 subtraction remains a non-executable source
  outcome rather than being silently called literal.
- The accepted non-executable-source contract requires the literal failure and
  every predeclared reasonable repair to be executed and reported side by side
  with the published target; see
  [`decisions/2026-08-11-non-executable-source-ladder.md`](decisions/2026-08-11-non-executable-source-ladder.md).
- Exact CER/ISET consumption archives are verified. The allocation CSV is the
  explicitly labeled semantic-equivalence branch, not the official `.tab`
  serialization.
- The five direct reproduction files exist under
  `studies/atk-2022-deep-autoencoder/reproduction/` and do not import the
  historical forensic `src/` implementation.
- Full preparation produced 2,251,290 benign profiles, 13,507,740 attacked
  profiles, 1,500,520 B1 training profiles, and the 14,258,510-row `B2+M` test
  population.
- Printed-position default ADASYN is not complete: its exact default full-scale
  neighbor query entails roughly 10.7 trillion first-pass distances. A 16-core
  same-machine benchmark estimates 14.16 wall-hours for both exact neighbor
  searches alone, before synthesis and persistence. This is expensive but
  feasible as an overnight job and gives no basis for claiming that the authors
  could not have run ADASYN. Do not relabel the no-resampling interpretation as
  the printed result.

## Paper 1: current experimental evidence

The new compact batch-512 anchor completed:

- method: `I-ADASYN-NONE-ISET-FC-SAE`, seed 11, batch 512;
- reproduced DR/FA/ACC/AUC/F1:
  26.18 / 58.22 / 33.98 / 31.04 / 40.46%;
- reported DR/FA/ACC/AUC/F1: 81 / 15 / 83 / 81 / 81%;
- fit/total time: 45:07 / 47:41 inside the pipeline; and
- Table-V FA is exactly 57.9152% for all six attacks, as required by the frozen
  common-model/common-benign interpretation but unlike the paper's varying FA.

This is a completed exploratory no-resampling anchor, not printed-ADASYN `P0`
and not yet a confirmatory verdict. Its score-distribution/reload audit is now
complete: an oracle threshold in the paper direction reaches only 50.00%
balanced ACC; reversing direction reaches 66.26%; and the trained score ranking
is 0.99946-correlated with the zero-reconstruction control. The committed
result records are
[`../studies/atk-2022-deep-autoencoder/results/compact_route_fc_sae_seed11_batch512_20260811.json`](../studies/atk-2022-deep-autoencoder/results/compact_route_fc_sae_seed11_batch512_20260811.json)
and
[`../studies/atk-2022-deep-autoencoder/results/iset_fc_sae_seed11_score_audit_20260811.json`](../studies/atk-2022-deep-autoencoder/results/iset_fc_sae_seed11_score_audit_20260811.json).

The one-factor linear-output control also completed:

- method: `C-OUTPUT-LINEAR-ISET-FC-SAE`, seed 11, batch 512;
- only change from the anchor: final output activation Softmax to linear;
- reproduced DR/FA/ACC/AUC/F1:
  12.32 / 30.78 / 40.77 / 28.14 / 21.61%;
- paper-direction oracle ACC is only 50.04%; reversing direction reaches 67.56%;
- benign mean error is 0.537 versus malicious 0.281, so the average ordering is
  opposite the paper's decision rule;
- trained versus zero-reconstruction score correlation falls from 0.99946 to
  0.82089, showing that the activation changes the learned score materially but
  still does not produce the reported separation; and
- Table-V FA is exactly 30.0696% for every attack under the common all-benign
  evaluation population, again unlike the paper's varying FA cells.

This closes output activation *alone* as a sufficient explanation for the
baseline gap. It remains a one-seed corrected control, not a paper-level verdict.
The committed summary is
[`../studies/atk-2022-deep-autoencoder/results/iset_fc_sae_linear_seed11_score_audit_20260811.json`](../studies/atk-2022-deep-autoencoder/results/iset_fc_sae_linear_seed11_score_audit_20260811.json).

The benchmark reuse check found no preserved ISET/Table-III benchmark attempt
in either the committed summaries or Panther's attempt manifests. Existing
benchmark results are SGCC/Table II and cannot fill the current ISET breadth
rows. Minimal ISET Naive Bayes job `373833` completed on Panther. It uses
Gaussian Naive Bayes, all original all-customer `B+M`, and an exact seeded 2:1
row split. The omitted supervised ADASYN step is explicit in method
`I-SUPERVISED-ADASYN-NONE-ISET-NAIVE-BAYES`; this is not the printed branch.
It reproduced DR/FA/ACC/F1/AUC = 88.78/44.53/72.12/90.50/79.17%, versus
73/18/77.5/73/70% reported. This completion does not reproduce the reported
operating point.

ARIMA job `373836` then completed in 1m02s using the predeclared smallest
pooled `(1,1,0)` residual-MSE completion on all B1 and original `B2+M` rows.
It reproduced DR/FA/ACC/F1/AUC = 21.48/57.20/32.14/34.46/24.72%, versus
86/12/87/86/87% reported. It is a large non-match with reversed/worse-than-
chance ranking, but the paper omits `p`, fit unit, and score, so it is not a
claim over all ARIMA implementations.

One-class SVM job `373837` completed in 1m04s using the explicit
`kernel=sigmoid, gamma=scale, nu=0.5` repair with a 12,000-row training cap and
30,000-row test cap. It reproduced DR/FA/ACC/F1/AUC =
91.87/50.94/70.47/94.35/79.67%, versus 90/9/90.5/89.5/87% reported. Its DR is
near the paper only at a false-alarm rate above 50%; this bounded diagnostic
does not reproduce the reported operating point and cannot fill the full cell.

Score-audit jobs `373854` and `373855` completed. Oracle-threshold analysis
separates the failures: Naive Bayes is primarily an omitted operating-point
issue (oracle ACC 74.74%, closest reported DR/FA gap 5.00 points); pooled ARIMA
is a fundamental paper-direction failure (oracle ACC 50.00%, reversed 69.74%,
closest gap 56.56 points); capped one-class SVM has useful ranking but cannot
reach the claimed high-DR/low-FA corner (oracle ACC 73.86%, closest gap 18.31
points). These are exploratory score diagnostics, not repeated-seed inference.

Supervised feed-forward job `373838` completed in 1:29:56 using the printed
five 500-unit ReLU hidden layers and Adamax, with the predeclared two-class
Softmax/categorical completion and no supervised ADASYN. At the ordinary 0.5
cutoff it reproduced DR/FA/ACC/F1/AUC =
96.41/23.72/86.35/96.24/97.05%, versus 90/11/89.5/89.5/88% reported. Score
audit job `374255` found that a threshold of 0.824 reaches DR=91.83% and
FA=9.17%, within 1.83 points of the reported pair; the best balanced threshold
reaches 91.66% ACC. This is therefore a threshold-procedure ambiguity with a
strong learned ranking, not a fundamental separation failure. The paper does
not specify a supervised threshold-selection rule, and this branch still omits
pre-split ADASYN.

Multiclass SVM job `373840` completed in 2m27s using the explicit seven-class
sigmoid/scale repair and deterministic 30,000-row train/test caps. Fixed
DR/FA/ACC/F1/AUC = 85.94/55.67/65.14/88.04/73.06%, versus
91/8/91.5/90.5/89% reported. Audit job `374302` found a best balanced ACC of
71.14%; the closest threshold remains 23.44 points from the reported DR/FA
pair. This bounded completion does not reproduce the operating point.

FC-VAE job `373842` completed in 6m14s and restored epoch-2 weights after seven
epochs. Under the predeclared missing-score completion
`exp(-0.5 * profile MSE)`, fixed DR/FA/ACC/F1/AUC =
11.51/32.62/39.45/20.32/30.13%, versus 88/11/88.5/88.5/85% reported. Audit job
`374303` found only 50.00% oracle ACC in the paper's low-probability direction;
reversing direction reaches 66.70%. Malicious mean probability 0.750 exceeds
benign 0.567, and the trained score correlates 0.99957 with the corresponding
zero-reconstruction control. This is a strong failure of the registered VAE
score completion, while the source's missing probability definition keeps
other materially distinct completions open.

The recurrent failures are preserved as evidence, not silently discarded.
Job `373839` saved trained weights before its 8,192-row scoring batch exhausted
the V100. Jobs `373841`, `373843`, and `373844` stopped before training because
the diagnostic inventory assumed one tensor per layer output. The repair only
records multi-output shapes and reduces recurrent inference batches to 512; it
does not alter a model, optimizer, training batch, data row, score, or metric.

Corrected supervised-LSTM job `374310` completed in 6:54:51 after six epochs,
restoring epoch-1 weights. Every test score is exactly 1.0, yielding
DR/FA/ACC/F1/AUC = 100/100/50/92.32/50%, versus
90.5/10/90/90/89% reported. Audit job `374387` confirms that both score
directions have oracle ACC 50%; the closest threshold is 90 points from the
reported DR/FA pair. This registered one-seed completion collapsed completely
and is not a threshold-selection failure.

LSTM-SAE job `374391` completed in 6:23:28 after 25 epochs, restoring epoch-20
weights. Fixed DR/FA/ACC/F1/AUC =
14.78/40.96/36.91/25.25/33.09%, versus 85/13/86/85/82% reported. Audit job
`374433` found paper-direction oracle ACC 50.004%; its closest point to the
reported DR/FA pair remains 47.11 points away. Reversing direction reaches
64.38% ACC. Benign mean reconstruction error 1.087 exceeds malicious 0.519,
and the trained score is 0.97495-correlated with zero reconstruction. This is a
fundamental wrong-direction failure for the registered one-seed no-test-ADASYN
completion, not a threshold mismatch.

One full compact-route cluster result exists:

- table/model: Table III, FC-SAE;
- method: `I-ADASYN-NONE-ISET-FC-SAE`;
- seed/batch: 11 / 32;
- population: exact original pre-ADASYN `B2+M` rows;
- epochs: 29;
- reproduced DR/FA/ACC/AUC/F1:
  26.44 / 58.51 / 33.97 / 31.03 / 40.78%; and
- reported DR/FA/ACC/AUC: 81 / 15 / 83 / 81%.

This is one exploratory batch-32 sensitivity, not a reproduction verdict and not
the batch-512 primary branch. Its full score arrays, weights, history, predictions,
and hashes are local. Its score/eligibility audit is unfinished.

## Exact next action

1. **Complete:** validate the one-factor linear-output control and record that
   output activation alone does not rescue the score separation.
2. Complete LSTM-VAE score recovery `374435` and harvest LSTM-AEA `374396`;
   then the
   one-factor population, split, scaling, threshold, and Attack-3
   interpretations. Existing historical results count only if they pass the
   renewed source/provenance/score gates.
3. Record the complete breadth map. Only then freeze the finite surviving
   branches and deepen them with repeated seeds and confirmatory intervals.

## Not on the critical path

- the historical 921-configuration branch matrix;
- new scheduling, manifest, DDP, or workflow infrastructure;
- Paper 2 execution;
- website or LaTeX polishing;
- the corrected preferred detector; and
- cross-paper conclusions.
