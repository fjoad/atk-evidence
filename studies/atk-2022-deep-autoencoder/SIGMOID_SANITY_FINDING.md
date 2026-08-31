# Sigmoid changes the answer—but does not reproduce the result

Date: 2026-08-31

Status: checks complete; local discussion finding, not approved for publication

## Short answer

**Changing Softmax to Sigmoid alone still cannot reproduce the stated result
at the printed 0.58 cutoff on these prepared inputs.** Even the most favorable
per-example reconstructions must falsely flag at least 29.66% of benign rows
(lower bound rounded down), versus 15% reported.

**If the cutoff can also change, the complete-evaluation bound no longer rules
the target out.** This is a meaningful opening that was not established by
the earlier original-row check. It is not an achieved result: our two concrete
no-training rules remain far below the target.

## What changed and what did not

We asked: even granting each example its most favorable Sigmoid-range output,
does the reported detection/false-alarm pair remain unreachable when the
paper's complete prepared test set is included?

The [frozen setup](SIGMOID_SANITY.md) kept all original inputs and synthetic
benign rows unchanged. All 8,884,989 rows were evaluated: 750,767 original
benign days, 4,504,602 attack derivatives, and 3,629,620 synthetic benign rows.
No model was trained or modified. The separate original-row view contains
5,255,369 rows and reproduces the preceding bound within 1e-7 percentage points.

For the allowed cube `0 <= r <= 1`, each input's smallest copying error comes
from clipping its coordinates into that range; the largest comes from the
farthest endpoint for each coordinate. The label-aware bound may choose the
largest error for an attack and the smallest for a benign example, even when
one real model could not make those choices. Reversal exchanges their roles.

## The bound on the complete prepared evaluation

Upper limits below are rounded upward; minimum false alarms are rounded down.
These are not trained-model scores.

| Question | Optimistic bound | Interpretation |
|---|---:|---|
| Detection at high-error cutoff 0.58 | At most 95.73% | Detection alone is not excluded |
| False alarms at high-error cutoff 0.58 | At least 29.66% | The printed cutoff still excludes the combined target |
| Detection with FA <=15%, any high-error cutoff | At most 85.33% | The 81% detection / 15% FA pair is no longer excluded |
| Detection with FA <=15.5%, allowing rounding | At most 86.08% | The rounding-tolerant pair is also not excluded |
| Balanced accuracy, any high-error cutoff | At most 85.70% | 83% is not excluded by this bound |
| AUC, high-error direction | At most 90.03% | 81% is not excluded by this bound |
| Detection with FA <=15%, reversed direction | At most 93.77% | Reversal remains an open control, not a reproduction |

The high-error relaxed score reaches its 85.32587% detection cap at a cutoff
near 1.24143, not 0.58. This is the label-aware bound's cutoff; it is **not a
recommended threshold for a trained model**. A permitted pair or separately
permitted metric does not establish one model matching all seven metrics.

At the original error cutoff with reversed scoring, the relaxed maximum DR
is about 85.14009% and minimum FA about 0.14175%. Reversing the rule changes an
explicit paper choice; no learned model is shown to achieve these limits.

## Why the original-row and complete answers differ

The original-row high-error ceiling remains 59.99% detection at FA <=15%.
That earlier result was correct and explicitly limited to original rows.
Adding the saved synthetic benign rows changes the distribution used to
calculate false alarms. On the complete evaluation the ceiling becomes
85.33%; the two evaluation populations must not be interchanged.

The fixed-cutoff attack-detection ceiling is unchanged because no attack row
changed. The original-row minimum FA at 0.58 was about 41.12%; on the complete
evaluation it is about 29.67%. This does not identify an author procedure or
establish that resampling creates useful learned capability.

## Can a real, label-blind rule exploit this opening?

We tested two fixed reconstructions on the same rows. Neither uses class
labels to construct its outputs. Clipping gives the closest vector in the
closed cube; exact endpoints are limits of finite Sigmoid outputs, so this is
not claimed as an implemented Sigmoid network. Constant 0.5 is a valid
Sigmoid output with zero logits. Neither control involves training.

| Control on complete evaluation | DR at 0.58 | FA at 0.58 | Balanced accuracy at 0.58 | Best balanced accuracy, any cutoff | Best DR at FA <=15% |
|---|---:|---:|---:|---:|---:|
| Clipped input, high error | 14.86% | 29.67% | 42.60% | 50.00% | 7.74% |
| Constant 0.5, high error | 78.13% | 86.68% | 45.73% | 50.00% | 8.31% |
| Clipped input, low error | 85.14% | 70.33% | 57.40% | 58.23% | 25.46% |
| Constant 0.5, low error | 21.87% | 13.32% | 54.27% | 55.32% | 23.04% |

Control measurements are rounded to nearest, not stated as universal bounds.
Their all-cutoff maxima are descriptive test-label diagnostics, not selected
reproduction thresholds. Full precision, all seven fixed-cutoff metrics,
confusion counts, both populations, and both directions are in the saved JSON.

The concrete controls did not realize the oracle's advantage. This does not
prove that a trained Sigmoid model could not: the closest reconstruction need
not be the best detector, and only two no-training rules were tested. We did
not make a head swap in Softmax-trained weights and call its outcome a
Sigmoid learning experiment.

## What the evidence changes

- **Numerical:** no new trained reproduction; the original FC-SAE mismatch
  remains unchanged.
- **Mechanism:** no new architectural or learned-mechanism finding.
- **Attainability:** the printed-cutoff exclusion extends to the Sigmoid
  range on the complete fixed evaluation. With a changed cutoff, or with
  reversed scoring, this bound no longer excludes the target pair. Treat
  these as unresolved alternatives, not proven matches or failures.

Former uncertainty → evidence → current conclusion: the earlier original-row
bound left the synthetic benign rows outside scope; checking them permits the
high-error target pair at a changed cutoff. It would be wrong to generalize
the original-row exclusion to the complete Sigmoid evaluation.

## Cost, verification, and stopping

All 223 deterministic tests passed before local code freeze `9d6c31b`.
CPU job `385137` completed `0:0` and released its allocation. The pilot took
4.97 seconds; full analysis 39.70 seconds, including 4.22 seconds of hash
verification and 11.73 seconds of geometry. Full process time was 42.02
seconds; total allocation including inspection was 2:17. Maximum full-process
RSS was 2,867,616 KiB. The code uses outward-padded float64, not certified
interval arithmetic. No confidence interval over training seeds is implied.

[Saved pilot, full results, and execution record](results/sigmoid_sanity_20260831/)
preserve all outcomes and hashes. Original data, weights, scores, and numerical
results are unchanged. No scientific execution was repeated.

Post-run verification passed all 225 deterministic tests (140 study, 85 root),
including eight Sigmoid checks and provenance/display guards. All 62 local
Markdown links checked across nine relevant documents resolve; `git diff
--check` passes. README, website, and reproduction files remain unchanged
from the published `dc37bbe` revision. The frozen script and contract were
not modified after execution.

The already discussed source findings are live at public revision `dc37bbe`;
these new findings are local only. **Stop for discussion.** No full training,
small fitted comparison, additional seed, new branch, public edit, or push
follows automatically. A possible next question is whether a tiny, properly
trained Sigmoid control with a non-test-selected cutoff shows a route; it
would require its own limited setup and approval.
