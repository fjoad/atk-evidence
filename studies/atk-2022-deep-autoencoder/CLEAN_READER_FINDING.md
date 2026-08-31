# Clean-reader initial numerical finding

**Date:** 2026-08-31

**State:** Phase 6 complete; Checkpoint 2 awaits user review

**Question:** `N` — numerical reproduction

**Implementation:** `P+I`, `CR-ISET-FCSAE-01`, one seed `20260824`

## Outcome and exact scope

The approved competent-reader completion executed successfully, but **did not
reproduce the paper's Table-III ISET FC-SAE row**. The numerical gap survives
independent artifact verification. This is one model, one dataset, one frozen
completion, and one seed—not a finding about every model or the entire paper.

The scientific execution used commit
`a88d17477ad96b01ffa44a50d8ce051dd8d2b5ca`, Panther job `384390`, and immutable
attempt `seed_20260824_2f483335536c`. It ran once, without target-guided changes.
The frozen [specification](CLEAN_READER_SPECIFICATION.md) and
[pre-run contract](CLEAN_READER_ANCHOR_PRERUN.md) remain the scope authority.

The literally printed route retains its previously documented non-executable
operations. The successful run is the explicitly interpreted completion,
including duration-first Attack 3, all 4,225 residential meters, customer-disjoint
training/test customers, attacks restricted to B2 in evaluation, joint scaling,
printed-position test ADASYN, and the approved semantic allocation CSV. It is
not evidence that the authors used these omitted choices.

## Complete reported-versus-reproduced row

All values are percentages. Deltas are reproduced minus reported, in percentage
points. Source: Table III, printed p. 4116 / PDF p. 11, independently checked
against [the source-transcribed CSV](reported/table_3.csv).

| Metric | Paper | This run | Delta |
|---|---:|---:|---:|
| Detection rate, DR | 81.00 | 25.48 | -55.52 |
| False-alarm rate, FA (lower is better) | 15.00 | 45.13 | +30.13 |
| Specificity, SP | 85.00 | 54.87 | -30.13 |
| Precision, PR | 81.00 | 36.73 | -44.27 |
| Balanced accuracy, ACC | 83.00 | 40.18 | -42.82 |
| F1 | 81.00 | 30.09 | -50.91 |
| Ranking AUC | 81.00 | 39.40 | -41.60 |

`ACC` follows the frozen paper formula `(DR + specificity) / 2`, not ordinary
prevalence-weighted accuracy. Exact confusion counts are TP 1,147,951;
TN 2,403,359; FP 1,977,028; FN 3,356,651.

The evaluation contains 4,380,387 benign rows and 4,504,602 malicious rows,
8,884,989 total. ADASYN generated 3,629,620 benign rows. Its integer allocation
does not produce exact class equality; the actual counts are preserved rather
than forced to the requested target or silently described as exactly balanced.

## Why the saved result is trustworthy within this scope

The unchanged predeclared audit **passed**. Source-branch/configuration gates,
prepared/run artifact hashes, saved predictions, all seven metrics, all four
confusion counts, history length, and score/floor alignment passed. The maximum
recorded-versus-recomputed metric difference was exactly zero.

The supplemental read-only check also **passed all 65 checks**:

- Full scans of 31 NumPy artifacts found zero non-finite values. The scan
  covers 1,713,359,202 stored scalar values, including redundant arrays and
  identities; these are not independent statistical observations.
- All 2,251,290 source profiles partition into training/B2 without omission or
  duplication. The 2,816 training customers and 1,409 test customers do not
  overlap.
- Training arrays agree with their source indices. All seven original test
  blocks agree with the corresponding benign/attack arrays indexed by B2.
- Original features, labels, and provenance survive resampling unchanged;
  appended synthetic rows are benign and have the declared synthetic markers.
- Scaler scales, model weights, and every loss are finite; scales are positive.
- Replaying the stopping rule gives 28 epochs, best epoch 23, exactly as saved.
- A fresh CPU model loaded the saved weights and reproduced scores on 256
  evenly distributed rows across benign, six attacks, and synthetic benign
  blocks. Maximum absolute CPU-versus-saved score difference was
  `1.1920928955078125e-7`; every sampled prediction agreed. This verifies a
  sample of fresh inference, not a second full scoring run.

The Keras optimizer-state loading warning is retained. It concerns optimizer
slots in a fresh inference model, not a failed layer-weight reload; finite
weights, the runtime inventory, and fresh inference agreement passed. No
training was resumed. The largest floor-minus-trained-score difference was
`7.62939453125e-6`, within the frozen `1e-5` floating-point tolerance.

The first supplemental checker itself had a last-chunk slicing defect: it
compared the original array's short final chunk with a longer resampled slice
that extended into synthetic rows. Its failed JSON is preserved. Clamping both
slices to the original row count corrected only that checker; the second JSON
passed. Original checker source is preserved in commit `281afc2`; reproduction
code, weights, scores, data, and metrics were never altered or rerun.

## Diagnostics, kept separate from reproduction

**Threshold choice alone does not rescue these saved scores.** Exact enumeration
of 7,202,671 threshold candidates gives maximum balanced accuracy 50.00072% in
the paper's high-error direction. Even the threshold closest to the complete
seven-metric target leaves a maximum gap of 41.60 points. The closest DR/FA pair
still misses at least one of those two targets by 40.75 points.

A post-hoc reversed-direction control reaches best balanced accuracy 60.21%
and AUC 60.60%, still below 83% and 81% reported. These test-label oracles are
diagnostics, not eligible repaired reproduction results, and they say nothing
about new scores produced by a different training procedure.

**The score often points in the wrong direction.** Mean error is 1.19676 for
benign rows versus 0.81304 for malicious rows, although higher error triggers
the alarm. AUC 39.40% confirms aggregate ranking is worse than chance in that
orientation; it does not mean every individual attack has the same behavior.

**Simple score geometry is a live explanation.** Trained scores have Pearson
correlation 0.999253 with zero-reconstruction input energy. The zero rule gives
38.99771% ACC and 37.22922% AUC, versus 40.17516% and 39.39865% trained: gains of
1.18 and 2.17 points, respectively. Correlation is not equality, and training
did change the score. The per-row Softmax-domain projection floor gives
40.20211% ACC and 39.35160% AUC. This motivates isolating output-domain/scoring
effects; it does not prove the network learned nothing or that architecture can
never help. A reconstruction-error floor is not a classifier-performance bound.

The frozen audit also emits per-attack diagnostic rows. Its field name
`FA_on_common_B2` refers here to all resampled benign test rows, including
synthetic rows—not only the original B2 rows. Do not present those rows as a
Table-V reproduction. Likewise the runner's stored Table-IV target field does
not expand this Table-III-only contract.

## The three-question boundary

- **N — initial numerical finding:** not reproduced in this single declared
  `P+I` completion; the instrument passed the recorded audits. No seed-level
  uncertainty interval or result over other completions is claimed.
- **M — mechanism:** unresolved. Geometry/energy diagnostics motivate a
  controlled question, but no new LSTM/FC contrast, component ablation, or
  structure-removal experiment has been performed here.
- **A — attainability:** unresolved for the method family. Fixed-score
  threshold reachability is checked, but there is no declared multi-seed,
  capacity, optimization, or source-completion performance envelope.

E11 (our implementation is wrong) is weakened for the specific checked chain,
not eliminated in principle. E7/E8 deserve focused follow-up; alternative
source completions, data choices, optimization effects, variation, and reporting
error remain open. No author intent or undocumented implementation is inferred.

## Timing, records, and next checkpoint

The original job completed `0:0` in 9:14:27, ending 2026-08-30 22:48:39 Qatar.
Extraction plus preparation took about 1:09:43, including 1:04:51 of ADASYN;
fitting took 8:02:46 on a Tesla P100; scoring took 11.12 seconds. Audit-only CPU
allocation `384939` completed `0:0` in 8:39 including interactive pauses. No new
training job was launched. The unchanged deterministic suite passes 179 tests.

Small unmodified result, metadata, history, log, frozen audit, failed checker,
and corrected checker records are saved in
[results/clean_reader_anchor_20260831](results/clean_reader_anchor_20260831/).
Large arrays and weights remain on Panther at the paths and hashes in those
records. The supplemental checker is
[checks/clean_reader_anchor_artifacts.py](checks/clean_reader_anchor_artifacts.py).

**Stop at Checkpoint 2.** Review this finding before promoting any additional
experiment. A cheap saved-artifact or exact-data diagnostic targeting the
remaining geometry/scoring/triviality explanations is a candidate next step;
repeated full training or model-family expansion is not automatic.
