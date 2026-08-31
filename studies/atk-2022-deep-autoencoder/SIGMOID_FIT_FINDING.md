# A trained Sigmoid model still missed the target on the small check

Date: 2026-08-31. Exploratory `X/A`; saved locally for discussion.

## What we asked

**Does replacing Softmax with Sigmoid actually produce the missing detection
performance, if we also allow the most favorable cutoff?**

The preceding range calculation left that alternative open on the complete
prepared evaluation. A permissive bound is not an achieved model. We therefore
trained a small paired Softmax/Sigmoid comparison, without assuming failure.
The [setup](SIGMOID_FIT_CHECK.md) was frozen before execution in `cc9af5e`.

## What ran

Both models retain the exact FC-SAE layer widths, eight sigmoid hidden layers,
dropout 0.4, 450,448 parameters, Adam 0.001, MSE, and batch size 32. Only the
final activation differs. They start with identical weight arrays and matched
dropout seeds. This is genuine separate training, not a head swap on previously
trained Softmax weights; the five reproduction files remain unchanged.

The fixed sample contains 2,048 benign fitting rows, 1,024 separate benign
calibration rows, and 12,119 held-out evaluation rows: 1,024 source days,
6,144 attack siblings, and 4,951 synthetic benign rows. The saved preparation
and its B1/B2 customer split are unchanged. Fit/calibration rows are disjoint
but not asserted to be customer-disjoint within B1.

Both models completed ten epochs and 640 updates, without budget interruption.
The checkpoint was selected only by benign calibration MSE: epoch 8 for
Softmax and epoch 10 for Sigmoid. Test labels selected neither model weights
nor the label-blind calibrated cutoff. They were used only for evaluation and
the explicitly optimistic all-cutoff diagnostic.

## What happened

The target is **at least 81% detection with no more than 15% false alarms**.
On the sampled prepared evaluation:

| Fitted model / direction | Largest detection at FA <=15% | At FA <=15.5% | Best balanced accuracy over cutoffs | AUC |
|---|---:|---:|---:|---:|
| Softmax, high error | 8.64258% | 8.83789% | 50.05667% | 37.68698% |
| Sigmoid, high error | 9.74935% | 9.99349% | 50.33353% | 37.70499% |
| Softmax, reversed rule | 25.52083% | 26.12305% | 61.48292% | 62.31302% |
| Sigmoid, reversed rule | 25.39063% | 25.81380% | 61.64614% | 62.29501% |

The relaxed 80.5% detection / 15.5% false-alarm target also fails. These are
different operating points and metrics, not a single jointly achieved row.
The two high-error AUCs are almost equal on this sample; this is an observation,
not a statistical equivalence finding or a mechanism test.

At the paper's cutoff 0.58, fitted Sigmoid has **25.24414% detection and
47.02929% false alarms**. A cutoff selected without test labels from benign
calibration data gives **7.12891% detection and 9.92469% false alarms**.
Choosing every cutoff with hindsight still raises detection only to 9.74935%
while keeping false alarms at most 15%.

The original-row-only view also fails: Sigmoid's maximum detection at FA<=15%
is 6.25% for the printed direction and 30.59896% for reversal. Both initial
untrained models also fail on both views and directions. Every outcome,
including all seven fixed/calibrated metrics and the pilot, is retained in
the [unchanged result records](results/sigmoid_fit_20260831/).

## What is actually proved—and what is not

**VERIFIED, conditional fixed-score statement:** on the 12,119 sampled rows,
no single cutoff on either selected model's saved MSE scores achieves the
target pair, in either direction. Predictions change only when a cutoff
passes a distinct score. The diagnostic enumerated all 12,060 ROC boundaries,
including ties and the no-detection case; this was not a sparse threshold grid.
For the selected Sigmoid high-error scores, allowing at most 896 false alarms
among 5,975 benign rows permits at most 599 detections among 6,144 attacks.
There is no untried in-between cutoff that supplies the missing detections.

**OBSERVED:** changing the output head and retraining under this small common
budget did not realize the permissive geometric ceiling. It therefore does
not provide a quick practical rescue under this tested setup.

**OPEN:** other fitted weights, more data, longer training, other optimization
settings, other preparations, or different scoring functions. This sample
does not extend the earlier Softmax all-weights proof to Sigmoid. It does not
show that the target is globally impossible, that training added zero useful
information, or what code the authors ran.

There is important counterevidence to a blanket plateau claim: Sigmoid's
calibration MSE fell from 1.61028 after epoch 1 to 1.33863 after epoch 10, with
a substantial improvement around epochs 5–6. Its best checkpoint is the last
epoch tested. We have **not** established a long-run plateau or that further
training cannot change its score ordering. Reconstruction improvement and
successful attack detection are distinct outcomes. On this sample, high-error
Sigmoid AUC actually fell from 43.93233% initially to 37.70499% after fitting;
reversed-direction AUC rose correspondingly. Do not erase either observation.

No seed-level confidence interval, general-population guarantee, search-time
extrapolation, or misconduct conclusion follows from this one small pair.

## Integrity, cost, and stop

- All consumed array hashes, source result/metadata, model-source hash, sample
  identities, finite inputs/outputs/scores/losses, output ranges, and identical
  starting-weight checks passed. Both weight sets changed after fitting.
- Sigmoid outputs remained in [0,1]; selected range was approximately
  0.0000784–0.9996473. Softmax row sums stayed within 2.39e-7 of one.
- CPU job `385198` completed `0:0`. Pilot analysis: 9.22 seconds. Small pair
  including verification/scoring: 24.81 seconds; fitting took 9.38 seconds
  for Softmax and 9.39 for Sigmoid. Process wall times were 106.09 and
  34.90 seconds including imports; total allocation was 3:52. The slow pilot
  startup is included, not hidden inside the faster analysis time.
- All 230 deterministic tests passed before freeze. Result files were copied
  byte-for-byte and their remote/local hashes match. Weights and scores remain
  outside Git; their hashes and locations are in the
  [execution record](results/sigmoid_fit_20260831/execution.json).
- Post-run verification passed all 232 deterministic tests (140 study, 92
  root), including frozen-source/transfer provenance and guards preserving
  both the cutoff failure and the unresolved broader Sigmoid possibility.
  All 73 checked local documentation links resolve; public files and the five
  reproduction files remain unchanged from `dc37bbe`.

The authorized pair is complete. Stop for discussion. No further training,
sweep, website/README/report edit, or push is authorized by this outcome.
