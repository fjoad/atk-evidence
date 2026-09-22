# First feed-forward pilot

**Date:** 2026-09-22. **Scope:** original twenty-customer pilot, one paired
initialization, p00/p30; not full Table III reproduction.

The six-hidden-layer network learns useful discrimination under the explicit
standard-BCE repair. Poisoning lowers ranking quality, but its large default
detection drop includes a substantial cutoff effect. At the corresponding
published false-alarm limits, saved-score detection meets the unpoisoned
printed DR to one-decimal rounding and exceeds the poisoned printed DR.
The full metric tuples do not reproduce, and the population is still a pilot.

## Frozen implementation and resumed execution

[FEED_FORWARD_PILOT.md](../../FEED_FORWARD_PILOT.md), models.py and the direct
runner were frozen at b5da23a7a5392d2c28b81d037869bfff637b4a48 before fitting.
The paper's six 500-unit ReLU hidden layers, Sigmoid output, Adamax,
no dropout, constraint 3, 50 epochs and batch 100 were retained. One binary
output, MaxNorm(3, axis=0) on all Dense kernels, Glorot initialization, and
Adamax learning rate 0.002 are explicit completions. The repeated-log(p)
printed loss remains documented; these fits use standard binary cross-entropy
and cannot be described as a literal run of Equation(1).

After the September 21 VPN interruption, the existing dependency setup job
398992 was checked rather than repeated. It had completed 0:0 in 12:54,
within its 15-minute limit. On September 22, the unchanged frozen pair ran as
GPU job 400825, completed 0:0 in 3:28 within 20 minutes. No experimental retry,
extra seed, parameter change or best-epoch selection occurred.

## Complete metrics

All values are percentages. Printed values concern the paper's larger
population and are context, not matched-population reproduction targets.

| Metric | Paper p00 | Pilot p00 | Paper p30 | Pilot p30 |
|---|---:|---:|---:|---:|
| DR | 90.8 | 89.14027 | 76.0 | 51.58371 |
| FA | 9.3 | 7.45342 | 24.4 | 0.53239 |
| SP | 90.7 | 92.54658 | 75.6 | 99.46761 |
| PR | 90.0 | 92.14219 | 75.8 | 98.95833 |
| ACC | 90.7 | 90.86022 | 75.8 | 75.76165 |
| F1 | 90.4 | 90.61638 | 75.9 | 67.81678 |
| AUC | 91.1 | 96.35480 | 76.1 | 90.89064 |

ACC is ordinary accuracy. Default labels use p>0.5, ties benign. TP/FN/FP/TN
are 985/120/84/1043 at p00 and 570/535/6/1121 at p30. Poisoned accuracy
rounds to the paper's 75.8%, while detection, false alarms, precision, F1 and
AUC differ. One matching column is not a reproduced row.

The exact same 4,464 training and 2,232 test rows were used at both levels.
Only the declared 675 malicious training-label flips differ, from the same
six selected customers (15.12097% of all training rows). The original
pre-split ADASYN and known dependent observations remain unchanged.

## Fixed-score cutoff checks

| False-alarm cap | p00 best DR | Actual p00 FA | p30 best DR | Actual p30 FA |
|---|---:|---:|---:|---:|
| 9.3% | 90.76923 | 9.05058 | 77.46606 | 9.05058 |
| 17.6% | 94.02715 | 17.39130 | 85.33937 | 17.48004 |
| 24.4% | 95.47511 | 24.22360 | 88.86878 | 24.04614 |
| 33.3% | 96.38009 | 31.76575 | 91.31222 | 33.09672 |

At p00's printed 9.3% FA allowance,90.76923% DR rounds to 90.8%, matching
the printed detection rate while using fewer false alarms. Do not describe
the 0.03077-point difference from the displayed 90.8 as a meaningful miss.
At p30's 24.4% allowance, best DR 88.86878% exceeds the displayed 76.0%.
Neither comparison matches the whole seven-metric row or validates the full
dataset experiment. The favorable thresholds used the test labels and are
diagnostics, not independently validated deployment calibration.

Default DR falls 37.55656 points, but AUC falls 5.46415. At a common9.3%
FA cap, DR falls 13.30317 points; at 24.4%, it falls 6.60633. Useful ranking
survives, while genuine ranking deterioration remains. All 1,700/2,149 ROC
boundaries were retained; reversing the score does not improve these corners.

The constant score has AUC 50%; negative daily mean has AUC 66.19624%.
Original-only AUC remains96.09277/90.07344. Original benign FA is
10.38251/1.09290% (19/183 and 2/183), versus synthetic benign FA
6.88559/0.42373% (65/944 and 4/944). Excluding synthetic evaluation rows
does not undo their role during preparation or training dependence.

## Actual learning, pairing and persistence

Both models have1,277,501 parameters and completed 50 epochs, with 45 batches
per epoch and 2,250 optimizer updates. Both started from initial-weight SHA256
a86bd4ec1fefcd36469a61acf08f5b62f2be01fd7fa9b573163e1ba288bc0d74.
Their final weight hashes differ from initialization and from each other.
No checkpoint was selected using test performance.

Training loss changes from 0.52485 to 0.08566 at p00 and 0.58365 to 0.13509
at p30. Epoch50's recorded within-epoch accuracies are 95.43011/93.52599%;
fixed final-weight training accuracy against observed labels is95.99014/
93.14516%, and against true labels 95.99014/79.27867%. These two accuracy
definitions differ because weights change during an epoch. The final few
loss values still change; no long-run plateau or unlimited-time bound is claimed.

Every Dense kernel ends below the max-norm bound 3; largest incoming norm is
1.59389/1.61300. This verifies the final constraint, not equivalence with other
unreported constraint interpretations. The original models were saved and
loaded on the same GPU/batches; all test probabilities and predictions matched
exactly after reload. The printed loss defect and repaired label gradients
were checked separately on constructed examples before fitting.

## Hardware, budget and checks

Actual device: Tesla V100-PCIE-16GB, compute capability 7.0, 16,384 MiB,
driver 570.133.07. The GPU constructed-update preflight, weight placement
and inference device all identify GPU:0. TensorFlow 2.16.2/Keras 3.4.1,
CUDA build 12.3, pinned NVIDIA libraries, float32, deterministic operations,
TF32 off, XLA off and no mixed precision. Four CPUs/16 GiB host memory were
allocated; the complete job took 208 seconds. No CPU fallback occurred.

Model fitting took 18.28386/17.76314 seconds; training/test scoring took
0.48923/0.45717, and fresh reload plus test scoring 0.44450/0.46651.
TensorFlow allocator peaks were 28,300,288/28,490,240 bytes. These are
allocator counters, not complete driver/process VRAM use. Host process peak
RSS was 987,800/995,536 KiB; Slurm sampled 986,584 KiB for the batch.
The original paper reports an RTX 2070 and full-data deep-model times of
1.5-3 hours. This small pilot neither reproduces nor refutes that runtime claim;
VRAM size alone does not establish a hardware speed ratio.

Before freeze, 327 main-suite tests passed with 10 environment-specific skips;
all eight neural and seven AdaBoost tests passed in their isolated environments.
All eight neural fixtures passed again on the allocated GPU before research
input loading. Startup logs include TensorFlow duplicate-plugin registration
messages and an unavailable optional TensorRT warning; the declared non-TensorRT
GPU preflight, training and persistence checks succeeded. Logs are preserved.

The cluster and local audits verify 20 consumed input arrays, metadata hashes,
matching paired features/test identities, 675 flips, output/source hashes,
all 50 epoch records and update counts, paired initial weights, and every
metric/cutoff calculation. Both the local comparison and pair-audit output
match their cluster files byte for byte. Local verification did not fit or
rescore the research data with a model.

## Interpretation and next decision

The explicit BCE repair supplies a working neural baseline in this pilot.
The default poisoned detection gap does not demonstrate unattainability of
the printed detection/FA point. The complete printed pattern, full population,
other missing choices, statistical replication, and proposed reconstruction
mechanism remain unresolved.

Stop the pair. Forest, AdaBoost and feed-forward now repeatedly show large
default-DR drops accompanied by falling FA and useful remaining ranking.
Before extending expensive model coverage, the proposed next step is a focused
review of poisoning versus balancing order and its effect on observed class
proportions. The paper leaves that order incomplete. Any resulting controlled
run must freeze its exact intervention, unchanged evaluation population,
cost and stopping rule first. We have not measured that causal effect, changed
the original preparation, or authorized an automatic search. GRU, ARIMA, AEA,
and both ensembles remain in scope; no additional model was run here.

## Preserved record

[p00](p00.json), [p30](p30.json), [p00 history](p00-history.json),
[p30 history](p30-history.json), [cluster comparison](cluster_comparison.json),
[artifact audit](artifact_audit.json), [execution](execution.json),
[runtime packages](runtime-packages.txt), [GPU-job output](slurm-400825.out),
and [original environment setup output](environment-398992.out).
Models, initial-weight arrays and probabilities remain outside Git in
data/derived/takiddin-2021-robust-poisoning/feed-forward-pilot-20260922-attempt1.

Use checks/verify_baseline_pair.py in the pinned neural environment for
read-only artifact verification, with --preparation pointing to the original
setup attempt and --attempt to this pair. Its output must reproduce
artifact_audit.json; the direct analyze_results.py command on p00/p30 must
reproduce cluster_comparison.json. No new fitting is required.
