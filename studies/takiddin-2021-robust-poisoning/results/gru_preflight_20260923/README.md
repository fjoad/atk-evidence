# GRU constructed preflight: software passes, launch gate held

**Date:** 2026-09-23. **Scope:** constructed software tests and GPU timing
only. No research data were loaded, no research model was fitted, and no new
paper-performance metric was measured.

The [GRU contract](../../GRU_PILOT.md), model, runner and tests were frozen at
`2d706b103ee03cc705cc3ef3f07718bc3ed7792a`. The native-Keras/table completion
uses eight 300-unit GRU layers over 48 one-reading time steps, ReLU candidates,
sigmoid gates, reset-before multiplication, input dropout .2, MaxNorm 5 and a
two-unit Softmax with standard classification-loss repair. It contains
4,058,702 parameters. Source ambiguities and untested alternatives remain in
the contract; this does not identify the authors' implementation.

## Completed checks

All 13 GRU fixtures pass locally and on the allocated V100, including a full-
architecture positive learning fixture, hand-calculated reset-before recurrence,
correct/reversed constructed-label learning, output normalization, constraints,
paired initialization, saved-model reload and partial-history handling.
All eight prior feed-forward neural fixtures and seven isolated AdaBoost
fixtures also pass locally; the old FF artifact audit remains byte-identical.
Main-environment checks total 352 passes and 17 environment-specific skips.

The timing program uses one `model.fit` call on a repeated constructed dataset,
matching the research runner's API. A local exploratory `train_on_batch`
loop produced retracing warnings; that was corrected in the timing instrument
before code freeze or GPU timing, not after a research-data outcome.

## Timing and hardware

Job **402376** completed 0:0 in **2 minutes 33 seconds**, within its ten-minute
allocation: one Tesla V100-PCIE-16GB, four CPUs, 16 GiB host memory, node
`crimv3mgpu005`. The tests took 105.343 seconds; the separate timing program
took 26.472 seconds. TensorFlow 2.16.2/Keras 3.4.1, float32, deterministic
operations, no XLA/TF32/mixed precision, backend-native GRU rather than cuDNN.
Trainable weights and the inference output are on GPU:0.

The fixed batch contains 100 constructed 48-step sequences, generated with
seed 20260923 and alternating binary labels. It is not a proxy CER dataset.
Two warmup steps precede ten measured updates; all 12 losses are finite and
weights change. The first traced step takes 11.807 seconds, the second 0.358.
The ten warmed steps range from **0.33034 to 0.34608 seconds**, mean 0.33672.
One eager inference batch takes 2.118 seconds; this is a separate measurement.

The frozen gate uses the slowest warmed step:

```
2,250 updates × 0.3460808843 seconds = 778.68199 seconds
                                     ≈ 12 minutes 59 seconds
launch threshold                     = 720 seconds (12 minutes)
```

The gate therefore **does not pass**. It is a conservative pilot-cost rule,
not a reproduction criterion or a mathematical limit. This constructed timing
does not establish whether the paper's full-population runtime is feasible,
or whether the model's detection results are attainable. It does not justify
switching the activation or tensor shape to make the timing favorable.

TensorFlow reports peak allocator use of 855,042,560 bytes, not whole-process
GPU memory. The timing process peak host RSS is 1,230,216 KiB; Slurm's maximum
sample across the job, which also includes fixtures, is 1,442,064 KiB.
Driver 570.133.07, GPU memory 16,384 MiB, compute capability 7.0. Startup
duplicate-plugin and optional TensorRT warnings are retained in the job log;
the declared native-GPU updates succeeded.

## Verification and decision

The transferred preflight SHA matches Panther. All six source-file hashes,
code revision, parameter/update counts and gate arithmetic verify. Local
read-only verification confirms the 17 kernel norm summaries, GPU placement
and changed weight hashes. The frozen verifier **correctly refuses to authorize
research-data loading** because the recorded gate is false.

No research pair was submitted. The user has been asked whether to permit
the unchanged pair under the already specified 40-minute job limit and
15-minute per-fit guards despite this small launch-gate miss. That is a
requested explicit exception, not a passing gate or silent contract revision.
Until that decision, preserve the checkpoint and do not launch fits.

Website work remains with the other session; no page or research journal was
regenerated or published by this step. Earlier scientific artifacts are intact.

## Files

[Preflight result](preflight.json), [read-only audit](artifact_audit.json),
[execution record](execution.json), [package inventory](runtime-packages.txt),
and [complete job output](slurm-402376.out).
Preflight SHA256:
`238384ed7e52f1438a9bf66202e850b1afa3b18215019dc8e284021633a3a040`.
