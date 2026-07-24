# Distributed neural execution is an explicit exploratory ambiguity branch

**Date:** 2026-07-21
**Status:** Accepted pending first four-GPU end-to-end validation

The frozen exploratory global training batch is 512, but the full recurrent
models do not fit that batch on a single available 16 GB the cluster GPU. The
primary the cluster execution branch therefore uses four `v100_16GB` replicas with
PyTorch DDP and a local training batch of 128. This is a resource-preserving
execution interpretation, not a claim of bitwise identity with an ordinary
single-device Keras `Model.fit` trajectory.

The branch preserves the paper-literal model and registered training contract:

- rank 0 broadcasts one deterministic global permutation per epoch;
- every real sample appears exactly once, with no padding or dropped tail;
- each rank's mean loss is multiplied by
  `world_size * local_count / global_count`, so DDP's mean gradient equals the
  global sample-mean gradient;
- gradients are applied through the compiled Keras optimizer;
- validation loss is reduced by real sample count over every rank;
- rank 0 drives the existing Keras EarlyStopping callback and broadcasts its
  restored best trainable weights;
- deterministic rank-specific stochastic streams avoid duplicated dropout or
  VAE sampling masks while remaining reproducible for the same seed;
- rank 0 scores the complete original-order test partition with inference
  batch 128, inside the measured per-GPU memory envelope;
- partition hashes, cardinalities, source hash, runtime versions, GPU
  inventory, thread settings, Git commit, and execution policy are recorded.

The global shuffle and distributed random streams are reasonable choices for
details the paper omits, but they are not the unknown single-device private
trajectory. Any alternate interpretation must be registered separately and
cannot overwrite these attempts.

Unit tests cover sharding, sample weighting, shuffle determinism,
EarlyStopping restoration, timing arithmetic, fingerprint separation, and
immutable resume behavior. A real four-GPU end-to-end run remains the gate
before matrix execution. Catastrophic CUDA/NCCL termination may bypass Python
failure persistence; the rank-0 preflight and Slurm output remain evidence.
