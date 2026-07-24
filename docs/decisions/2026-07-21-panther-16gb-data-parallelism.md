# the cluster 16-GB-only data parallelism

**Date:** 2026-07-21
**Status:** Accepted for the authorized exploratory run

Use only the cluster's 16 GB GPU classes for neural execution; do not depend on
the less reliably available 32 GB class. Keep the scheduler limit at no more
than three submitted jobs.

The primary global batch remains 512. the cluster reports three `T4_16GB` GPUs per
T4 node, so the rejected four-T4 request was structurally impossible. The
cluster also reports nodes with four `v100_16GB` GPUs. Use that four-way 16 GB
allocation so each replica receives a local batch of 128; do not use the
`v100nv_32GB` class.

The resource validation must show that every allocated GPU is used without
changing the declared model, loss, data shape, or global batch. Record the GPU
type/count, local batch, software backend, and timing in the run evidence.

The currently running one-T4 batch-32 job is a declared memory/runtime
sensitivity only. It cannot replace a primary batch-512 cell. All earlier OOMs
and the cancelled 32 GB attempt remain preserved as execution evidence.

Keras' Torch backend does not distribute an ordinary `fit()` call. The first
gate therefore uses a separate low-level PyTorch DDP resource probe, following
the official [Keras multi-GPU Torch
guide](https://keras.io/guides/distributed_training_with_torch/). Probe outputs
are excluded from Table II results. Production promotion requires exact
early-stopping, validation reduction, scoring, and immutable result semantics;
none is inferred merely because the resource probe fits.

The initial SAE/VAE resource probes used Torch-native losses and an optimizer
configured to the same class and defaults as the compiled Keras model. Keras
and Torch Adam are not algebraically identical, so those attempts remain
approximate memory/timing evidence. The current probe revision uses the
compiled Keras loss and applies the compiled Keras optimizer after DDP gradient
reduction. The supervised diagnostic also checks loss, gradients, parameters,
and optimizer state for non-finite values across every rank; that synchronization
slightly inflates its timing. Repeated per-rank random states still do not
reproduce a single-device stochastic stream. A production loop must separately
preserve and test stochastic behavior, stopping rule, scoring, and persistence
before its outputs can enter Table II.
