# the cluster CUDA compatibility

**Date:** 2026-07-21

the cluster's observed NVIDIA driver is `570.133.07`. The previously resolved
PyTorch 2.13 Linux package uses CUDA 13, whose documented minimum driver is
580.65.06. That environment cannot be the the cluster execution environment.
See the official [CUDA 13 release
notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)
and [PyTorch installation
matrix](https://pytorch.org/get-started/previous-versions/).

Pin PyTorch 2.7.1, whose official Linux wheel is available for CUDA 12.6. This
changes only the unspecified software environment, not the paper-described
data, model, batch size, optimizer, threshold, seed, or stopping rule. the cluster
runs record their actual package and CUDA versions and remain distinguishable
from earlier local runs.

Each neural job must fail before training unless Keras selected the Torch
backend and PyTorch can see its allocated GPU. No silent CPU fallback is
allowed.
