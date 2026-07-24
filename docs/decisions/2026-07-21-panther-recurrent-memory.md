# the cluster recurrent-model memory

**Date:** 2026-07-21
**Status:** Superseded later on 2026-07-21 by the 16-GB-only execution policy

The first paper-literal LSTM-SAE and LSTM-VAE attempts used one 16 GB T4 each.
Both passed CUDA visibility and exact-data checks, then failed while building
or fitting the frozen batch of 512 full 1,034-reading sequences. Their job logs
and immutable failure manifests are retained.

The initial response was to run the unchanged models and batch on one 32 GB
V100 per model/seed. One such LSTM-SAE attempt also OOMed, and an LSTM-VAE
attempt was cancelled after 20:38 when the user clarified that 32 GB GPUs are
not a dependable project resource.

Current policy is recorded in
`2026-07-21-cluster-16gb-data-parallelism.md`. This superseded decision remains
here so the resource and reasoning history is not silently rewritten.
