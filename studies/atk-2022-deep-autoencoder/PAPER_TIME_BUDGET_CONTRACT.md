# Paper-time LSTM-SAE contract

**Date:** 2026-09-01

**Status:** frozen for one execution after local verification.

**Evidence type:** time-bounded numerical/attainability evidence for one
declared completion. It is not a universal performance bound and cannot infer
author intent.

## Source claim

Table IV reports full-ISET LSTM-SAE training in 183 minutes and describes the
full training cardinality as 60 million. The paper identifies Keras Sequential
but omits hardware, GPU count, epochs, batch size, software versions, stopping
rule, timing boundary, and repetitions. The surrounding prose says training is
offline. Its separate claim that online testing takes one to two seconds per
decision is not treated as the time for scoring the full prepared test array.

The most conservative ordinary reading is that 183 minutes covers one final
model fit after hyperparameter selection; preprocessing, sequential search,
and complete audit scoring are outside that clock. Including those stages
would only reduce the fitting budget.

## Hardware inference

The unpublished device remains unknown. Contemporaneous primary records bound
a plausible institutional range:

- Texas A&M's Terra system exposed K80 and V100 GPU nodes during 2020-2021.
- The Qatar/HBKU Raad2 GPU system exposes V100 GPUs. Its official getting
  started page includes a shell prompt whose account string closely matches
  the first author's name. That supports access to Raad2; it does not prove the
  paper ran there.
- NVIDIA rates one A16 GPU at 4.5 TFLOPS FP32 and 200 GB/s memory bandwidth.
  NVIDIA rates a PCIe V100 at 14 TFLOPS FP32 and 900 GB/s. Therefore the A16 is
  newer and stronger than one K80 GPU in compute capability, but it is not a
  conservative substitute for a V100.

Use exactly one V100-16GB for the time-bounded test. This favors the strongest
specific historical clue without adding GPU count or using post-period H200 or
A100 hardware. Hardware specifications do not predict exact Keras LSTM
throughput; the actual run records measured wall time.

Primary sources:

- paper DOI: https://doi.org/10.1109/JSYST.2021.3136683
- Texas A&M 2020 HPRC overview:
  https://hprc.tamu.edu/files/training/2020/Spring/Intro_to_HPRC_clusters_2020_spring.pdf
- Raad2 systems: https://rccg.hbku.edu.qa/wiki/Systems
- Raad2 GPU getting started: https://rccl.hbku.edu.qa/wiki/Raad2-gpu/Getting_Started
- NVIDIA A16: https://images.nvidia.com/content/Solutions/data-center/vgpu-a16-datasheet.pdf
- NVIDIA V100: https://www.nvidia.com/en-gb/data-center/tesla-v100/

## Frozen scientific fields

- model: `CR-ISET-LSTMSAE-01` from `REMAINING_PAPER_CONTRACT.md`;
- exact prepared cache: 1,500,523 benign fitting profiles and 8,884,989 test
  profiles, with all existing hashes and provenance unchanged;
- seed: 20260824 and the existing initialization, shuffling, and scoring seed
  streams;
- architecture: `(500,300)` LSTM encoder, `(300,500)` decoder, first-step
  latent followed by zeros, top-state-only initialization, Sigmoid hidden and
  output activations, dropout 0.2;
- Adam at Keras default `1e-3`, MSE, batch 32, seeded epoch shuffling;
- maximum 100 epochs, but stop at the first completed batch whose elapsed fit
  time reaches 10,980 seconds (183 minutes); and
- no validation, learning-rate search, retry, additional seed, additional GPU,
  mixed precision, distribution, compilation optimization, or outcome-driven
  change.

The batch-boundary callback may exceed 10,980 seconds by the duration of the
single already-started batch. Record that overshoot. This maximally uses the
declared budget while preventing another batch from starting afterward.

## Preservation and evaluation

Save the initialized-weight digest, final time-capped weights, complete epoch
history, partial-epoch cumulative objective where available, update count,
wall time, device, versions, memory, configuration, warnings, failures, and
artifact hashes. Reload saved weights in a new model before scoring.

Score every prepared test row at the first safe predeclared batch in
`256,128,64,32`; scoring occurs after the 183-minute training clock. Save
row-aligned MSE. At cutoff 0.61 calculate DR, FA, SP, PR, balanced ACC, F1, and
AUC. Enumerate every distinct score boundary in the paper direction and the
reversed diagnostic, including maximum DR at FA<=13%, 13.5%, 15%, and 15.5%,
best balanced accuracy, and the smallest complete seven-metric gap to the
reported LSTM-SAE Table-III row.

## Decision language

If the target is not recovered, report:

> The reported LSTM-SAE result did not arise from this declared implementation
> within the paper's reported 183-minute training budget on one V100.

If every cutoff also misses the target, add that no cutoff rescues this fixed
score vector. Do not say that every undocumented implementation is impossible.
The exact runtime statement already available for the A16 is narrower: its
measured projection cannot complete the predeclared ten-epoch anchor within
183 minutes.
