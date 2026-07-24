# Paper 1 neural runtime and score boundary

**Date:** 2026-07-21
**Status:** Accepted for the exploratory paper-literal run

## Context

The paper identifies Keras and Table I hyperparameters but gives no Keras or
backend version, hardware, epoch count, batch size, loss/output encoding for
the supervised networks, executable VAE reconstruction-probability formula,
or SGCC reshaping from 1,034 days to its repeatedly stated 48-neuron input.

## Decision

- Use Keras 3 with its Torch backend on the available Apple MPS runtime. Record
  this as an implementation completion, not an author fact.
- Keep the frozen primary SGCC representation at all 1,034 chronological daily
  features. Do not invent 48-day windows to make recurrent execution easier.
- Implement decoder-conditioned additive AEA attention at every output step,
  with the previous decoder state as the query. A static one-context draft was
  rejected before result runs.
- Preserve the Table I output activations even when they conflict with z-scored
  inputs.
- Report VAE reconstruction MSE and MSE-plus-KL only as surrogate branches.
  Neither is described as the paper's undefined reconstruction probability,
  and the printed thresholds are dimensionally assumption-bound on them.
- Use the documented supervised output/loss completions in the contract.
- Measure or bound resource feasibility before full recurrent runs. A resource
  failure is preserved as a result; sequence length, widths, and outcomes are
  never silently reduced.

## Consequences

The fully connected models are directly executable. Recurrent SGCC runs are
backend- and resource-sensitive because the paper's nonstandard activations
disable common fused LSTM paths. Decoder-conditioned AEA attention has
quadratic sequence storage; at 1,034 steps its broadcast alignment tensor is
not feasible at the frozen primary batch size of 512 on this host. Batch 32 is
reported only as the predeclared sensitivity branch, not substituted into the
primary result.
