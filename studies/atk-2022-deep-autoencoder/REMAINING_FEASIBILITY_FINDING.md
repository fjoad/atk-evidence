# Remaining-model feasibility finding

**Date:** 2026-09-01

**Evidence type:** operational `X`; no numerical (`N`), mechanism (`M`), or
attainability (`A`) conclusion about the paper.

## Plain-language question

Before spending days on a full reproduction, we asked whether each remaining
model could do the basic job reliably: build, update its weights for two
epochs, save and reload exactly, produce finite scores that do not materially
change with inference batch size, stay within memory, and project to no more
than 72 hours for the frozen full anchor.

One model passed. Three stopped at predeclared gates.

| Job | Model | Slurm outcome | Feasibility outcome |
|---:|---|---|---|
| 385552 | LSTM-SAE | failed `1:0`, 7:44 | score-batch agreement failed |
| 385553 | FC-VAE | completed `0:0`, 1:16 | all gates passed |
| 385554 | LSTM-VAE | failed `1:0`, 4:49 | score-batch agreement failed |
| 385555 | LSTM-AEA | failed `3:0`, 34:11 | full-anchor time projection failed |

The jobs consumed 2,880 GPU-seconds in total, or 0.8 GPU-hours. Every attempt
used commit `052ac373b77786ad58829b0ffe35568e971bb92d`, the same 32,768 fit
rows, the same 12,119 score rows, seed 20260824, batch size 32, two epochs, and
the frozen `remaining-paper-feasibility-v1` contract.

## What worked

All four models crossed the repaired input gate, built, completed both epochs,
changed their weights, and retained finite losses:

| Model | Epoch 1 loss | Epoch 2 loss | Completed updates |
|---|---:|---:|---:|
| LSTM-SAE | 1.5899 | 1.4972 | 2,048 |
| FC-VAE | 90.4320 | 90.3201 | 2,048 |
| LSTM-VAE | 85.3018 | 68.5680 | 2,048 |
| LSTM-AEA | 1.8107 | 1.8033 | 2,048 |

The transferred records, weights, selections, scores, and logs match their
recorded SHA-256 values. Every saved numeric array has the declared shape and
contains only finite values. The large weights and score arrays remain in the
ignored local derived-data tree and the Panther derived-data tree; their exact
hashes are recorded in
`results/remaining_pilot_feasibility_20260901.json`.

## Why the recurrent SAE and VAE stopped

The gate scores the same rows with two safe inference batch sizes and requires
every preserved score to agree within an absolute tolerance of `1e-6`.

- LSTM-SAE's deterministic MSE differed by
  `1.8380262108763645e-05`.
- LSTM-VAE's primary kernel score differed by only
  `1.1779597763883487e-07`, but its auxiliary deterministic MSE-plus-KL score
  differed by `1.966813361342634e-06`.

These are small floating-point discrepancies, not NaNs, divergent training,
or failed model construction. Under the frozen all-score gate they are still
failures. Neither model may be promoted unless we first discuss and predeclare
whether the absolute tolerance, the set of scores it governs, or a
decision-level invariance check is the scientifically relevant requirement.
No tolerance was changed after seeing the result.

## Why the attention model stopped

LSTM-AEA passed every gate except projected runtime. Its two pilot epochs took
`1964.9483` seconds. The frozen conservative projection estimates:

- `193.2628` hours for the minimum ten-epoch full-data run; and
- `1879.9254` hours for the 100-epoch full-data anchor.

That is about 78 days on the tested single NVIDIA A16, not a borderline miss
of the 72-hour cap. The estimate deliberately scales the slowest measured
pilot epoch to the full fit population and applies the frozen 1.5 safety
factor. It is a budget projection, not a measured full run and not a statement
that every possible implementation or hardware configuration must take that
long.

## FC-VAE result and boundary

FC-VAE passed all eight gates. It completed 2,048 updates, saved and reloaded
the same fitted weights, produced finite scores for all 12,119 rows, and its
largest two-batch discrepancy was `8.881387403292251e-09`. Peak resident memory
was 2.278% of the 96-GiB allocation and peak reserved GPU memory was 0.591% of
the visible GPU. Its conservative 100-epoch full-anchor projection is
`44.8106` hours, below the 72-hour limit.

This makes FC-VAE eligible for discussion about promotion. Its two-epoch
scores are not a reproduction result and must not be compared with the paper's
headline table as if they were a fully trained model.

## Manifest correction discovered after the repair

The first failed wave stopped at the first omitted entry,
`table_iv_order.npy`. Once the repaired resolver examined the complete
required set, it showed that the prepared metadata also omits
`test_attack_id.npy` and `test_source_row.npy`. For all three files, the
expected checksum came from the same exact committed eligible-anchor result
after verifying both that record's SHA-256 and its exact prepared-metadata
identity. The actual input bytes then matched those expected values.

Prepared metadata remained primary for `x_train.npy`, `x_test.npy`, and
`y_test.npy`. This is additional detail about the same approved fallback rule,
not a change to data or model semantics.

## Decision boundary

Stop for discussion. Do not launch FC-VAE's full anchor, relax either
score-consistency gate, optimize or redistribute AEA, publish pilot metrics, or
start another model until the next action is named and approved. The failed
attempts and the single passing attempt remain operational evidence only.
