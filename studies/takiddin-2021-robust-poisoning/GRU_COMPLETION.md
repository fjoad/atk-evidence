# Same GRU, complete schedule, revised operational budget

The user approved proceeding after the audited partial result, specifically
to correct the runtime estimate and budget the unchanged 50-epoch experiment.
This is a same-seed completion attempt, not a new hyperparameter, seed search,
or confirmatory experiment. The prior partial result remains preserved.

## Fixed science and initialization

All scientific settings in GRU_PILOT.md remain in force: original p00/p30
preparations, 8x300 native ReLU GRU, 48x1 inputs, sigmoid gates, reset-before,
input dropout .2, MaxNorm5, two-class Softmax and repaired cross-entropy,
Adam .001, seed20260920, 50 epochs, batch100, final weights and unchanged
metrics. No new data preparation, correction, tuning or selection by results.

Start each condition fresh from the original seed and initialization. Do not
call loading the partial weights an exact continuation: the full data-shuffle
and dropout-generator state is not established as restorable. The new p00
must have the earlier initial-weight hash; both new conditions must match.
After p00, compare the first32 completed epoch losses/accuracies/update counts
with the earlier trajectory, excluding timing. Record exact agreement or
disagreement; a difference cannot be hidden or used to select a better run.
This repeats earlier updates and counts their cost. It does not erase them.

## Timing evidence and replacement budget

Preserved prerequisite job402378/launcher46966c7 stopped at900.07785s, 32
full epochs plus seven batches. Its result SHA256 is
`fa19293a1bac847fd7fb3ab8fe37b25f3ea92f52d88ad8e53cee129bbbdd4602`;
history SHA256 `8afac3da8677604ac21f785bf01be98b91901e72128e977573c58b88899ae6b1`.
No completed result or poisoning contrast is claimed for it.

The old constructed estimate used fixed TensorSpec(100,48,1); the research
dataset uses TensorSpec(None,48,1), 45 batches/epoch and a64-row remainder.
This difference was checked using constructed local arrays only. It explains
why the preflight is not an exact matched timing instrument, not which
operation caused the timing gap. Do not optimize the model or discard the
remainder to make the estimate fit. No new GPU microbenchmark is necessary.

Budget from the32 full measured epochs only, not accuracy: mean27.990782s,
maximum39.869955s. Fifty times the maximum is1993.497735s; round upward to
the next five-minute boundary, **2100 seconds (35 minutes) per fit**.
An **85-minute job** allows both guards plus15 minutes for other work.
One V100-16GB, four CPUs,16 GiB, same pinned environment. This is a finite
operational allowance, not a full-data RTX2070 timing comparison or guarantee.

The new CLI explicitly records this guard. The original900s default, old
preflight's false720s gate and first exception remain intact. This document
supersedes those limits only for this one new p00/p30 attempt. No automatic
extra time, retry, seed, alternative setting or additional model on failure.

## Gate, artifacts and stopping

Before research inputs, validate the prior result/history/output hashes and
versions of the computational sources. Model, analysis, original contract
and requirements remain byte-identical. The GRU fit/reshape/probability/
norm/data-loading/hardware setup functions remain syntactically identical;
only CLI/dispatch, budget recording and stop-reason reporting may change.
The approval is saved separately, and GRU_COMPLETION.md joins source hashes.
Fixture tests and normal input/device checks must pass.

Run p00 then p30, exactly50 epochs/2250 updates each if the guard allows.
Stop on partial/failed execution and preserve its model/history/scores; do not
promote it to complete. Save and audit paired initialization, all histories,
input/source/output hashes, raw and normalized probabilities and GPU reload.
No local research model inference; saved-artifact audits are read-only.
After the pair, record numerical results and limits before any further model.
No website edits or publication are included.
