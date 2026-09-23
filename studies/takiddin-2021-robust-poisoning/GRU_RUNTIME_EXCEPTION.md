# Authorized GRU runtime-gate exception

Before any GRU research fit, the user answered “sure incresae it and keep going”
to the question whether to proceed with the unchanged model under the existing
40-minute job limit and 15-minute per-fit guards despite the roughly 13-minute
projection exceeding the conservative 12-minute launch gate.

This authorizes a **900-second launch ceiling**, not a new model or a longer
training/job limit. The original 720-second gate remains failed and preserved.
No research performance was available when this exception was approved.

This exception applies only to preflight job 402376, scientific revision
`2d706b103ee03cc705cc3ef3f07718bc3ed7792a`, result SHA256
`238384ed7e52f1438a9bf66202e850b1afa3b18215019dc8e284021633a3a040`.
Its slowest-warmed-step projection is 778.6819897592068 seconds. Historical
source hashes and current scientific file hashes must both verify; only the
launch authorization/checks and research records may change. GRU_PILOT.md,
the three computational model/run/analysis files and pinned requirements remain
byte-identical to that scientific freeze.

Run exactly the original p00/p30 pair: seed 20260920, 50 epochs, batch 100,
one V100-16GB, four CPUs, 16 GiB, one 40-minute allocation, 900-second
batch-boundary fit guards. No repeated preflight, extra seeds, altered
activation, tensor layout, optimizer or epoch budget. Save the authorization
alongside the new output. Preserve partial fits and stop on failure; never
describe partial training as a completed 50-epoch result.

After execution, audit the saved histories, initial weights, models, scores,
inputs and metrics. Record the outcome and next named scientific question;
do not launch another model or edit/publish the website in this step.
