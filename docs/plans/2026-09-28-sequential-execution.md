# Execute the bounded ensemble sequence

Date: 2026-09-28. State: approved, starting with full-width software learning.

The user asked to "go ahead and do it all" after the four-stage proposal:
learning validation at the published widths, full-model GPU timing, the
fixed original p00/p30 ensemble pair, then evaluation/audit. This authorizes
those dependent stages when their gates pass. It supersedes earlier wording
that the proposed sequence was awaiting approval; it does not waive failed
checks or authorize parameter/seed searches.

## Stage 1: width-only constructed comparison

The named question is whether reducing layer widths caused an unrepresentative
learning failure. Change only the layer widths from the old fixture to the
published500/300/200 mirrored LSTM, eight300-unit GRUs and Dense500. Retain
the same32 training/32 test synthetic arrays,8 time steps, model seed20260920,
normal/reversed labels, Adam.001, activations, constraints, full-batch training
and300 updates. Model parameter count must be9,240,802. The eight-step
sequence is deliberately retained to isolate width; this is not yet a
48-step training demonstration or research-data experiment.

Run the two fixed label orientations once on the local pinned CPU runtime,
as software fixtures only. Allow600 seconds per fit at batch boundaries,
prospectively larger than the small fixture's300s because the model grows
from5,082 to9,240,802 parameters. Maximum scheduled fitting time20 CPU-minutes,
plus bounded persistence/audit overhead; no extension based on outcomes.
Preserve partial/failure states, histories, initial/final weights, probabilities,
gradient summaries, constraints and reload comparisons. Both orientations
must finish300 updates, achieve at least90% held-out synthetic accuracy and
BCE<log(2)/2, with identical initial weights and passed artifact/software
checks. Do not retry seeds/settings or select a favorable checkpoint.

The old narrow-model failure remains unchanged. A passing wide pair would
qualify this declared full-width implementation for timing; it does not
retroactively pass the narrow fixture or prove temporal/poisoning capability.
A failed/partial wide pair stops dependent stages for diagnosis and is not
paper evidence. It does not select another optimizer/activation automatically.

## Stage 2: full-sequence GPU timing, only after Stage 1 passes

Use the existing20-minute V100-16GB/4CPU/16GiB constructed preflight contract
in SEQUENTIAL_ENSEMBLE_PILOT.md. Full widths and48 time steps,4,464 synthetic
rows, three epochs with batch100 plus the64-row remainder. Confirm finite
updates, correct device, persistence and memory. Require50 times the slower
of warmed epochs2/3 <=3,600 seconds. Check scheduler/output state before
submitting once. No research inputs in this allocation. Preserve a failure;
do not silently substitute faster activations, hardware or sequence layout.

## Stage 3: real-data pilot, only after all gates pass

Implement the direct runner and audit path from the frozen source
specification, verify/freeze code, then use the existing p00/p30 classifier
arrays and hashes. One fresh paired seed20260920,50 epochs,batch100, final
weights, joint corrected BCE. One180-minute V100 allocation with70-minute
fit guards; stop before p30 if p00 fails or remains partial. Original inputs,
models and historical source snapshots stay intact. Experimental operations
run on compute nodes only. No dataset regeneration, extra seed, selected
best epoch or full-population claim.

## Stage 4: audit and interpret

Compare all Table-V metrics, saved-score detection at common FA caps and
earlier matched-row baselines. Test-selected cutoffs are diagnostics, not
calibration. Record intermediate reconstruction behavior descriptively;
existing GRU scores are not a matched AEA-removal ablation. Save every outcome,
update current state and commit. A separate component-removal experiment,
full-population work and the statistical attainability study remain later
questions, not silently included in this bounded first ensemble pair.

Website work remains outside this task. A cluster-access problem does not
authorize local research fits; continue independent implementation/fixtures
where useful and report any genuinely external prerequisite.
