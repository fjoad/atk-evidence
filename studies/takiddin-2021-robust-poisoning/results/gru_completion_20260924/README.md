# Same-seed GRU completion — running

Job **402625** was submitted once, frozen at
`1f847039ea3c5d410db5e9ee194a6625ecd2b54d`. It is running on
crimv3mgpu005 with one V100-16GB, four CPUs and 16 GiB, an 85-minute job cap
and 35-minute per-fit guards. No completed research result is available yet.

The user approved completing the same 50-epoch question after the partial
attempt. See [GRU_COMPLETION.md](../../GRU_COMPLETION.md) and the
[step plan](../../../../docs/plans/2026-09-24-gru-completion.md).
Model, data, seed, optimizer, batch 100 and 50 epochs remain unchanged. The
scientific fit/data/device functions are syntactically identical to the old
freeze; only time-guard dispatch/metadata/stopping text changed. Original
900s defaults and partial outputs remain preserved.

The budget is derived from 32 measured full epochs, not performance:
50*39.869955s=1993.497735s, rounded to 2100s per fit, plus 900s other overhead
for 5100s overall. The constructed timing used a fixed 100 batch dimension,
whereas the research pipeline has a variable dimension and 64-row remainder.
This verified difference does not isolate the cause of the runtime gap or
test the paper's full-population RTX 2070 timing claim.

Each condition starts from seed 20260920. This is a same-seed restart, not
an exact checkpoint continuation. After p00, the script audits its outputs,
checks its starting weights against the partial attempt and records whether
the first 32 epoch losses/accuracies/update counts match (excluding timing).
Any difference is reported, not used to choose a favorable run. Then p30
and the usual pair/comparison audits run. Partial/failed execution stops
the script and is preserved. Do not submit a duplicate or expand the budget.

Local checks: 21 GRU and 8 feed-forward fixtures pass after a corrected test
path-type expectation; main suite 360 passes/17 environment-specific skips.
Strict data verification passes. Prior partial-GRU and completed feed-forward
audits remain byte-identical. Website files were not changed or published.

Cluster prerequisite verification matches local authorization bytes. All 21
GRU fixtures passed on GPU in 101.495s, and the p00 research runner has started.
The first two research epochs have completed (90 updates). Their losses and
accuracies exactly match the earlier attempt, and the initial_weights.npz
SHA256 is the same: f9be1638f70706c9580de0b358c200d676443aedc50519394bff378a02dc45d2.
The job remains running; no completed fit or new performance finding yet.

## Resume paths

- Checkout: `/export/home/fjoad/atk-evidence-paper3-gru-completion-20260924`
- Output: `/export/home/fjoad/atk-evidence/data/derived/takiddin-2021-robust-poisoning/gru-completion-20260924-attempt1`
- Log: `/export/home/fjoad/robust-gru-completion-transfer-OceGkq/slurm-402625.out`
- Original preparation: `setup-20260920-attempt1`, unchanged.
- Previous partial: `gru-pilot-20260923-attempt1`, unchanged.

Check this job/output before any new action. After it terminates, copy all
artifacts into ignored local data, preserve logs/status, audit read-only and
record the bounded outcome before another experiment. Do not perform local
research-model inference or edit the website in this research session.
