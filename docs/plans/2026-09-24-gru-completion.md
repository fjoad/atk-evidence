# Finish the unchanged GRU schedule with a measured budget

The user approved correcting the runtime estimate and budgeting the unchanged
50-epoch experiment after the preserved partial fit. No website work.

- [x] Check saved epoch timing and compare constructed/research input signatures.
- [x] Freeze the revised operational budget and same-seed restart in
  GRU_COMPLETION.md; expose only the fit guard, not model settings, in the CLI.
- [x] Verify guard dispatch, unchanged scientific functions, historical audits
  and neural fixtures; freeze code before the new attempt.
- [x] Check scheduler/output state; launch one original p00/p30 pair, one
  V100-16GB, four CPUs/16 GiB, 35-minute fit guards, 85-minute allocation.
- [x] Preserve/audit all outcomes, compare the unpoisoned run's initial weights
  and first 32 epochs with the earlier attempt, and record the next decision.

No additional constructed GPU timing job is needed: 32 measured full research
epochs directly inform the budget. The fixture checks only tensor signatures;
no local research-data inference/training or claimed causal timing test.
The prior fixed batch has TensorSpec(100,48,1), whereas ordinary batch(100)
on the 4,464-row fixture has TensorSpec(None,48,1), 45 batches, last batch64.
The pipeline difference is verified, its causal runtime contribution is not.

Mean full-epoch time27.990782s implies1399.539119s for50; assigning the
slowest full epoch39.869955s to every epoch implies1993.497735s. Round that
conservative guide up to2100s per fit. Allow another900s overall for fixtures,
loading, scoring, persistence and audits: 2*2100+900=5100s. These are resource
guards, not guarantees or the paper's full-population hardware/time claim.

Checks: all 21 GRU and eight feed-forward fixtures pass after correcting one
test's string-versus-Path expectation (no scientific-code correction). The
initial fixture failure is retained in the local test log. Main suite:
360 passes and17 environment-specific skips. Strict source-data verification
passes. Original partial-GRU and completed feed-forward read-only audits
still match their saved files byte-for-byte. The new prerequisite check
verifies prior hashes and unchanged science and derives2100/5100s from the
actual history. Before launch, scheduler was empty and only the original
GRU preflight/partial output directories existed. No website files changed.

Frozen at `1f847039ea3c5d410db5e9ee194a6625ecd2b54d`. Submitted once as
job402625, running on crimv3mgpu005 with the exact85-minute cap, one
V100-16GB/four CPUs/16 GiB. Output `gru-completion-20260924-attempt1`.
See `results/gru_completion_20260924` for paths. Check it before any further
submission; the wrapper automatically handles both conditions and audits.

Cluster prerequisite/authorization bytes match the local check. All21 GRU
fixtures pass on GPU in101.495s; the p00 research runner has started. Job
remains running, with no completed result yet. Do not launch a duplicate.

At the running checkpoint, p00's first two full epochs/90 updates exactly
match the earlier loss/accuracy values, and the initial-weight archive hash
matches the old file. The full32-epoch prefix comparison remains pending.

## Completion check

The user asked to check the existing job. Slurm reports COMPLETED/0:0 in
56:11. Both conditions finished; cluster comparison/pair-audit and prefix
records exist. The first32 unpoisoned epoch metrics match exactly. Copy full
artifacts to ignored data, re-run saved-artifact audits locally (no model
inference), compare bytes with cluster outputs, inspect learning/timing and
fixed-score tradeoffs, then record the bounded result and next question.
No new fit, seed, allocation, website edit or publication is authorized by
this status check.

All local pair, comparison and prefix audits match cluster bytes. All20
consumed inputs, labels/identities, source/output hashes,50epochs/2250updates
and paired initialization pass. Read-only serialized-model inspection also
checks final-weight hashes, optimizer count, configuration and constraints.
The first32 p00 epochs match the partial attempt exactly; old outputs remain
intact. No local model inference or new allocation.

Completed DR/FA/AUC: p00 62.35294/9.93789/89.07390; p30
39.72851/6.12245/79.88605. At corresponding paper FA caps6.8/20.6%, best
DR51.40271/67.14932 versus92.4/78.5. Reversal does not rescue these scores.
The poisoned AUC is close to79.4 reported; do not call that full reproduction.
P00 loss rises sharply at epochs39–40 then partly recovers; no stable plateau
or global attainability limit. Stop the pair. Proposed next coverage step is
source-specifying standalone AEA/reconstruction before fitting or ensemble
work, not more GRU seeds/epochs. No further experiment or publication occurred.

Outcome-check verification: main suite360 passes/17 environment-specific
skips; strict data verification passes. All journal/site consistency checks
pass without regenerating any page. Earlier partial audit still matches bytes.
