# Finish the unchanged GRU schedule with a measured budget

The user approved correcting the runtime estimate and budgeting the unchanged
50-epoch experiment after the preserved partial fit. No website work.

- [x] Check saved epoch timing and compare constructed/research input signatures.
- [x] Freeze the revised operational budget and same-seed restart in
  GRU_COMPLETION.md; expose only the fit guard, not model settings, in the CLI.
- [x] Verify guard dispatch, unchanged scientific functions, historical audits
  and neural fixtures; freeze code before the new attempt.
- [ ] Check scheduler/output state; launch one original p00/p30 pair, one
  V100-16GB, four CPUs/16 GiB, 35-minute fit guards, 85-minute allocation.
- [ ] Preserve/audit all outcomes, compare the unpoisoned run's initial weights
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
