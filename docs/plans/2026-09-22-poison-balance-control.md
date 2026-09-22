# Poisoning versus balancing: one controlled comparison

**Date:** 2026-09-22. **State:** specified and locally verified; execution awaits VPN.

The user asked to continue after the feed-forward result. Review the source,
then freeze the smallest check of the next named shared-setup question (E19).

- [x] Re-read and visually inspect pp. 2677-2678. Separate the printed
  pre-split balancing from the incompletely placed training-label corruption.
- [x] Record the intervention, competing predictions, fixed evaluation,
  model, seed, budget and stopping in `POISON_BALANCE_CHECK.md`.
- [x] Implement one direct check without changing the original preparation,
  model, saved inputs or prior results; validate only constructed fixtures locally.
- [x] Freeze the tested implementation and update the journal/site/handoff.
- [ ] Once Panther is accessible, run at most one new CPU forest fit and a
  zero-poison preparation-parity guard; audit all saved artifacts and report.

The QCRI VPN is disconnected at this turn's first check; hostname resolution
fails. Do not substitute local research-data preparation/fitting. No new
experimental job has been submitted. No new seed, model family, full-data run,
or website publication is part of this step.

## Local verification

Ten new constructed fixtures pass, including full saved-artifact auditing and
rejection of invented truth accuracy. The full main-environment suite reports
337 passes / 10 isolated-environment skips (347 tests total); the targeted
ten-test suite was rerun after strengthening its end-to-end audit. Both saved
B reference records and all 56 input arrays verify unchanged. Strict data
verification succeeds through the existing ScienceDB branch; the separate
`cer-authorized` directory is absent. No new data acquisition or substitution.
Shell syntax, whitespace checks and generated-journal consistency pass. The
local journal's new-entry navigation and final layout were visually checked.

## Resume without changing the experiment

On reconnect, inspect account jobs and existing output paths before submitting.
Use the frozen scientific commit in a new isolated Panther worktree; do not
alter the Panther main checkout or any prior attempt. Main interpreter:
`/export/home/fjoad/atk-evidence/.venv/bin/python`.
Under `/export/home/fjoad/atk-evidence/data/derived/takiddin-2021-robust-poisoning/`,
use `setup-20260920-attempt1` as PREPARATION_ROOT and
`split-control-20260920-attempt1` as BASELINE_ROOT. Choose a new dated output
directory with no existing contents; never overwrite/reuse a partial attempt.
Submit `checks/run_poison_balance.sbatch` with the explicitly required variables.
No environment setup job is needed unless the pinned runtime fails verification.
Transfer and audit completed artifacts without local research-data inference,
append favorable or adverse findings, and stop before another experiment.
