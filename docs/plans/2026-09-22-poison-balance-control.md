# Poisoning versus balancing: one controlled comparison

**Date:** 2026-09-22. **State:** completed and audited on September 23.

September 23 resume: VPN and authenticated Panther access are restored. No
active account jobs or prior order-control attempt directories were found.
The original preparation and saved B references are present. Resume the
unchanged scientific freeze below; ten local fixtures pass again. No new
scientific result is available at this checkpoint.

Submitted once as CPU job **402290** from frozen `9b23b32`, with output
`poison-balance-20260923-attempt1` and isolated worktree
`/export/home/fjoad/atk-evidence-paper3-order-20260923`. The main Panther
checkout and earlier attempts are untouched. The existing runtime is reused.

Job 402290 failed 1:0 in 57 seconds after all ten fixtures passed. The version
preflight accessed `globals()['numpy']` although the module is imported as
`np`, raising KeyError before output-directory creation, input loading,
preparation or fitting. There are zero research fits from this attempt.
Preserve its job log. Repair the version-module lookup only, add preflight
regression tests, and freeze the repair before one resumed allocation. Keep
the 9b23b32 contract, seed, sampler, forest and analysis unchanged. Use a new
`attempt2` output identity despite the first directory never being created.
The resumed allocation is capped at **14:03**, so its maximum plus the first
job's 57 seconds stays inside the original 15-minute cumulative wall budget.
This records two scheduler submissions for one planned research fit, not an
unreported repeat of a model experiment.
The startup-only repair is frozen at
`579a3f5cafc0f8943f230ff7458a1c14103a6718`; all 12 targeted fixtures pass.
The original contract at `9b23b32` remains unchanged, including all scientific
functions outside the version lookup. Execution now uses the repaired freeze.

Completed as job **402291**, 0:0 in 20 seconds; exactly one research fit.
The 28-array zero-poison guard, 12 fixtures, model reload and all artifact
audits pass. Local/cluster audit files are byte-identical. On 1,288 original
test rows, B-p30→D-p30 DR is 63.89→63.17, FA 8.74→12.57, AUC 84.38→79.80;
common-cap detection worsens too. More balanced observed labels did not
produce the proposed rescue. The contrast does not isolate class proportions.
Full repaired suite: 339 pass/10 environment-specific skips. Stop the control.
See the [result](../../studies/takiddin-2021-robust-poisoning/results/poison_balance_20260923/README.md).

Budget correction: although the resumed submission requested 14:03, Slurm
recorded 15:00. Actual combined wall time was 57+20=77 seconds, below the
15-minute budget, but the intended cumulative hard limit was not enforced.
Both scheduler submissions and the failed startup are preserved.

The user asked to continue after the feed-forward result. Review the source,
then freeze the smallest check of the next named shared-setup question (E19).

- [x] Re-read and visually inspect pp. 2677-2678. Separate the printed
  pre-split balancing from the incompletely placed training-label corruption.
- [x] Record the intervention, competing predictions, fixed evaluation,
  model, seed, budget and stopping in `POISON_BALANCE_CHECK.md`.
- [x] Implement one direct check without changing the original preparation,
  model, saved inputs or prior results; validate only constructed fixtures locally.
- [x] Freeze the tested implementation and update the journal/site/handoff.
- [x] Once Panther is accessible, run at most one new CPU forest fit and a
  zero-poison preparation-parity guard; audit all saved artifacts and report.

At the September 22 first check, QCRI VPN was disconnected and hostname
resolution failed; no experimental job had been submitted. Access returned
on September 23 as recorded above. No local research-data preparation/fitting
was substituted. No new seed, model family, full-data run or publication is
part of this step.

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

## Preserved resume procedure (now completed; not authorization to repeat)

Original scientific contract/code freeze: `9b23b329aaefe76c9a1559691366ecdb424acd3f`.
The documented version-lookup repair at `579a3f5` supplied the successful
execution revision; no scientific setting changed. Later documentation
commits do not replace that execution revision or authorize another fit.

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
