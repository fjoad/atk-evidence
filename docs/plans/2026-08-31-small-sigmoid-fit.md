# Test the remaining Sigmoid alternative without assuming the answer

Date: 2026-08-31

User request: test the claim that even Sigmoid with a changed cutoff fails.
The investigation is authorized; the desired conclusion is not assumed.
The prior quick-check constraint remains: no full-data retraining or sweep.

1. **COMPLETE — Freeze and verify.** Follow
   [SIGMOID_FIT_CHECK.md](../../studies/atk-2022-deep-autoencoder/SIGMOID_FIT_CHECK.md).
   Build a small paired control from the unchanged FC-SAE architecture, with
   identical starting weights and only the final activation differing.
   All 230 deterministic tests passed, including five new hand-sized checks.
2. **COMPLETE — Pilot.** Frozen code `cc9af5e`, CPU job `385198`: 9.2163-second
   pilot passed data identity, finite updates, head construction, and identical
   initial weights. The predeclared estimate was 62.2256 seconds; scaling
   non-batch overhead by 160 for a conservative check gave 92.0258 seconds,
   also below the 240-second promotion cap. Promotion did not use accuracy.
3. **COMPLETE — Single bounded pair.** Promoted: 2,048 benign fitting
   rows, 1,024 benign calibration rows, 1,024 held-out source days plus attack
   siblings and proportional synthetic benign rows, ten epochs, one seed.
   Both heads completed ten epochs / 640 updates. Small analysis took 24.8064
   seconds; job `385198` completed `0:0`, total allocation 3:52. Sigmoid max DR
   at FA<=15% was 9.74935% (25.39063% reversed), versus 81%. All cutoffs and
   the relaxed target fail for this fit/sample; all outcomes are retained.
4. **SAVED — Stop for discussion.** See
   [SIGMOID_FIT_FINDING.md](../../studies/atk-2022-deep-autoencoder/SIGMOID_FIT_FINDING.md).
   Exact fixed-score exclusion is distinct
   from failure of every possible Sigmoid configuration. Save results locally
   and stop; no push, public edits, full training, or further search.
   Sigmoid's best calibration checkpoint was the last epoch and loss still
   improved; no long-run plateau or universal failure is inferred.
   Post-run verification: 232 tests and 73 local links passed; public and
   reproduction files unchanged. Commit records locally only, then discuss.

This is a Phase-7 exploratory attainability question under the governing
clean-reader plan, not a new reproduction or mechanistic verdict.
