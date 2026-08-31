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
2. **IN PROGRESS — Pilot.** One small CPU pilot, checking data identity, finite
   updates, head construction, calibration isolation, and the measured budget.
3. **PENDING — Single bounded pair.** Only if promoted: 2,048 benign fitting
   rows, 1,024 benign calibration rows, 1,024 held-out source days plus attack
   siblings and proportional synthetic benign rows, ten epochs, one seed.
   Score every cutoff diagnostically and retain all outcomes.
4. **PENDING — Save and discuss.** Exact fixed-score exclusion is distinct
   from failure of every possible Sigmoid configuration. Save results locally
   and stop; no push, public edits, full training, or further search.

This is a Phase-7 exploratory attainability question under the governing
clean-reader plan, not a new reproduction or mechanistic verdict.
