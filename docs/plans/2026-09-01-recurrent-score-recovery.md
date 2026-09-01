# Recover recurrent pilot scores without retraining

Date: 2026-09-01

User direction: run the two cheap score-only diagnostics proposed after the
remaining-model feasibility wave, then discuss the results.

1. **Complete:** freeze the exact question and decision-level outputs in
   `REMAINING_SCORE_RECOVERY.md`.
2. **Complete:** write a small direct scorer that verifies and reloads the
   preserved LSTM-SAE and LSTM-VAE weights and selections. It has no training
   path.
3. **Complete:** test comparison logic, source bindings, immutable output, and
   the pilot-only Slurm wrapper. All 251 repository tests (140 study and 111
   root) and strict-data verification pass; compilation, shell syntax, source
   hashes, and whitespace checks pass.
4. **Complete:** commit `43abc09` was pushed and synchronized to an exact clean
   detached Panther checkout.
5. **Complete:** jobs `385583` and `385584` ran exactly the two approved
   at-most-30-minute score-only recoveries and completed `0:0`.
6. **Complete:** all outputs were transferred and audited. The bounded
   operational result is in `REMAINING_SCORE_RECOVERY_FINDING.md` and
   `results/recurrent_score_recovery_20260901.json`. Stop for discussion.

FC-VAE promotion, LSTM-AEA optimization, full-data training, mechanism work,
and publication remain outside this plan.
