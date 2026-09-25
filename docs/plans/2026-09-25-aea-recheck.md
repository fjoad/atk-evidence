# Recheck the AEA geometry, repair report and prototype

The user asked to revisit the last several steps after a model switch. This
audits existing evidence and constructed software; it does not launch another
research experiment or promote a surviving branch.

- [x] Recheck the source statements against the paper's relevant complete pages.
- [x] Verify original artifacts and independently recalculate the saved bounds,
  threshold conditions and oracle ROC/AUC limits.
- [x] Challenge the prototype's claimed tests, including query dependence,
  learning evidence, determinism and serialization on constructed inputs.
- [x] Correct reports, unsupported promotion and any justified implementation
  defects; preserve original scientific results and history.
- [x] Verify the correction and update current state; commit as the closing step.

Initial findings: the repair README uses the 5.25% FA cap for three p30 values
instead of 18.45%. MAE's free-cutoff interval excludes the printed 0.51 threshold;
selecting it as a threshold-preserving repair was unsupported. The source
specifies zero-mean/unit-variance scaling but does not explicitly name its axis.
The scoring-norm experiment also did not identify a neural training loss.

## Outcome

See the [audit record](../../studies/takiddin-2021-robust-poisoning/results/aea_recheck_20260925/README.md).
Both original artifact audits replay exactly. Independent arithmetic verifies
all ten interval archives, direct AUC bounds, cutoff endpoints and original
geometry. MAE promotion is withdrawn; three p30 report values are corrected.
The original prototype query test was non-discriminating and reload failed;
six revised constructed tests pass. Serialization changes preserve all 24
small-fixture weights, outputs and attention exactly. No new research job,
fit, branch, direct reproduction change or website work.

Verification: `scripts/test.sh` passes 392 tests with 23 environment-specific
skips (415 cases). The six isolated pinned TensorFlow AEA checks all pass;
the three new reporting regressions are included in the main suite. Strict
data verification succeeds on the recorded ScienceDB semantic-equivalence
branch; restricted official CER copies are absent locally, not substituted.
The original scientific result JSON/archives/contracts remain preserved.
