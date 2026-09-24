# Execute the frozen zero-fit AEA geometry check

The user approved the next action after source specification: implement and
run AEA_GEOMETRY_CHECK.md. Its scientific contract is frozen at aa2c4ab.
No neural model, new preparation, altered interpretation, or website work.

- [x] Implement the direct diagnostic and read-only artifact audit; test them
  on constructed inputs, keeping existing data and implementations unchanged.
- [ ] Freeze executable code. Check Panther scheduler/output state and submit
  one10-minute/1CPU/8GiB/noGPU job on the preserved novelty p00/p30 cases.
- [ ] Copy and independently audit saved inputs/identities/bounds/scores and
  metrics. Preserve every outcome, including failure or inconclusive bounds.
- [ ] Record the bounded finding, changed conclusions and next question;
  verify and commit. Do not automatically implement/train an AEA afterward.

The check reports all predeclared score units, threshold-rounding allowances,
original/synthetic strata and zero-parameter comparisons. A bound's AUC is
not a ceiling on model AUC. No independent-row statistical intervals.

Eighteen constructed tests pass, including end-to-end persistence/audit,
incorrect scaling/labels/ancestry/overlap rejection, hash tampering, favorable
rounding and explicit null metrics for absent classes. All56 original input
arrays will be hash-checked before scoring and again in the artifact audit;
originals are never rewritten. The existing CPU numerical environment and
requirements-svm.txt are reused, with no package installation. Research-data
calculation is gated by Slurm; local tests contain only constructed rows.

Pre-freeze verification passes:378 main-suite tests with17 environment-specific
skips,18 focused geometry/diagnostic fixtures, and strict source-data verification.
The original contract, METHOD.md and five reproduction files are unchanged.
