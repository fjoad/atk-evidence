# Execute the frozen zero-fit AEA geometry check

The user approved the next action after source specification: implement and
run AEA_GEOMETRY_CHECK.md. Its scientific contract is frozen at aa2c4ab.
No neural model, new preparation, altered interpretation, or website work.

- [x] Implement the direct diagnostic and read-only artifact audit; test them
  on constructed inputs, keeping existing data and implementations unchanged.
- [x] Freeze executable code. Check Panther scheduler/output state and submit
  one10-minute/1CPU/8GiB/noGPU job on the preserved novelty p00/p30 cases.
- [x] Copy and independently audit saved inputs/identities/bounds/scores and
  metrics. Preserve every outcome, including failure or inconclusive bounds.
- [x] Record the bounded finding, changed conclusions and next question;
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

Execution frozen at `5d5d85050850f21f344c7509267b8215ca18767d`; submitted
once as job402811 after confirming no account jobs or previous output.
Checkout `/export/home/fjoad/atk-evidence-paper3-aea-geometry-20260924`;
output `aea-geometry-20260924-attempt1` under the usual derived-study root;
log `/export/home/fjoad/aea-geometry-transfer-ufeqQr/slurm-402811.out`.
Do not resubmit; inspect this allocation/output first.

Completed0:0 in30s on crimv3srv025. The one requested CPU task used one
numerical thread; Slurm allocated two logical CPUs. All18 fixtures pass on
the compute node. Both score archives and56 input arrays/provenance/metrics
pass; local audit matches cluster bytes. All original files remain intact.

Rounding-favorable MSE minimum FA22.15909/22.36842 exceeds5.25/18.45,
so the printed operating corners are excluded on these fixed inputs for any
[0,1] reconstruction at the stated threshold. RMSE/SSE also fail; the
optimistic DR bound stays100%. This is not an AUC ceiling, another scaler/
population/threshold/output-range or ensemble result. No trained AEA exists.
Stop; next proposed work is a bounded score/scale clarification or repair
comparison, separately specified/approved. No extra fit, seed or publication.

Final verification:24 focused diagnostic/geometry/source-audit tests pass;
pre-freeze main suite378 passes/17 environment-specific skips and strict data
verification passed. Input/score/result transfer hashes and read-only local
artifact audit match the cluster. No scientific implementation or original
contract changed after the frozen run.
