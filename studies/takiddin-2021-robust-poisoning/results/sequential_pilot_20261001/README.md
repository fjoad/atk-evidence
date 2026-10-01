# Sequential ensemble pilot: running record

The one authorized pair is running on Panther as **job407294**, scientific
code `3705bcca04279bbfa8523a573b9b573f123463b7`. Inspect that existing job
and output before continuing; do not submit a duplicate. P00 is complete
and audited; p30 is running. No completed poisoning comparison yet.

## Completed unpoisoned checkpoint

P00 completed50epochs/2,250updates in2125.66s of fitting, followed by207.48s
scoring and73.71s reload verification. Every test probability is exactly
0.5047591328620911, so the0.5 rule flags all2,232 rows: DR100%,FA100%,
AUC50%,ACC49.50717%. All saved-score cutoffs and favorable score reversal
give bestDR0 at the declared FA caps. This is a completed weak learner,
not merely a poor choice of cutoff. Initial test probabilities were also
constant at float32 precision. Intermediate outputs are not constant, but
mean finalMSE1.261641 exceeds zero-outputMSE1.038686; classification-only
training does not establish reconstruction of the input.

Local source/artifact/metric replay passes, as do serialized52-variable/
9,240,802-parameter/optimizer2250/finite/norm checks. The secondary archive
auditor initially rejected raw config equality because Keras stores
`shared_object_id` metadata only in the archive; recursively removing only
that identity field yields exact equality for all model settings. This was
an audit-reader correction; no model, data, score or fit was changed.
[Raw result](p00_result.json) and [history](p00_history.json) are preserved.
P30 remains the same approved job407294; no scientific conclusion about
poisoning follows until that fixed second fit completes and is audited.

## Frozen setup

- Interpretation: **I-SEQ-scalar-sigmoid**, declared in
  [the interface addendum](../../SEQUENTIAL_INTERFACE.md) before its learning
  results; all remaining settings/scope follow the
  [sequential contract](../../SEQUENTIAL_ENSEMBLE_PILOT.md).
- Original20-customer p00/p30 preparations,4,464 training/2,232 test rows;
  675 observed training labels change in p30. No regeneration.
- Fresh same-seed20260920 starts;9,240,802 parameters,48-step inputs,
  joint repaired BCE,Adam.001,50epochs,batch100,final weights only.
- One V100-16GB oncrimv3mgpu005,4CPUs/16GiB,3h allocation ceiling and
  70-minute batch-boundary guard perfit. P00 runs first; incomplete/failed
  p00 stops the wrapper beforep30. No extra seed or silent guard extension.
- Passed [preflight407255](../sequential_preflight_20261001/README.md):
  exact accepted SHA `721256059d34a104fa2eda04ccb0bb8775603d01206117e4e82e1a9e54782c07`.
  Constructed warm timing projects about30.76min per50-epoch fit.

Slurm initially held this job pending: gpu-short permits only2h. Before any
fit started, the same pending job was moved using `scontrol` to **gpu-all**,
retaining its3h ceiling and V10016GB resources. No second job was submitted
and no scientific setting changed. The local launcher's partition comment
was corrected afterward; the executing scientific checkout remains frozen.

## Locations and completion work

Remote code: `/export/home/fjoad/atk-evidence-paper3-seq-20261001`.
Remote outputs:
`/export/home/fjoad/atk-evidence/data/derived/takiddin-2021-robust-poisoning/sequential-sigmoid-pilot-20261001-attempt1`.
Remote log: `/export/home/fjoad/seq-transfer-20261001/pair-407294.out`.

The wrapper saves each fit's weights, optimizer, history, probabilities,
initial/final intermediate representations, source/input hashes and config;
compute-node reload checks precede completion. It then writes Table-V
comparison and pair artifact audit. After completion copy and verify all
artifacts, replay saved-score/identity audits locally, record all outcomes
and limits, and stop this pair. No local CER inference, further fit or
website work follows automatically. This is a descriptive pilot with
dependent observations, not a full-population reproduction or causal test
of reconstruction-mediated robustness.
