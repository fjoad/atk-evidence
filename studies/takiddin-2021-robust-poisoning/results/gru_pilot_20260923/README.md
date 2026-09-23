# GRU pilot — partial unpoisoned fit, audited

Checked September 24. Job **402378** ended with the intended partial-fit
exit code 2:0 after **22:45** of allocation. P00 reached its 900-second fit
guard after **32 full epochs plus seven batches of epoch 33**: 1,447 of
2,250 planned updates. P30 **never started**. This is not a completed
50-epoch experiment or a completed pair.

## Setup and stopping

Launcher revision: `46966c7e021e964ff582e1eb56422b75cc6a4ec7`; scientific
files are unchanged from `2d706b103ee03cc705cc3ef3f07718bc3ed7792a`.
The original inputs, 8x300 ReLU GRU, 48x1 sequence, repaired two-class
cross-entropy, seed 20260920 and batch 100 remain as specified in
[GRU_PILOT.md](../../GRU_PILOT.md). The [approved exception](../../GRU_RUNTIME_EXCEPTION.md)
raised only the launch ceiling; the old 720s gate remains false, and the
40min allocation/15min fit guards were unchanged. All 16 constructed GPU
fixtures passed in 100.295s before research inputs.

The first research epoch took 39.87s, the second 29.64s; full epochs 19–32
took about 24.72–25.90s. Constructed timing underestimated research training
cost. Its repeated fixed-size batch was not identical to the shuffled,
variably sized research pipeline; the timing-gap cause is not isolated.
This does not test the full-population RTX-2070 timing claim.

Fit 900.07785s; train/test scoring 135.86020s; on-GPU reload check 47.37145s;
research process total 1,096.92745s. Allocation also includes fixtures and
other overhead. Hardware: one V100-PCIE-16GB on crimv3mgpu005, four CPUs,
16 GiB. Peak TensorFlow allocation 903,443,968 bytes (not whole-device memory);
process peak RSS 1,544,432 KiB; Slurm sampled 1,535,128 KiB. No OOM reported.

## Partial scores

These scores are from the interrupted p00 model on the original 2,232-row
pilot, including synthetic test rows. They do not replace the 50-epoch
experiment or establish population uncertainty.

| Metric | Partial p00 | Paper p00, full-data context |
|---|---:|---:|
| Detection | 42.98643% | 92.4% |
| False alarms | 8.78438% | 6.8% |
| Specificity | 91.21562% | 93.2% |
| Precision | 82.75261% | 92.3% |
| Accuracy | 67.33871% | 92.8% |
| F1 | 56.58130% | 92.3% |
| AUC | 80.82379% | 92.1% |

Counts: TP475/FN630/FP99/TN1028. At FA<=6.8%, best saved-score detection is
**35.74661%**, versus 92.4% printed; reversal reaches 1.44796%. A cutoff/sign
change cannot rescue this saved partial model, not necessarily a completed
fit or another interpretation. At FA caps 17.6/20.6/33.3%, best detection is
60.72398/65.15837/81.71946%. These test-selected cutoffs are diagnostics,
not validated calibration.

Ranking exceeds the same rows' daily-mean AUC66.19624%. Original-only AUC
is 82.66499%; removing synthetic test rows does not undo pre-split synthesis
or dependence. Saved-model observed-label training accuracy is 67.06989%.
Full-epoch loss fell from .67070 to .40140, with fluctuations; a converged
plateau is not established. Partial epoch 33's .37901 uses only seven batches,
not a full epoch. No poisoning effect or paired-initialization comparison exists.

## Verification and preservation

[The read-only audit](audit_partial.py) verifies ten consumed input arrays,
metadata/poisoning, source/output hashes, test identities, raw/normalized
probabilities, predictions and the full saved-score analysis. It also checks
initial/final weight hashes, 4,058,702 parameters, serialized optimizer update
count, model configuration and kernel norms without importing/running the
neural model. Only Keras process-local shared-object IDs are omitted from
configuration comparison. Original status stays partial; the ordinary
completed-fit auditor correctly refuses it.

On-GPU reload equality is recorded in the original hashed result, not
independently replayed locally. The completed-pair audit never ran because
the wrapper stopped before p30. The [partial audit](partial_artifact_audit.json)
is not a pair certificate. Result SHA256:
`fa19293a1bac847fd7fb3ab8fe37b25f3ea92f52d88ad8e53cee129bbbdd4602`.

Preserved: [original result](p00_result.json), [history](p00_history.json),
[scheduler](slurm-accounting.txt), [log](slurm-402378.out),
[execution](execution.json), [authorization](launch_authorization.json).
Full weights/scores remain in ignored
`data/derived/takiddin-2021-robust-poisoning/gru-pilot-20260923-attempt1`.
Remote output has that suffix under `/export/home/fjoad/atk-evidence/data/derived/`.
Checkout: `/export/home/fjoad/atk-evidence-paper3-gru-approved-20260923`.
Log: `/export/home/fjoad/robust-gru-resume-transfer-jfMFay/slurm-402378.out`.

## Next decision

Stop this attempt. Reconcile the timing estimate with the actual pipeline and
specify sufficient bounded time for the unchanged 50-epoch question before
another allocation. No automatic resume/refit, p30, seed or ensemble launch.
A saved optimizer alone does not establish exact continuation of unsaved
shuffle/dropout state; any continuation or fresh attempt must disclose the
distinction and preserve this attempt. No new fits, local inference, website
changes or publication occurred during this check.
