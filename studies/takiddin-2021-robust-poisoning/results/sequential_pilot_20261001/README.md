# Sequential ensemble: completed constant-score pilot

The declared scalar-Sigmoid interpretation completed both original p00/p30
fits on Panther, but learned no test-set ranking: **AUC50% in both cases**.
Every unpoisoned test probability is0.5047591328620911; every poisoned
probability is0.3539589047431946. The0.5 rule therefore flags everything in
p00 and nothing in p30. No cutoff or score reversal rescues these fitted
scores at the paper's false-alarm limits. All artifact and metric audits pass.

This is one interpreted20-customer pilot, not a failure of every possible
implementation or the paper's full population. Passing the earlier eight-step
constructed learning gate did not establish useful48-step research learning.

## Frozen method and execution

Scientific code: `3705bcca04279bbfa8523a573b9b573f123463b7`.
The [interface addendum](../../SEQUENTIAL_INTERFACE.md) selected
**I-SEQ-scalar-sigmoid before results**, from the component-output reading of
TableII. Its scalar shape/readout remains an explicit completion of omitted
source details. The separate linear control was not promoted. Other settings
follow the [sequential contract](../../SEQUENTIAL_ENSEMBLE_PILOT.md):
9,240,802 parameters,48-step inputs, native ReLU recurrent hidden layers,
joint repaired BCE,Adam.001, no dropout,MaxNorm1,50epochs,batch100,final
weights only. There is no reconstruction objective or pretraining.

Both fresh starts use seed20260920 and byte-identical initial-weight files.
The unchanged preparations contain4,464 training/2,232 test rows from20
customers/28days, with the original pre-split ADASYN and dependence intact.
P30 flips675 observed malicious training labels fromsix customers:15.12097%
of training rows, not30% of all rows. Features and test identities match.
The existing ScienceDB CSV semantic-equivalence provenance is retained;
official restricted CER archives remain unavailable locally. Strict data
verification passes that declared branch.

The [V100 preflight407255](../sequential_preflight_20261001/README.md) passed
all14 software checks and the full48-step timing/device/reload gate. Its
verified SHA is `721256059d34a104fa2eda04ccb0bb8775603d01206117e4e82e1a9e54782c07`.
The research pair was submitted once as **407294**, completed **0:0 in1:21:28**
ononeV100-16GB (crimv3mgpu005),4CPUs/16GiB, within its3h allocation and
70min/fit guards. P00 andp30 each completed50epochs/2,250updates.

| Measured stage | p00 seconds | p30 seconds |
|---|---:|---:|
| Training | 2125.66 | 2008.09 |
| Scoring | 207.48 | 198.73 |
| Saved-model reload and verification | 73.71 | 70.53 |
| Whole research process | 2501.50 | 2362.85 |

The8:01 preflight plus1:21:28 pair used1:29:29 of actual GPU allocations.
The20min/180min requests were ceilings, not measured costs. TensorFlow's
allocator peak was1,582,497,792/1,583,084,544 bytes, excluding memory outside
that allocator. This is a V100 pilot measurement, not RTX2070/full-data timing.

Slurm initially held the pair because gpu-short permits only2h. Before any
fit, the same pending job was moved to gpu-all, retaining the3h ceiling and
V100 resources. No duplicate submission or scientific setting change occurred.
The local launcher's partition was corrected afterward; the executing code
remained frozen. [Accounting](pair-accounting.txt), [job log](slurm-407294.txt)
and [execution details](execution.json) preserve this history.

## Complete metrics

All values below are percentages. Paper columns are TableV full-population
context, not a matched-population comparison. The previously established
single-confusion-matrix inconsistencies also remain explicit.

| Metric | Paper p00 | Pilot p00 | Paper p30 | Pilot p30 |
|---|---:|---:|---:|---:|
| DR | 95.2 | 100.00000 | 92.2 | 0.00000 |
| FA | 2.9 | 100.00000 | 5.8 | 0.00000 |
| SP | 97.1 | 0.00000 | 94.2 | 100.00000 |
| PR | 95.6 | 49.50717 | 92.7 | undefined |
| ACC | 96.1 | 49.50717 | 93.2 | 50.49283 |
| F1 | 95.4 | 66.22715 | 92.4 | 0.00000 |
| AUC | 97.4 | 50.00000 | 92.0 | 50.00000 |

P00 TP/FN/FP/TN:1105/0/1127/0. P30:0/1105/0/1127. Precision is undefined
when no row is predicted positive; it is retained as null in JSON. Original-
row AUC is also50% in both cases. Every attack type is detected at100%/0%,
and both original/synthetic benign strata haveFA100%/0%, respectively.
Full strata/counts remain in [p00](p00_result.json) and [p30](p30_result.json).

The constant scores admit only no alarms or all alarms. Consequently best
DR is0% in both score directions at each declared FA cap:2.9,2.95,5.8,5.85,
9.3 and24.4%. Even favorable one-decimal allowances cannot rescue the
printed95.2/2.9 or92.2/5.8 detection corners for these fitted scores. This is
not a bound over other learned weights, model interpretations or populations.

The default-rule DR drop100→0 follows a constant probability crossing0.5;
it is not a decline from useful initial discrimination. Unchanged50%AUC
is not evidence of robustness. The observed training attack-label fractions
are0.505152/0.353943. Final epoch BCE0.693111/0.650009 is close to the
optimal constant-prior BCE0.693094/0.649853. Training observed-label accuracy
50.51523/64.60573% likewise gives no evidence of useful classification.
Initial saved test probabilities were already constant at float32 precision.
No claim about an infinite-time limit or every intermediate epoch follows.

## Existing comparisons on the exact same test rows

These are replays of preserved scores, with verified identities and no new
fitting. Cutoffs use test labels diagnostically and are not calibrated rules.

| Model | p00 AUC | p30 AUC | p00 bestDR atFA≤2.9 | p30 bestDR atFA≤5.8 |
|---|---:|---:|---:|---:|
| Sequential interpretation | 50.00 | 50.00 | 0.00 | 0.00 |
| Random forest | 98.55 | 94.36 | 91.86 | 85.61 |
| Feed-forward | 96.35 | 90.89 | 84.07 | 70.95 |
| Earlier GRU | 89.07 | 79.89 | 44.16 | 38.28 |
| Negative daily mean, no learned parameters | 66.20 | 66.20 | 8.05 | 10.68 |

Useful signal exists in these pilot rows; its absence from this sequential
score is not universal across learners. The earlier GRU has different head,
dropout and constraint settings, so it is not a matched removal-of-AEA control.
No causal attribution to adding the AEA follows from this table.

## Intermediate output and mechanism boundary

| Mean per-row error | Initial, both | Final p00 | Final p30 |
|---|---:|---:|---:|
| MSE | 1.26676968 | 1.26164112 | 1.03868614 |
| MAE | 0.93437343 | 0.93101129 | 0.67248430 |

Zero-output MSE is1.038686137003; training-mean MSE is1.038686137011.
P00's intermediate output ranges0.486082–0.497284, with2,223 distinct saved
profiles, yet final classification scores are identical. P30's intermediate
output ranges0–4.918e-7: effectively zero, with only tiny residual variation.
Its lower MSE matches the zero-output comparison rather than demonstrating
useful reconstruction. Both norms were recorded regardless of outcome.
These output descriptions do not identify the causal route to failure or
test whether reconstruction can cause robustness in a functioning model.

## Audit, preservation and next decision

All15 copied files match [Panther SHA256 values](pair-artifact-sha256.txt).
Local [pair audit](artifact_audit.json) and [TableV comparison](cluster_comparison.json)
match cluster bytes. Source/input hashes, paired initialization,50-epoch
histories, labels/identities, raw probabilities and per-row error arithmetic
verify. Pure HDF5 inspection confirms52 variables/9,240,802 parameters,
finite weights/optimizer state,2,250updates, source configuration and norms.
Compute-node reload reproduces probabilities and representations exactly.
The [read-only audit script](audit_saved_pair.py) and
[descriptive output](saved_artifact_analysis.json) preserve those checks.
No research model inference or training ran locally.

Two audit-operation errors are preserved: raw config equality initially
rejected Keras archive-only `shared_object_id` fields; removing only that
identity metadata gives exact equality for every setting. A whole-file
login-node hash process was killed before creating its manifest; streaming
`sha256sum` then succeeded. The first copy request reported the missing
manifest, and the completed streaming manifest was subsequently copied.
Neither error changed a model/output or caused a research rerun.

Stop this pair. The next named question is why profile differences do not
reach the48-step classification score, including at initialization, despite
passing the eight-step software learning check. Propose a bounded zero-fit
activation/gradient inspection of the saved initial/final states before
another training run, seed or source alternative. It has not been launched.
The scalar interface and native-cell source ambiguities remain open. No
family-wide impossibility, author-intent, full-population or independent-row
uncertainty claim is supported. Other studies and the website are unchanged.

Raw artifacts remain at
`data/derived/takiddin-2021-robust-poisoning/sequential-sigmoid-pilot-20261001-attempt1`
locally and under `/export/home/fjoad/atk-evidence/` on Panther. The isolated
executing checkout remains `/export/home/fjoad/atk-evidence-paper3-seq-20261001`.
