# GRU pair — 50 epochs completed and audited

Job **402625** completed 0:0 in **56:11** on one V100-PCIE-16GB, four CPUs
and 16 GiB. Both original p00/p30 models completed **50 epochs and 2,250
updates**, without reaching their 35-minute fit guards. The pair learns useful
rankings but does not reach the paper's detection/false-alarm operating points
on this 20-customer pilot. This is not a full-population reproduction or a
limit on all GRU implementations.

## What changed from the interrupted attempt

The user approved a larger operational budget, not a new model or seed.
Freeze `1f847039ea3c5d410db5e9ee194a6625ecd2b54d` changes time-guard
dispatch/recording while preserving the scientific fit, data and device
functions from the original implementation. Model, inputs, seed 20260920,
50 epochs, batch 100, optimizer and final-epoch evaluation stayed fixed.
See [original contract](../../GRU_PILOT.md) and
[completion contract](../../GRU_COMPLETION.md).

Each case was freshly initialized. Both match the prior starting weights;
the entire first 32 unpoisoned epoch losses/accuracies/update counts also match
the earlier attempt exactly. This is a same-seed restart, not a different
seed or unverified checkpoint continuation. The interrupted run remains
preserved, including its inferior partial scores and compute cost.

The new 85-minute allocation allowed two 35-minute guards plus other work.
It was based on actual epoch times, not accuracy. The old fixed-batch preflight
underestimated the research pipeline's runtime; the precise cause is still
unidentified. There was no extra GPU timing job or method optimization.

## Completed metrics

All values are percentages. Paper columns concern the full reported
population; ours concern the original 20-customer/28-day pilot.

| Metric | Paper p00 | Pilot p00 | Paper p30 | Pilot p30 |
|---|---:|---:|---:|---:|
| Detection | 92.4 | 62.35294 | 78.5 | 39.72851 |
| False alarms | 6.8 | 9.93789 | 20.6 | 6.12245 |
| Specificity | 93.2 | 90.06211 | 79.4 | 93.87755 |
| Precision | 92.3 | 86.01748 | 79.0 | 86.41732 |
| Accuracy | 92.8 | 76.34409 | 78.9 | 67.06989 |
| F1 | 92.3 | 72.29801 | 78.7 | 54.43273 |
| AUC | 92.1 | 89.07390 | 79.4 | 79.88605 |

Counts TP/FN/FP/TN: p00 689/416/112/1015; p30 439/666/69/1058.
Both use the same 4,464 training and 2,232 test rows, with only the original
675 training-label flips between conditions. Pre-split ADASYN and known
customer-day/synthetic dependence remain part of this preparation.

Poisoned AUC is close to the reported value, and useful ranking survives.
That support is retained plainly; it does not reproduce the other metrics.
Relative to the interrupted p00 model, AUC improves from 80.82 to 89.07 and
default detection from 42.99 to 62.35. The partial result was not its final
50-epoch performance.

## Cutoff checks

All saved-score boundaries were evaluated in both orientations. These
test-selected cutoffs are diagnostic limits on the fixed scores, not
independently validated calibration.

| FA cap | p00 best DR | Actual FA | p30 best DR | Actual FA |
|---:|---:|---:|---:|---:|
| 6.8% | 51.40271% | 6.74357% | 41.35747% | 6.74357% |
| 17.6% | 80.99548% | 17.48004% | 64.43439% | 17.21384% |
| 20.6% | 83.71041% | 20.49689% | 67.14932% | 20.58563% |
| 33.3% | 92.66968% | 33.09672% | 80.18100% | 33.18545% |

At the corresponding printed corners, p00 cannot reach 92.4% DR within
FA<=6.8% (best 51.40271%); p30 cannot reach 78.5% within FA<=20.6%
(best 67.14932%). Favorable reversal reaches only 0.27149%/6.24434%.
Thus a cutoff/sign change does not rescue these two fitted models on these
rows. This does not exclude another model, training trajectory, population or
source interpretation.

Default detection drops 22.62443 points under poisoning; AUC drops 9.18785.
At common FA caps 6.8/17.6/20.6/33.3%, DR drops
10.04525/16.56109/16.56109/12.48869 points. Decision-point changes matter,
but measurable ranking deterioration remains.

The identical rows' no-parameter daily-mean score has AUC 66.19624%;
constant-score AUC is 50%. Original-only AUC is 88.07087/80.56326.
Removing synthetic test rows does not remove pre-split synthesis or training
dependence. No independent-customer confidence interval or equivalence claim
is made from this single pilot.

## Learning and runtime

The saved models' observed-label training accuracies are 76.56810%/73.40950%;
p30 true-label training accuracy is 67.20430%. Training-time accuracy includes
dropout and changing weights, so it is not the same measurement.

P00 loss falls from .67070 to .37406 at epoch 37, rises to .56124/.58846 at
epochs 39/40, then recovers to .44132 at epoch 50. P30 ends at .43377.
All values are finite, but this is not evidence of smooth convergence or a
stable performance ceiling. No best-epoch test score was selected, no extra
training was performed, and the cause of the late p00 deterioration is open.

Fits took 1,316.25543s and 1,375.62402s (21.94/22.93 minutes).
Scoring took 138.26919/146.12083s; GPU reload checks 47.13218/47.72522s.
Whole research processes took 1,513.03924/1,580.91651s. The allocation also
includes 101.495s of constructed fixtures, audits and other overhead.
TF peak allocations were 904,575,488/904,992,256 bytes (not whole GPU memory);
process peak RSS 1,546,792/1,547,308 KiB; Slurm sampled 1,548,244 KiB.
No OOM or fit-guard stop. These pilot measurements do not test the paper's
full-population RTX-2070 time claim.

Including prior preflight153s and partial-job1365s, GRU allocations actually
used 4,889s (81:29), not only this successful run's time. Repeated p00 updates
and constructed fixtures are included rather than erased.

## Verification and records

Both models, raw/normalized scores, initial weights, complete histories and
metadata are preserved in ignored local data and on Panther. The local
pair audit, comparison and repeated-prefix audit match cluster bytes exactly.
Checks bind all 20 consumed arrays, labels/identities, 675 flips, source/output
hashes, pairing, complete schedules, budgets and recomputed metrics.

Read-only HDF5 inspection additionally confirms final-weight hashes,
2,250 stored optimizer updates, 4,058,702 parameters, model configurations
(ignoring only process-local shared-object IDs) and recorded kernel norms.
GPU save/reload equality is recorded in the hashed original results; it was
not rerun locally. No model inference, fitting or new allocation occurred
during this outcome check. The earlier partial audit still matches.

Records: [p00 result](p00_result.json), [p30 result](p30_result.json),
[p00 history](p00_history.json), [p30 history](p30_history.json),
[pair audit](artifact_audit.json), [comparison](cluster_comparison.json),
[repeated prefix](repeated_prefix.json), [execution](execution.json),
[scheduler](slurm-accounting.txt), [job log](slurm-402625.out).

Result hashes: p00
`ce6847aba5c8c5da942779c02b100272ae7b0fa4e21e9eec9e2b0d52a05b4f9b`;
p30 `c4de02d467b0c584477a93a7ea7130e9b474e4989bfdefbd1222a231d6817a26`.

Full output suffix: `data/derived/takiddin-2021-robust-poisoning/gru-completion-20260924-attempt1`,
locally and under `/export/home/fjoad/atk-evidence/` on Panther.
Frozen checkout: `/export/home/fjoad/atk-evidence-paper3-gru-completion-20260924`.
Log: `/export/home/fjoad/robust-gru-completion-transfer-OceGkq/slurm-402625.out`.

## Boundary and next question

Stop this pair. Numerical finding: the declared repaired/native-Keras
completion did not reproduce the full metric pattern or corresponding
operating corners in this pilot. Mechanism: no temporal-structure ablation
or ensemble-reconstruction mechanism has been tested. Attainability: no
all-parameter, all-data or convergence limit is established.

Keep the late training deterioration and source ambiguities (including
generic tanh/inter-layer projection notation versus selected ReLU/native
layers) open. No extra seed or epoch is automatically justified by the gap.
The proposed next coverage step is source-specification of the standalone
AEA and its reconstruction objective before any fit or ensemble claim.
No AEA, ARIMA or ensemble experiment has been submitted. Website unchanged.
