# Output limits and useful information after the first reproduction

Date: 2026-08-31

Status: diagnostic round and small explanatory control completed; experiments stopped

## What changed

**VERIFIED conditional attainability limit (`C/A`).** On the frozen prepared
evaluation, no Softmax reconstruction scored by mean squared error can reach
the reported FC-SAE row. This is stronger than a failed seed or an empirical
training plateau. It follows from a label-aware relaxation that grants each
example its most favorable allowed reconstruction.

**OBSERVED small, nonzero contribution (`C/M`).** The trained score improves
original-row balanced accuracy over zero reconstruction by 0.891 percentage
points. The conditional customer-bootstrap interval is [0.805, 0.981]. It is
inside the stated +/-1-point region, but positive; some attack-specific effects
are larger. “Training learned nothing” is not supported.

## Bound and scope

For the fixed input `x`, an allowed reconstruction has nonnegative coordinates
adding to one. Minimum and maximum MSE are its squared distance to the simplex
and to its farthest vertex, respectively, divided by 48. Their derivation,
strict cutoff semantics, rounding allowance, and outward numerical padding are
in the [frozen contract](POST_ANCHOR_DIAGNOSTICS.md).

Assigning attacks their largest allowed score and benign examples their
smallest score provides an optimistic upper bound across every shared cutoff.
The actual network cannot outperform this label-aware relaxation. Different
network widths, weights, seeds, optimizers, and training durations cannot alter
this conclusion while inputs, the Softmax output domain, and MSE score remain
fixed. A different preparation, output domain, or score is outside this bound.

| Quantity | Paper | Optimistic limit, post-ADASYN (approx.) | Original-row diagnostic limit (approx.) |
|---|---:|---:|---:|
| Balanced accuracy | 83% | 50.92105% | 50.15672% |
| AUC | 81% | 45.10918% | 36.84295% |
| Detection at FA <= 15% | 81% | 9.24779% | 5.59785% |
| Detection at FA <= 15.5% | at least 80.5% allowing rounding | 9.53747% | 5.79752% |

At the printed 0.58 cutoff, even the optimistic post-ADASYN reconstruction can
detect at most 29.57856% and must falsely flag at least 44.28282%. A separate
reversed-direction relaxation also fails: maximum balanced accuracy 64.55683%,
AUC 68.23056%, and detection 35.81886% at FA <= 15%. Reversal is a control,
not the printed method. The original-row reversed limits are 69.86019% balanced
accuracy and 74.08041% AUC.

All 8,884,989 saved trained scores lie inside the padded bounds. This is a
float64 numerical evaluation of the analytic bound, not certified interval
arithmetic or a population-wide probability statement. It does not use a
confidence interval to extrapolate over training seeds. It does not identify
what the authors implemented or imply intent.

## Useful work: small average gain, heterogeneous effects

The original-row view contains 750,767 benign source days and six attack
derivatives of each, from 1,409 held-out customers. Synthetic benign rows are
excluded from uncertainty calculations because their full generation
dependencies are not available in these records. This view does not replace
the paper's resampled evaluation.

| Comparison | Original-row balanced-accuracy gain | Conditional 95% interval |
|---|---:|---:|
| Trained minus zero reconstruction | +0.89081 points | [0.80454, 0.98117] |
| Trained minus constant uniform Softmax output | +0.85335 points | [0.77034, 0.94131] |
| Trained minus nearest simplex reconstruction | +0.02975 points | [0.00916, 0.05077] |

The intervals resample customers together with every day and attack sibling,
2,000 times with seed 20260831. Model, scaler, split, and generated attacks
remain fixed. They require exchangeable customer clusters under that
conditioning; they are not seed-level uncertainty or independent confirmation.
The +/-1-point region was recorded before these new measurements but after the
initial aggregate result was known; it is an operational effect criterion, not
a universal definition of usefulness.

Trained-minus-zero balanced-accuracy differences for attacks 1–6 are
`+0.756, -0.248, +0.165, +1.724, +2.029, +0.919` points. The mean does not
justify saying every attack has negligible benefit. The original-view gain is
mostly fewer benign alarms: FA falls from 61.25789% to 57.05046%, while aggregate
attack detection falls from 27.90977% to 25.48396%. Fewer false alarms can be a
real useful change, but it is not better attack detection.

The full resampled view remains the unchanged 40.17516% balanced accuracy,
versus 38.99771% for zero reconstruction. Original-row balanced accuracy is
34.21675% versus 33.32594%; its poor AUC is 32.11707% versus 30.42594%.
Ordinary row-weighted correctness can decrease while balanced accuracy improves
because this original view has six attacks for every benign row. Both the
beneficial/harmful decision counts and balanced metrics are preserved.

## Important counterevidence to “the scores contain no new information”

Within 100 label-blind input-energy bands on original rows, pair-weighted AUC is
49.63391% for energy alone, 65.24185% for the trained score, and 68.12116% for
trained score minus energy. The scores are not information-identical merely
because their global correlation is 0.999253.

This does not by itself identify the contribution as learned: a no-training
reconstruction using additional input statistics might recover the same
distinctions. We therefore froze one small follow-up in
[ENERGY_BAND_CONTROL.md](ENERGY_BAND_CONTROL.md) before executing it.

On 10,000 seeded source days and all six attack siblings (70,000 original rows),
the same within-band comparison gave:

| Score | Pair-weighted within-energy-band AUC |
|---|---:|
| Input energy | 49.74125% |
| Uniform constant Softmax output | 55.02019% |
| Nearest simplex reconstruction | 62.18387% |
| Trained FC-SAE | 65.48803% |

The trained score exceeds both tested no-learning geometry controls on this
conditional ranking measure. This weakens the hypothesis that those controls
fully explain its residual information. The probe is adaptive `X/M`, not an
independent confirmatory test, not a new reproduction, and not proof that no
other simple feature can match it. There is no confidence interval for these
within-band AUC differences. The separately calculated customer-bootstrap
intervals above apply to balanced accuracy, not this ranking statistic.

**Current conclusion:** the fixed pipeline has an unattainable target, but the
trained score is not interchangeable with the tested no-learning scores.
“Nothing useful was learned” is weakened, not established. These are score
comparisons, not a matched trained-versus-untrained causal test. A small
aggregate gain and a large structural shortfall can coexist. Neither result
establishes the paper's recurrent/attention mechanism.

## Runtime and evidence

CPU job `385090` completed `0:0` in 4:16 including the pilot and interaction.
The 512-row pilot took 5.00 seconds; the full analysis took 112.92 seconds,
including 51.57 seconds for geometry. Maximum recorded RSS was 2,627,800 KiB.
All 200 deterministic tests passed before execution. No model was retrained;
the original checkout, preparation, scores, and weights remain unchanged.

The [complete records](results/post_anchor_20260831/) include the pilot, full
JSON, figures, and execution/transfer checks. Analysis revision `1175e8d` and
its contract are immutable; input and source hashes are inside the JSON.
The adaptive control used code `26a42db`, CPU job `385091`, and completed `0:0`
in 53 seconds including startup; its analysis took 18.03 seconds and maximum
RSS was 163,468 KiB. Its JSON transfer hash is
`d988527f801c2d30d3bc8337e8ce26568939155514539843f39bee256ab58ee4`.
The [control record](results/post_anchor_20260831/energy_band_control.json)
includes the sample-index checksum; indices remain on the cluster and are
regenerable from the recorded source count and seed.

Publication checks pass all 206 deterministic tests, including 14 static
report tests. These verify displayed measurements, upward-rounded public
bounds, paired intervals, figure bytes, contract/code/result hashes, and links.
No public number replaces a raw measured record.

## Consequence for the next step

Do not run another expensive seed to test whether this fixed scoring system can
reach the target: the bound already answers that question. Broader model-family
attainability and the paper's architecture-specific causal claims remain open.
The residual-information control is now complete. Stop this diagnostic round.
The next proposed work is to examine source-supported alternatives that would
actually change the bound's assumptions, starting with a source/semantic map
and cheap controls, not a broad search. No alternative is inferred from target
proximity. This proposal does not authorize another run.
