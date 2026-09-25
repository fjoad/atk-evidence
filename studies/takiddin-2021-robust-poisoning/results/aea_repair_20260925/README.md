# AEA score/scale repair envelope

**Correction after independent recheck:** the original JSON and interval
calculations are unchanged. Three p30 DR values below previously used the
p00 FA cap; those reporting errors are corrected. Passing the free-cutoff
relaxation must not be described as passing the printed0.51 cutoff. See the
[recheck](../aea_recheck_20260925/README.md).

Job 403408 failed in one second before the batch script ran because its Slurm
output path was misspelled. No research input loaded and no output directory
was created. This failure is preserved. Corrected job 403409 completed 0:0 in
41 seconds on one CPU task, 8 GiB, no GPU.

This is a finite label-informed interval relaxation, not an AEA or detector.
It gives benign rows their smallest possible error and malicious rows their
largest. A passing branch is only not excluded; it has not been trained or
shown realizable.

| Branch | p00 oracle AUC | p30 oracle AUC | p00 upper DR, FA≤5.25% | p30 upper DR, FA≤18.45% |
|---|---:|---:|---:|---:|
| Feature-z MSE | 97.92032% | 97.66592% | 88.24405% (excluded) | 100% |
| Feature-z MAE | 99.77434% | 99.66222% | 100% | 100% |
| Raw-unit MSE | 98.67922% | 98.13610% | 94.73214% | 100% |
| Global-z MSE | 97.37289% | 96.82960% | 92.26190% (excluded) | 96.81548% |
| Training-only min-max MSE (C control) | 100% | 100% | 100% | 100% |

The reported AEA corners are DR/FA 94.1%/5.2% at p00 and 80.8%/18.4% at
p30. Favorable one-decimal rounding is allowed. Feature-z MSE p00 has an
empty favorable threshold interval: required cutoff [1.44910, ...), allowed
malicious upper-error cutoff less than 1.04309. Global-z MSE p00 is also empty
[1.00886, 0.89953). A positive score multiplier cannot repair an empty interval.

These are **free-cutoff** upper bounds, not results at0.51. At the printed
cutoff, even allowing0.505–0.515 and the numerical guard, MAE has minimum
FA36.75239%/33.79187%; both miss the paper caps. Its necessary free-cutoff
intervals are [0.78932,0.97132) and [0.60363,1.25402), excluding0.51.
Raw-unit MSE's p00 maximum DR at the favorable fixed cutoff is90.02976%,
below94.05%, despite its adequate FA bound. Feature-z/global-z MSE also fail
fixed-cutoff FA. Only min-max is not excluded at that cutoff in both cases
among these five branches; it is a separately labeled control, not a selected
repair or evidence of a successful model. Joint attainability does not follow
from separate favorable endpoint bounds.

Surviving free-cutoff branches are not evidence that the paper used them. No weights,
attention, decoder, threshold selection or classifier was trained. Linear or
unbounded output remains analytically inconclusive.

All 56 input arrays, metadata, scalers, raw identities/labels, 108 p30
overlaps, synthetic ancestry and ten repair archives pass the independent
audit. The prior geometry result is unchanged; local audit matches cluster.
Global-z and min-max statistics are training-only; tests were not clipped and
test statistics were not used. No AEA fit, ensemble, extra seed, website edit
or publication follows automatically. No training branch is selected. Any
future proposal must separately specify input scaling (the source omits the
axis), reconstruction training loss, anomaly score and threshold rule. The
score-norm comparison does not choose a training objective.

Records: [result](result.json), [audit](artifact_audit.json),
[scheduler](slurm-403409.out), [accounting](slurm-accounting.txt).
