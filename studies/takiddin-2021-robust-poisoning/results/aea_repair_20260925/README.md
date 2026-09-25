# AEA score/scale repair envelope

Job 403408 failed in one second before the batch script ran because its Slurm
output path was misspelled. No research input loaded and no output directory
was created. This failure is preserved. Corrected job 403409 completed 0:0 in
41 seconds on one CPU task, 8 GiB, no GPU.

This is a finite label-informed interval relaxation, not an AEA or detector.
It gives benign rows their smallest possible error and malicious rows their
largest. A passing branch is only not excluded; it has not been trained or
shown realizable.

| Branch | p00 oracle AUC | p30 oracle AUC | p00 upper DR at paper corner | p30 upper DR |
|---|---:|---:|---:|---:|
| Feature-z MSE | 97.92032% | 97.66592% | 88.24405% (excluded) | 87.23214% |
| Feature-z MAE | 99.77434% | 99.66222% | 100% | 100% |
| Raw-unit MSE | 98.67922% | 98.13610% | 94.73214% | 92.14286% |
| Global-z MSE | 97.37289% | 96.82960% | 92.26190% (excluded) | 90.26786% |
| Training-only min-max MSE (C control) | 100% | 100% | 100% | 100% |

The reported AEA corners are DR/FA 94.1%/5.2% at p00 and 80.8%/18.4% at
p30. Favorable one-decimal rounding is allowed. Feature-z MSE p00 has an
empty favorable threshold interval: required cutoff [1.44910, ...), allowed
malicious upper-error cutoff less than 1.04309. Global-z MSE p00 is also empty
[1.00886, 0.89953). A positive score multiplier cannot repair an empty interval.

Surviving branches are not evidence that the paper used them. No weights,
attention, decoder, threshold selection or classifier was trained. Linear or
unbounded output remains analytically inconclusive.

All 56 input arrays, metadata, scalers, raw identities/labels, 108 p30
overlaps, synthetic ancestry and ten repair archives pass the independent
audit. The prior geometry result is unchanged; local audit matches cluster.
Global-z and min-max statistics are training-only; tests were not clipped and
test statistics were not used. No AEA fit, ensemble, extra seed, website edit
or publication follows automatically. The next question is which surviving
score/scale branch is textually defensible before implementing an AEA.

Records: [result](result.json), [audit](artifact_audit.json),
[scheduler](slurm-403409.out), [accounting](slurm-accounting.txt).
