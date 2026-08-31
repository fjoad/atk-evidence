# Study 1: an electricity-theft autoencoder

This study examines:

> A. Takiddin, M. Ismail, U. Zafar, and E. Serpedin, “Deep
> Autoencoder-Based Anomaly Detection of Electricity Theft Cyberattacks in Smart
> Grids,” IEEE Systems Journal 16(3), 4106–4117, 2022.

## Current result

We implemented the paper's FC-SAE model and tested the Table III result on the
named Irish electricity data. In one documented run:

| Measure | Paper | Our run |
|---|---:|---:|
| Detection | 81.00% | 25.48% |
| False alarms | 15.00% | 45.13% |
| Balanced accuracy | 83.00% | 40.18% |
| AUC | 81.00% | 39.40% |

The saved artifacts and metrics passed the recorded checks. Every cutoff on the
same scores remains far below the reported result. The model's scores closely
follow a zero-reconstruction input-magnitude score.

This numerical result is one implementation and one seed. A subsequent
no-training bound goes further: on the fixed prepared data, even label-aware
optimal Softmax reconstructions with MSE scoring stay below 50.93% balanced
accuracy. Detection at at most 15% false alarms is bounded by 9.25%, versus
81% reported. This excludes every weight and seed under those assumptions,
not different preprocessing, output domains, or scores.

The fitted score is not identical to simple input magnitude: original-row
balanced accuracy improves by 0.89 points, and within-energy ranking exceeds
both tested no-training geometry controls. These comparisons do not identify
a learned causal mechanism or establish “nothing useful learned.”

Read the [initial finding](CLEAN_READER_FINDING.md), the
[completed diagnostic finding](POST_ANCHOR_FINDING.md), the
[paper-to-code fidelity record](CLEAN_READER_FIDELITY.md), and the
[remaining explanations](EXPLANATION_REGISTER.md).

## What was implemented

The current paper-facing implementation is exactly five direct files:

```text
reproduction/download_data.py
reproduction/prepare_data.py
reproduction/models.py
reproduction/run_experiment.py
reproduction/analyze_results.py
```

The run uses the final FC-SAE architecture, all officially labeled residential
meters, disjoint training/test customers, six paper-derived attacks, the
paper-positioned test-set ADASYN step, joint feature scaling, benign-only
training, mean squared reconstruction error, and the printed threshold 0.58.
Necessary completions are documented in
[CLEAN_READER_SPECIFICATION.md](CLEAN_READER_SPECIFICATION.md). The literal
Attack 3 and threshold-selection failures remain visible.

Older study-root wrappers, `src/`, and earlier results are retained as
historical evidence. They are not the implementation of this run and should not
be resumed as though they were.

## Next decision

The first run, audit, and user-approved bounded diagnostic round are complete.
No experiment remains running. The next proposed question is which
source-supported alternatives would actually change the bound's assumptions.
Begin with a source/semantic map, not repeated seeds or a broad search. A
matched recurrence or attention comparison remains untested.

Repository setup and authorized data access are in
[Getting started](../../docs/GETTING_STARTED.md). Do not launch another seed,
model, configuration, or control without the approved scientific question.
