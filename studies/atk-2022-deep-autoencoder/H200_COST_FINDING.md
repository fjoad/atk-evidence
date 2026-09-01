# H200 LSTM-SAE cost-pilot finding

**Date:** 2026-09-01

**Evidence type:** operational `X`; no numerical (`N`), mechanism (`M`), or
attainability (`A`) conclusion about the paper.

## Plain-language question

The clean-reader LSTM-SAE is too slow on the tested NVIDIA A16 for the frozen
72-hour full-anchor budget. We therefore asked one narrow question: can the
same two-epoch workload run fast enough on one H200 to make the unchanged
100-epoch ceiling affordable?

## What happened

The exact frozen commit `93ecd0d898349822187c5a22bb61ba0d26ff3730`
was synchronized to Panther and job `385602` requested one H200 for at most two
hours. Slurm did not allow the job to enter the partition. The account's only
association QOS was `gpulimit`; the H200 partition accepts dedicated H200 QOS
values instead. The job remained pending with reason “Job's QOS not permitted
to use this partition” and was canceled.

The accounting record reports zero elapsed compute. No model was constructed,
no epoch ran, no score was calculated, and no experimental artifact was
created.

## Decision

This is an infrastructure-access result, not evidence that the model passes or
fails. The required H200 timing and conservative projection were not observed.
The only measured projection therefore remains the A16 estimate: about 42.79
hours for the minimum ten epochs and 417.14 hours for the possible 100 epochs,
which fails the existing 72-hour promotion gate.

Under the frozen contract, the full LSTM-SAE anchor is not eligible and is not
launched. Continuing would require a new decision: obtain authorized faster
hardware, increase the compute ceiling, or explicitly define a bounded partial
run. None of those is silently substituted for the promised full anchor.

**Subsequent decision:** the user rejected faster or additional hardware. The
separate `PAPER_TIME_BUDGET_CONTRACT.md` instead treats the paper's reported
183-minute LSTM-SAE training time as the fit budget on one plausible
contemporaneous V100. That later decision does not change this zero-compute
H200 record.

## Boundary

The failed scheduling request says nothing about the truth of the paper's
numbers. It also does not strengthen the current implausibility conclusion.
It records why the next clean-reader numerical anchor could not be obtained
under the declared resources and stopping rule.
