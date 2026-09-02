# Source and printed-table audit finding

**Date:** 2026-09-02

**Status:** complete; stopped before data preparation or training

**Evidence:** paper-source reconstruction plus a preregistered static numerical
consistency check (`P` and `C/N`). This is not a trained reproduction,
mechanism test, or attainability search.

## Short answer

The paper gives an unusually useful compute boundary: an NVIDIA GeForce RTX
2070, 50 epochs, batch size 100, and approximately one to four hours of
training depending on the detector. Those values are now fixed as constraints
for any later reproduction rather than details to infer after a result misses.

Before spending that compute, the printed tables expose a source-level problem.
Across 68 model-by-poisoning rows, specificity and the accuracy expected for a
balanced evaluation are internally coherent. Precision usually is not. Three
sequential-ensemble rows are stronger: their printed detection rate, false-
alarm rate, precision, and accuracy cannot all describe one evaluation
population at any class prevalence within the full allowance for one-decimal
rounding.

That is an internal inconsistency in the reported metrics. It does not show how
the values were produced, does not establish fabrication or intent, and does
not yet say whether the models or broader qualitative ordering can be
reproduced.

## Frozen source and scope

The ten-page paper was rendered and visually inspected in full. Tables II–V
were transcribed directly from the rendered pages. The source PDF SHA-256 is:

`03a372fb5ee129799d43a50e6bec8f86fb108e9f8bdc85350e3ffe3e13d4a4ff`

The audit contract, all table CSVs, and the initial checker were frozen in
commits `5e92700`, `85d96c7`, and `6bfaca5` before the first execution. The
first output is preserved as
[`source_table_audit_20260902_attempt1.json`](results/source_table_audit_20260902_attempt1.json).
That attempt calculated every predeclared identity correctly but omitted some
promised prose comparisons from its report. Commit `1c7e45b` added only those
reporting calculations; the corrected result is
[`source_table_audit_20260902.json`](results/source_table_audit_20260902.json),
SHA-256
`a9113f6b58f2c3ea15ed44b2e3b8fb9081367d946e04afddf8f69c6d0c52da56`.
The identity counts did not change.

## Arithmetic result

Each printed value `v` received the favorable closed interval
`[v-0.05, v+0.05]` percentage points, clipped to `[0,100]`. Merely touching a
derived interval counted as a pass.

| Check | Passed | Failed | What it tests |
|---|---:|---:|---|
| `SP = 100 - FA` | 68 | 0 | Printed specificity |
| `F1 = 2 DR PR / (DR + PR)` | 67 | 1 | Printed F1 |
| balanced `ACC = (DR + SP)/2` | 68 | 0 | Accuracy under the stated balance |
| balanced `PR = DR/(DR + FA)` | 10 | 58 | Precision under the stated balance |
| one common prevalence exists | 65 | 3 | DR, FA, PR, and ordinary ACC can coexist |

The isolated F1 failure is Table III, random forest, 20% poisoning. Printed DR
72.7 and PR 73.8 imply F1 in `[73.1959, 73.2959]`, while printed F1 72.7 means
`[72.65, 72.75]`. The intervals are disjoint. An isolated cell or reporting
error remains a straightforward explanation.

The stronger failures are Table V's sequential ensemble at 0%, 10%, and 30%
poisoning:

| Poisoning | Prevalence allowed by DR/FA/PR | Prevalence allowed by DR/FA/ACC |
|---:|---:|---:|
| 0% | 39.12–40.54% | 47.37–57.89% |
| 10% | 42.11–43.31% | 45.00–55.00% |
| 30% | 44.00–44.82% | 45.00–55.00% |

Because these ranges are disjoint, changing the balanced-evaluation assumption
does not rescue those three rows. No continuous confusion-rate construction on
one population can yield all four printed values within the rounding intervals.
The 20% row is not excluded by this stronger screen: its ranges overlap narrowly.

The 58 balanced-precision failures require more care. The novelty procedure
explicitly applies ADASYN to balance its test population. The two-class
procedure balances before a later split, which need not leave each split
exactly balanced. The fact that all 68 printed ACC values agree with balanced
accuracy suggests a common balanced-accuracy calculation, but does not prove
that every evaluated split had exactly 50% prevalence. Accordingly, the 58/68
result is reported as conditional on the stated balance; the three
any-prevalence failures are the balance-independent source-level inconsistency.

## Prose recalculation

Most broad degradation and ordering statements are reasonably represented by
the printed DR values, although several rounded prose values are not the exact
table averages:

- Generalized mean DR loss from 0% is 4.3, 9.2, and 14.9 percentage points at
  10%, 20%, and 30% poisoning; the paper says approximately 4%, 9%, and 14.8%.
- Customer-specific mean loss is 4.54, 9.91, and 16.33 points; the prose says
  about 4.4%, 9.8%, and 16%.
- The generalized advantage over customer-specific detectors rises from 2.60
  to 4.03 points, consistent with the stated 2–4% range.
- Generalized deep-minus-shallow mean DR rises from 6.21 to 7.78 points. That
  supports the body text's roughly 5–8% description; the abstract's 12% can
  only be read as an individual or “up to” comparison, not this group mean.
- AEA > GRU > feed-forward DR holds at every generalized poison level.
- Sequential-ensemble DR falls 0.9, 1.9, and 3.0 points from its unpoisoned
  value, close to the stated 1–3%. Ensemble averaging falls 3.3, 7.5, and 13.3
  points; the prose's approximately 4%, 8.4%, and 13.4% is not the exact
  percentage-point sequence and leaves “performance” ambiguous.
- At 30% poisoning, sequential DR exceeds AEA by 11.4 points and ensemble
  averaging by 10.9 points.

These checks support the qualitative ordering printed in the tables. They do
not repair rows whose metrics cannot come from one confusion matrix.

## What this establishes

### Numerical

No detector was run. The narrow finding is that at least three headline Table
V rows are internally inconsistent as printed, independent of dataset,
implementation, seed, or hardware. A faithful reproduction cannot match every
metric in each such row using one evaluation population and the paper's metric
definitions.

### Mechanism

Untested. The paper attributes the sequential ensemble's robustness to staged
AEA reconstruction, GRU feature extraction, and feed-forward classification.
The tables alone do not isolate those capabilities from data volume,
preprocessing, selection, or thresholding.

### Attainability

Untested for model performance. The only structural statement is about the
printed metric tuple: the three identified tuples are unattainable from one
continuous confusion-rate population within the stated rounding allowance.
This says nothing about whether DR 92.2% and FA 5.8%, considered alone, are
attainable by a model.

## Explanations still open

The source cannot distinguish a reporting or copy error, metrics calculated on
different populations or thresholds, a precision column produced by a
different definition, an undocumented evaluation protocol, or another process
failure. The systematic agreement of SP and balanced ACC and the broad failure
of balanced precision are facts to explain, not evidence of intent. See the
[`EXPLANATION_REGISTER.md`](EXPLANATION_REGISTER.md).

## Compute boundary and next decision

The paper states 50 epochs and batch 100 on one RTX 2070, with approximately:

- one hour for shallow detectors;
- 1.5–3 hours for deep detectors;
- three hours for ensemble averaging;
- four hours for the sequential ensemble; and
- two seconds for an online decision.

Any later run must freeze what each clock includes and use the named RTX 2070
when reasonably available. If it is unavailable, a favorable pre-result
throughput calibration against a period-appropriate single device is required.
Newer or additional GPUs, a retry, or a longer fit cannot rescue the bounded
paper-time question.

**Stop here for discussion.** The smallest sensible next phase is not the
four-hour ensemble. First identify the exact 3,000-customer population and
freeze one nested split and poison construction. Then cost one generalized
baseline at 0% and 30% poisoning against the paper's one-hour shallow or
1.5–3-hour deep envelope. The random forest is cheapest; the feed-forward
model is more informative for the central deep-versus-shallow claim. No data
preparation, model implementation, training, scoring, or cluster submission is
authorized by this finding.
