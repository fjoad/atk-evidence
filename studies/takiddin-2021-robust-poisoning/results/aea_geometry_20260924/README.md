# AEA zero-fit check: conditional false-alarm exclusion

Job **402811** completed 0:0 in **30 seconds**, frozen at
`5d5d85050850f21f344c7509267b8215ca18767d`. It fitted no model, performed no
neural inference and changed no data. The scientific contract remains the
pre-outcome [AEA_GEOMETRY_CHECK.md](../../AEA_GEOMETRY_CHECK.md) from aa2c4ab.

The result is stronger than a bad training run, but conditional: on the
frozen novelty-pilot inputs, a reconstruction restricted to [0,1] cannot
reach the printed AEA false-alarm rates at the printed error threshold.
This is not a claim about another population, scaler, output range, error
definition, threshold, or the sequential classifier's output.

## Why no training was needed

For a standardized daily row x and any reconstruction r in [0,1]^48,

`MSE(x,r) >= L(x) = mean((x - clip(x,0,1))^2)`.

For every reading outside [0,1], its closest allowed reconstruction is the
nearest endpoint. Even choosing the best allowed value independently for
every coordinate cannot beat this lower bound. Thus, if L exceeds the
decision threshold, that row must be flagged regardless of network weights,
seed, optimizer, attention implementation or training duration, provided the
stated output range, score and prepared inputs remain fixed.

Here 741 unpoisoned-condition benign test rows and 748 poisoned-condition
benign rows satisfy `L > 0.515 + 0.000001`. This permits the most favorable
rounding of the printed0.51 threshold and a further numerical guard. There
are3,344 benign test rows in either case.

| Condition | Unavoidable FP, favorable rounding | Minimum FA | Paper FA | Favorable paper FA upper limit |
|---|---:|---:|---:|---:|
| p00 | 741 / 3344 | **22.15909%** | 5.2% | 5.25% |
| p30 | 748 / 3344 | **22.36842%** | 18.4% | 18.45% |

At the literal0.51 cutoff without the numerical guard, minimum FA is
22.24880% (744 rows) and22.78708% (762 rows). Therefore rounding is not the
explanation for this fixed-interpretation gap. No confidence interval or
independence assumption is needed for the pointwise statement on these
specific rows. Generalizing to other customers/preparations does require
additional evidence.

The upper error bounds do **not** exclude the reported detection rates: their
optimistic DR upper limit remains100%. The exclusion is from false alarms,
not a proved universal detection or AUC ceiling. Nor is every row's allowed
error endpoint necessarily jointly realizable by one network.

## Predeclared score-unit alternatives

Keeping the same printed0.51 cutoff and favorable rounding/guard:

| Error interpretation | Minimum FA p00 | Minimum FA p30 |
|---|---:|---:|
| MSE | 22.15909% | 22.36842% |
| RMSE | 55.14354% | 53.70813% |
| Sum of squared errors | 99.52153% | 99.55144% |

All three predeclared interpretations miss the corresponding printed FA
limits here. These are different score units, not three fitted models. MAE,
other normalizations/rescalings, another threshold and other output meanings
were not tested and are not excluded.

## Original versus synthetic benign data

The primary test contains187 original benign rows,3,157 synthetic benign
rows and3,360 original malicious rows. At favorable rounding, MSE's minimum
FA among original benign rows is28.34225% (53/187) at p00 and33.15508%
(62/187) at p30. Synthetic-benign minima are21.79284% and21.72949%.
The range obstruction is therefore not confined to synthetic evaluation rows.
Original-only percentages are diagnostics of this subset, not replacement
full-paper operating points or independent-population estimates.

## What simple scores achieve

These are untrained controls, not an AEA reproduction. The clipped-input
score is the pointwise lower bound itself; its AUC is not a bound on the AUC
of a learned model with different errors.

| Score | AUC p00 | AUC p30 | Literal0.51 FA p00 / p30 |
|---|---:|---:|---:|
| Constant-zero reconstruction MSE | 51.86267% | 48.65709% | 42.91268% / 44.01914% |
| Constant-half reconstruction MSE | 60.86976% | 58.98843% | 85.67584% / 85.10766% |
| Clipped-input reconstruction MSE | 59.44402% | 56.65526% | 22.24880% / 22.78708% |
| Negative mean raw consumption | 65.70123% | 65.70123% | No paper-defined cutoff |

None of these fixed untrained scores reaches the paper's operating corners
under its FA allowance, even with the predeclared score-reversal diagnostic.
That does not exclude a trained model's rankings at another cutoff. The
raw test arrays were verified equal directly, and their daily-mean AUC is
consequently unchanged. The reconstruction controls change with their scaler.
The full result includes every cutoff summary, seven operating-point metrics
for reconstruction controls, both orientations and all strata. AUC/DR are
explicitly null for synthetic-only evaluation, which has no malicious rows.

## Input and artifact audit

All56 input-array hashes, shapes, dtypes and finite values pass. The stored
scaler transforms verify without refitting. Every synthetic interpolation has
verified original benign parents and mixing weights. Raw test arrays,
identities and labels match across p00/p30; standardized values need not.

Training has373 observed-benign rows:373 true benign at p00,265 benign plus
108 attacks from six selected customers at p30. All108 contaminated training
rows occur exactly in test too;373 original source days are shared at both
levels. These consequences of the frozen replacement/all-attacks-to-test
interpretation remain explicit. This is not clean held-out validation and is
not the classifier's2,232-row evaluation. No resampling/splitting was redone.

Both saved score archives, identities, bound arithmetic, domains and metrics
were independently recomputed as read-only artifact verification locally;
the audit matches the cluster file byte-for-byte. All original inputs remain
unchanged. Result SHA256:
`d6e8b877ce31bd6ccaccbfa9ed478fad9587665782a9192590243f58834a42e5`.

The calculation took1.80581s, excluding the separate artifact audit/fixtures/
startup. Process peak RSS204,144KiB; Slurm's coarse sample51,316KiB is not
the process peak. Requested1 CPU/8GiB/noGPU, one numerical thread. Slurm
allocated2 logical CPUs on this node (crimv3srv025); CPUs/Task remained1.
All18 fixtures passed locally and on the compute node. No package install.

Records: [full result](result.json), [artifact audit](artifact_audit.json),
[execution](execution.json), [scheduler](slurm-accounting.txt),
[job log](slurm-402811.out). Full per-row score/identity archives remain in
ignored `data/derived/takiddin-2021-robust-poisoning/aea-geometry-20260924-attempt1`,
locally and under `/export/home/fjoad/atk-evidence/` on Panther.
Checkout: `/export/home/fjoad/atk-evidence-paper3-aea-geometry-20260924`.

## Decision

Stop this diagnostic. Training the same bounded-output/MSE model cannot
repair its printed0.51 false-alarm point on these frozen inputs; an extra
seed is not the next question. The proposed next step is a bounded source/
score-scale clarification or repair comparison, with changes to threshold,
scaler, reconstruction range or error definition kept explicit and separate.
No such follow-up has been run or silently substituted.

This does not authenticate or discredit the paper as a whole, infer intent,
test the sequential ensemble's classification mechanism, or transfer a finding
to another paper. No neural model, website update or publication occurred.
