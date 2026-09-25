# Rechecking the recent AEA conclusions

The user questioned the last few steps. We re-read the relevant source pages,
replayed the saved geometry/repair audits, independently recalculated their
bounds, and challenged the constructed neural prototype. There were real
interpretation, reporting and software-validation errors. The original
numerical artifacts themselves survived the recheck.

## What changed

### MAE was promoted for the wrong reason

The previous record called feature-standardized MAE the least invasive repair
and said it preserved the threshold context. That conflated two different
questions: can any cutoff pass an optimistic bound, and can the printed 0.51
cutoff pass? MAE passed only the first.

| Pilot condition | Necessary MAE cutoff interval, favorable rounding | Minimum FA near printed 0.51 | Favorable paper FA cap |
|---|---|---:|---:|
| p00: unpoisoned | [0.78931975,0.97132070) | 36.75239% | 5.25% |
| p30: poisoned | [0.60363007,1.25401509) | 33.79187% | 18.45% |

Here “near” allows the most favorable0.505–0.515 printed rounding and 1e-6
numerical guard. Thus MAE alone does not rescue the printed point. We withdraw
its promotion and do not select another training branch automatically.

The source also does not say “featurewise” scaling: III-A.1, p2677 specifies
zero mean/unit variance with training statistics reused for test data, but
does not specify the axis. Featurewise scaling remains our executable
interpretation. III-B.1(b), p2678 and III-D.2(b), p2680 leave the standalone
error norm/training loss unspecified. An MAE **score** does not entail an MAE
**training objective**. These choices must be separately declared.

Raw-unit MSE is another example of the cutoff distinction: at the printed
cutoff its optimistic p00 detection bound is 90.02976%, below 94.05%, despite
adequate FA. Of the five declared branches, only the min-max control is not
excluded at that cutoff in both conditions. This does not establish shared
model realizability or validate min-max as the paper's method.

### Three poisoned-case values used the unpoisoned FA cap

The repair README mistakenly used 5.25% instead of 18.45% for three p30 DR
upper bounds. The raw JSON had the correct cap-specific values throughout.

| Branch, p30 | Former reported DR bound at wrong cap | Correct bound at FA ≤ 18.45% |
|---|---:|---:|
| Feature-z MSE | 87.23214% | 100% |
| Raw-unit MSE | 92.14286% | 100% |
| Global-z MSE | 90.26786% | 96.81548% |

The [repair report](../aea_repair_20260925/README.md) is corrected; its original
result JSON, score archives and execution contracts are unchanged. Three
regression tests now check condition-specific caps and the fixed/free-cutoff
distinction against the preserved summary.

### The prototype's checks were weaker than their description

At original commit 4545e92, perturbing the encoder input changed attention by
5.21354e-6. The same test still passed with the decoder-query kernel zeroed
(change 4.03635e-6), because the intervention changed encoder memory too.
It therefore did not identify query dependence. This is a defect in the test,
not evidence the actual query connection was absent.

The original prototype could be saved but not reloaded by Keras. Its nested
custom layer lacked registration. We moved that layer to module scope,
registered it and serialized the initializer seed. The constructor now
requires an explicit loss choice instead of silently selecting MAE.

Six pinned TensorFlow 2.16.2/Keras 3.4.1 constructed tests pass. The revised query
test changes only the last returned encoder hidden state (the first decoder
query), leaving all encoder outputs/memory unchanged; disabling the query
kernel removes the first-step attention change. A fresh-process reload test
verifies exact weights, outputs, attention and optimizer iteration count.
The other tests cover 5,031,701 parameters, shape/range/normalization, finite
updates/constraints, and explicit loss identities. A separate before/after
comparison preserves all 24 weight arrays, outputs and attention exactly for
the original small seed 42 fixture.

These checks do not establish useful reconstruction learning, convergence,
target-data performance or timing. Tiny connected attention gradients in the
original probe were finite; that fact is not a learning guarantee.

## What held up

Both existing artifact auditors reproduced their preserved outputs exactly.
An independent implementation in `checks/aea_recheck.py` also checked:

- All ten score-box intervals using piecewise endpoint arithmetic instead of
  the original clipping implementation.
- Nominal/favorable AUC limits using direct benign–malicious pair counts,
  without the original ROC integration routine.
- All four saved FA-cap DR bounds using benign order statistics, plus
  necessary cutoff endpoints and the primary reversed-score relaxation.
- Fixed-cutoff FA/DR endpoints across original, synthetic and combined rows;
  original geometry bounds agree with the corresponding repair branch.

The original feature-z/MSE fixed-cutoff minimum FA remains22.15909%/22.36842%.
Even allowing **any** cutoff, its p00 optimistic detection upper bound remains
88.24405% at FA ≤ 5.25%, below the rounding-favorable target94.05%. The global-z
MSE p00 upper bound is92.26190%, also below target. Neither branch's oracle
AUC establishes an AUC exclusion. These results concern the frozen 20-customer
pilot, specified scoring and [0,1] reconstruction range—not another population,
unbounded outputs, the ensemble, or author intent.

The 108 exact p30 training/test overlaps remain explicit. No research fit,
cluster job, new score/scale branch or website update occurred in this audit.
The five direct reproduction files, original preparations and scientific
results remain untouched. The next decision is source justification of a
complete scaling/loss/score/cutoff interpretation, not training the previously
promoted MAE branch.

## Evidence and replay

Original geometry result SHA256:
`d6e8b877ce31bd6ccaccbfa9ed478fad9587665782a9192590243f58834a42e5`.
Original repair result SHA256:
`fd14a718560989e5575e7e1a8f4f5f153a17fc6fd023ccfdde3ee35ade9c7f69`.
Original prototype source SHA256 at 4545e92:
`2c3beac4dc92f9b4850898d22be71621f9270bb0629ce38dd5b360d0a5e6279d`.

The paper PDF SHA256 remains
`03a372fb5ee129799d43a50e6bec8f86fb108e9f8bdc85350e3ffe3e13d4a4ff`.
The relevant complete pages 2677–2680 were visually rechecked; all ten source
pages had been inspected for the earlier source specification. We did not
obtain reference[22]'s unavailable full text in this audit.

Saved audit outputs: [independent arithmetic](independent_arithmetic.json),
[original prototype defect probe](prototype_probe.json),
[geometry auditor](geometry_audit.json), [repair auditor](repair_audit.json).

```bash
.venv/bin/python studies/takiddin-2021-robust-poisoning/checks/aea_recheck.py \
  --geometry data/derived/takiddin-2021-robust-poisoning/aea-geometry-20260924-attempt1 \
  --repair data/derived/takiddin-2021-robust-poisoning/aea-repair-20260925-attempt1
.venv/bin/python -m unittest tests.test_robust_aea_recheck -v
KERAS_BACKEND=tensorflow TF_ENABLE_ONEDNN_OPTS=0 \
  tmp/robust-feedforward-venv/bin/python -m unittest tests.test_robust_aea_model -v
```

The replay requires the preserved local archives; it accepts only the original
result hashes and does not fit or generate research scores. The neural tests
use constructed arrays only. The failing original probe is retained as a
historical observation, not rewritten to describe the repaired implementation.

Final verification: repository suite 392 passed / 23 environment-specific
skips (415 cases); six isolated pinned TensorFlow AEA tests passed. Strict
data verification succeeds using the already recorded ScienceDB semantic-
equivalence branch. Restricted official CER copies are absent locally; no
new proxy was substituted. `git diff --check` passes. Website consistency was
checked read-only; no journal rendering or publication occurred.
