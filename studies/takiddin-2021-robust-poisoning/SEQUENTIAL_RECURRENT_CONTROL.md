# Equation-led recurrent activation: one declared alternative

Recorded 2026-10-02 before implementation or new learning outputs. The user
approved defining one explicit alternative/control and validating full48-step
learning. This stage includes no CER fitting, scoring or data loading.

## Source decision

The complete paper was previously read. Complete printed pages2678,2679,2682
and2683 were re-read and visually checked for this decision. Original PDF
SHA256 remains03a372fb5ee129799d43a50e6bec8f86fb108e9f8bdc85350e3ffe3e13d4a4ff.

| Source | Consequence |
|---|---|
| III-B.1(b), Fig.2, p2678 | AEA uses recurrent encoder/decoder states and attention; scalar readout/feedback completion remains explicit. |
| III-B.2(e), item4, p2679 | GRU candidate explicitly uses tanh, followed by gated state mixing. |
| III-C, p2679; IV-C, p2682 | Hidden activation is tuned, and the ensemble selection is ReLU. Its scope conflicts with the explicit recurrent formulas. |
| Algorithm1 lines11/13 and30/32, p2683 | LSTM candidates and the cell-to-hidden transform explicitly use tanh. |
| Algorithm1 line42, p2683 | GRU candidate explicitly uses tanh. |
| IV-B/C, p2682 | Joint classification training, Adam, no dropout, constraint1, six LSTMs/eight GRUs/Dense500 are retained. |

Declare **I-SEQ-tanh-cells**: give the explicit recurrent tanh formulas
priority inside all six LSTM and eight GRU cells, retaining the ReLU Dense
hidden layer and scalar-Sigmoid intermediate/final outputs. Sigmoid gates
remain unchanged; this is not a change to Keras `recurrent_activation`.

This is an equation-led native-cell interpretation, not literal Algorithm1
or uniquely specified author code. Native cells still omit its peepholes,
per-GRU output projections, and unresolved ordering/readout details. The
existing IV-C-led ReLU interpretation and all failures remain intact.
Changing to tanh follows an explicit source alternative and tests the gate
findings; it is not evidence that the source unambiguously requires it.

## Fixed change and comparison

Add a serialized `cell_activation` option to the direct implementation,
limited to relu/tanh, with **relu as the unchanged historical default**.
For LSTMs this changes candidate activation and h=o*activation(c); for GRUs
it changes the candidate only. Gates stay Sigmoid. Keep scalar-Sigmoid
bridge, attention, state mapping, dimensions, parameter count9,240,802,
Adam.001, repaired BCE, MaxNorm1 coverage, no dropout, float32, deterministic
settings, native execution (`use_cudnn=False`), and all initializers fixed.
No speed-driven kernel change, clipping, normalization, auxiliary loss,
pretraining, pooling or seed/optimizer alteration.

One new alternative plus the existing ReLU reference, each under normal and
reversed labels: four predeclared software fits. Order: tanh-normal,
tanh-reversed, relu-normal, relu-reversed. The alternative is selected for
validation before outputs, not from whichever score is best. No extra branch
or seed follows a failure.

## Full-length constructed gate

Use32 training and32 independent test profiles,48 steps, model seed20260920
and NumPy seed20260926. Labels comprise16 zeros then16 ones. As in the
previous fixture, x=.75*(2*y-1)+.05*Normal(0,1), now drawing complete48-step
profiles rather than tiling the old eight-step data. Reverse only labels
for the second orientation. The sign-of-daily-mean rule (reversed when
labels reverse) is the zero-parameter floor; the balanced constant prior
has50% accuracy/BCE log(2). This is a trivial instrument check, not temporal
capability, reconstruction quality or poisoning robustness evidence.

Exactly300 full-batch updates, batch32, final weights only, per completed
case. Same >=90% held-out accuracy and BCE<log(2)/2 criterion as before,
plus finite connected gradients, changed parameter groups, valid constraints,
paired initialization and exact save/reload. Preserve initial/final train/test
outputs, intermediate sequences, attention, layer ranges, histories, weights,
optimizer/configuration and all failures. All four initial hashes must match
and equal the prior GPU initialization11c3bf3746800eebd9e695a9dda3abaafc9e7c445cbe8044a2110ca4da77f549.

The new alternative qualifies only if BOTH orientations complete300 updates
and pass every check. A passing ReLU reference does not invalidate its earlier
CER failure; a passing tanh alternative does not establish a successful CER
repair or identify the paper's intended implementation. If both pass, no
exclusive causal advantage is inferred. If tanh fails, stop promotion.

## Budget and boundary

One Panther gpu-short allocation, oneV100-16GB,4CPUs/16GiB,35min ceiling.
Each fit has a360s guard checked at update/epoch boundaries:24min nominal
fitting guards plus single-update check granularity, with11min reserved for
checks, tracing/startup, persistence/audit.
The earlier full48-step V100 measurement was about37s/45 batches of100;
this is a planning reference, not a measured runtime for batch32/tanh.
No guard extension or duplicate submission. Continue after a completed
low-learning result to finish the fixed comparison, but stop on a partial
fit, nonfinite training or execution/provenance/persistence failure and
preserve everything. No automatic retry or missing-case expansion.

Small software/parity/serialization tests run locally. All full-width48-step
learning runs on Panther. Freeze source/tests/contract before submission and
inspect scheduler/output state first. Afterward copy and audit artifacts,
report all four outcomes (or the exact partial stopping point), update the
record and commit. **No CER fit or follow-up job is authorized by passing
this constructed gate.** A later research pair needs its own explicit
runtime/experiment contract. Website and previous studies stay untouched.
