# Recurrent gate arithmetic in the saved sequential models

Date: 2026-10-02. State: authorized; specified before new observations.
The user approved the proposed saved gate/cell check. No fitting, optimizer
step, altered network parameter, new seed, repaired model or website work.

## Question and authority

Which native recurrent operations account for the final p00 GRU8 attenuation
and final p30 encoder/decoder amplification identified by job408550? This is
an interpreted implementation mechanism diagnostic (I/M), not a reproduction
or model-family attainability result. It feeds explanation E34.

The complete paper and existing source reconstruction remain authoritative:
III-B/Fig.2 p2678, GRU description p2679, IV-B/C p2682 and Algorithm1 p2683.
The model's ReLU/native-cell completion is explicitly different from the
paper's tanh/peephole equations where they conflict. Source ambiguity remains
open; gate arithmetic verifies the declared implementation, not author code.
The exact pinned Keras3.4.1 LSTMCell and GRUCell implementation2 formulas are
read locally and hashed; runtime source hashes must agree. Scientific model
files remain unchanged at the original pair's3705bcca definitions.

## Fixed scope and measurements

Use the same first100 TEST rows saved by408550, in the same batch order and
float32 runtime. Test labels are unused. Three saved weight states: shared
initialization, finalp00, finalp30. This narrower forward-only observation
needs no repeated training-batch derivatives. Bind all15 pair artifacts and
all10 previous diagnostic artifacts before/after; preserve all originals.

For each state:

1. Replay all three encoder LSTMs and all three decoder LSTM cells through
   the48 steps, with unchanged attention, initial-state mapping and feedback.
   Compare manual one-step arithmetic with the native cell at every step;
   propagate native states. Check encoder sequences/final states against
   native layers and decoder output against the saved408550 outputs.
2. Replay GRU8 on the preserved native GRU7 sequence. Record update/reset
   gates, candidate preactivation/output, retained old-state contribution,
   injected candidate contribution and final state. Compare each native cell
   step and the final state with the saved native GRU8 output.
3. Decompose LSTM candidate input/recurrent/bias terms and c=f*c_old+i*g,
   h=o*ReLU(c). Decompose GRU candidate similarly and h=z*h_old+(1-z)*g.
   Report exact0/1 and near-saturation gate counts (<=1e-6/>=1-1e-6),
   candidate activity, retention/injection magnitude
   and state growth at every step. The GRU weighted-injection/product identity
   must reconstruct the final state in float64 within relative1e-4/absolute1e-30.
4. At the observed states only, calculate candidate preactivations with the
   bias term omitted and with the recurrent term omitted. These are fixed
   local algebraic decompositions, never fed into subsequent states. They
   do not test a repaired trajectory or successful parameter intervention.
5. Save batchwide per-step summaries. Preserve complete raw gate/term traces
   for fixed rows0/1 plus the row containing the largest final native hidden
   value for that cell (first-index tie break), chosen by this rule before
   outcomes. This witness rule explains extremes, not prevalence or accuracy.
   Preserve full native state sequences and GRU8 injection/retention arrays
   for all100 rows; use witness-only detailed LSTM gate arrays to bound storage.

Native replay must reproduce the previously saved top-encoder sequence,
decoder hidden sequence/bridge and GRU8 final state. If exact equality fails,
report its maximum error and stop; do not waive it after seeing the result.
Manual versus native one-step equations allow relative1e-5/absolute1e-30;
record errors even when passing. Hash model/optimizer states before/after.
Stop/preserve on nonfinite values, changed inputs/state or failed parity.

## Predictions and limits

If GRU candidates stop injecting while update gates remain below1, old state
can decay over later time steps; the saved product identity measures this
without assigning a different bias. If LSTM candidate recurrent terms grow
while input/output gates admit them, the nonnegative injection term can
amplify state even though the forget gate is bounded by1. Identify which
layers show each behavior; do not infer stability from kernel column norms.
A temporal sequence here is within a48-step forward pass at fixed weights,
not the chronology of50 training epochs. Neither a single causal training
update nor a successful repair is identified by endpoint arithmetic.

## Execution

First verify discriminating constructed controls: GRU candidate shutoff and
geometric retention, LSTM retention without injection, multiunit recurrent
growth despite unit column norms, and native full-sequence parity. Tiny
software fixtures may run locally. Research replay is on Panther only.

One10min gpu-short allocation, oneV10016GB/4CPUs/16GiB,7min internal guard.
Run the constructed checks before research loading, then only the fixed
three states. No extra branch/sample/seed or follow-up job follows a result.
Freeze/check queue/output and submit once. Copy/hash/audit all outputs,
replay saved witness arithmetic and summaries locally without model calls,
update the evidence/status/context, verify and commit. Stop before repair
selection or any training. User's go-ahead authorizes this bounded sequence.

## Freeze verification

Four pinned local gate controls pass (2.60s): native sequence/state parity,
exact GRU geometric retention, LSTM retention without injection, and explicit
multiunit ReLU LSTM amplification with unit recurrent-column norms. The
repository suite passes403 cases with41 environment-specific skips; strict
data verification passes the established ScienceDB semantic-equivalence
branch and journal consistency passes. Restricted official archives remain
unavailable locally. All five original scientific files remain unchanged.
No new research observation has run at this checkpoint.
