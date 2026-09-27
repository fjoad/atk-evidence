# Implement and validate the sequential detector

Date: 2026-09-27. State: implementation complete; constructed learning gate failed.

The user approved continuing after the source/design checkpoint a351652.
Implement I-SEQ-native-IVC and execute its constructed software/learning
gates. Only if they pass, proceed to the specified constructed GPU timing
gate. Research-data training remains a separate checkpoint.

1. Add the actual model to the direct `reproduction/models.py` implementation;
   retain the old standalone prototype and all existing model functions.
2. Verify full dimensions/settings, deterministic initialization, BCE,
   decoder query/feedback interventions with disabled-path negative controls,
   finite updates/constraints and fresh-process persistence.
3. Run the frozen reduced-width learning pair: model seed 20260920, synthetic
   data seed 20260926,32 training/32 test profiles,8 steps,300 updates and a
   five-minute guard per fit. Preserve both normal/reversed-label outcomes,
   even when they fail. No seed, activation, width or budget search.
4. If any required gate fails, diagnose the failed instrument cheaply and
   report it before GPU timing. Do not change the frozen interpretation to
   manufacture a pass. Otherwise freeze code and run the authorized
   constructed 20-minute GPU timing allocation, without research inputs.
5. Verify, record the outcome and next decision, update handoff and commit.

Implementation details fixed before the first output: explicit Glorot seeds
base+i for encoder kernels, base+10+i recurrent, base+20+i decoder kernels,
base+30+i decoder recurrent, base+40 intermediate projection, base+50/51/52
attention; GRU kernels base+100+i and recurrent base+200+i; dense head
base+300/301. These complete the previously unspecified per-layer seed
mapping. Use native LSTM encoder sequence execution and a differentiable
decoder loop with the specified causal operations. These do not add layers
or change the declared mathematical recurrence.

The custom layer is defined/registered lazily inside models.py so classical
detectors remain usable without TensorFlow. A fresh reload process explicitly
registers that layer before Keras deserialization. The serialized config must
contain widths, steps and seeds. There is no sixth hidden model module.

No website work, real-data scoring, alternate source branch, extra seed or
research fit is included. A failed constructed task is not paper evidence.

## Bounded follow-up after the constructed learning gate

The normal fit completed 300 updates, then report serialization failed on a
NumPy boolean. Its saved final model and predictions were recovered without
refitting after a writer-only repair; the original failed attempt remains.
The reversed fit then completed 300 updates. Normal passed (100% accuracy,
BCE .2253); reversed failed (50%, BCE log(2)). Do not launch GPU timing.

Before interpretation, read only these saved constructed models: verify all
artifact hashes and reload outputs, inspect intermediate and each downstream
layer's output range/variation, and record the intermediate ReLU's raw
preactivations. As a capacity sanity check compute the normal model's
complemented probabilities against reversed labels; this uses the identity
sigmoid(-z)=1-sigmoid(z), not another fit. No optimizer update, changed
architecture, seed retry or research-data observation is added.

## Completed checkpoint

The full direct model builds with 9,240,802 parameters and actual 48-step
forward outputs. All 9 ensemble/gate software tests and 6 prior AEA tests
pass in the pinned runtime. Both declared small learning fits completed 300
updates from identical weights. Normal: 100% accuracy/BCE .22528521; reversed:
50%/BCE .69314718. The gate fails and GPU timing was not attempted.

The saved-state audit finds identically zero last-GRU and classifier-hidden
outputs for the reversed test rows, while the AEA intermediate remains
positive. Complementing the successful model's probabilities solves reversed
labels without training. This demonstrates representational capacity, not a
successful reversed training run or a cause of the collapse. It gives no
result for full-width training or CER data.

The normal report-writing error was fixed without repeating its fit; both
original and recovered records are preserved. Two read-only audit-helper
errors were also corrected and recorded without optimizer updates. See the
[complete record](../../studies/takiddin-2021-robust-poisoning/results/sequential_constructed_20260927/README.md).

Verification: 394 repository tests passed/30 environment skips; 15 pinned
neural checks passed. Strict data verification and read-only journal
consistency pass. Existing model functions are AST-identical; original
standalone code and previous results remain unchanged. No cluster job,
research fit/score, alternate branch or website work. Next: specify a bounded
diagnosis of the failed reversed-label trajectory/reduced-width instrument
before considering full-model GPU timing.
