# Implement and validate the sequential detector

Date: 2026-09-27. State: in progress.

The user approved continuing after the source/design checkpoint a351652.
Implement I-SEQ-native-IVC and execute its constructed software/learning
gates. Only if they pass, proceed to the specified constructed GPU timing
gate. Research-data training remains a separate checkpoint.

1. Add the actual model to the direct `reproduction/models.py` implementation;
   retain the old standalone prototype and all existing model functions.
2. Verify full dimensions/settings, deterministic initialization, BCE,
   decoder query/feedback interventions with disabled-path negative controls,
   finite updates/constraints and fresh-process persistence.
3. Run the frozen reduced-width learning pair: model seed20260920, synthetic
   data seed20260926,32 training/32 test profiles,8 steps,300 updates and a
   five-minute guard per fit. Preserve both normal/reversed-label outcomes,
   even when they fail. No seed, activation, width or budget search.
4. If any required gate fails, diagnose the failed instrument cheaply and
   report it before GPU timing. Do not change the frozen interpretation to
   manufacture a pass. Otherwise freeze code and run the authorized
   constructed20-minute GPU timing allocation, without research inputs.
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
