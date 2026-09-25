# AEA experimental implementation fixture

The earlier promotion of feature-z MAE as the “least invasive surviving”
threshold-preserving interpretation is **withdrawn**. It passes an optimistic
free-cutoff relaxation, but its minimum FA at the printed 0.51 cutoff is
36.75239%/33.79187%, even allowing favorable rounding. The paper explicitly
specifies zero-mean/unit-variance scaling, not its featurewise axis. Neither
the score comparison nor the paper selects MAE as a training loss.
See the [audit](results/aea_recheck_20260925/README.md).

`checks/aea_model.py` is an experimental constructed prototype, not a chosen
research branch. It has no CER loader and is not connected to the five direct
reproduction files. It implements the proposed native/table topology from
AEA_SPECIFICATION.md: three Sigmoid LSTM encoder cells (500,300,200), mirrored
decoder cells (200,300,500), no peepholes; additive tanh attention with a
decoder query; mirrored state bridge; zero first reconstruction, free-running
feedback and one Sigmoid output per step; SGD .01, no dropout, MaxNorm 1 on
declared kernels and seeded initializers. Its training loss must now be
chosen explicitly; MAE and MSE are separate software options, neither inferred
from a score bound.

The original four tests did not justify their “query dependence” claim:
changing the encoder input also changed memory, so the test passed even with
the decoder-query connection disabled. Saving succeeded but fresh model
loading failed because the nested custom layer was not registered. Those
failures are preserved in the audit, not erased by the fix.

The custom layer is now serializable at module scope, with its initializer
seed in the configuration. Six constructed tests pass in the pinned
TensorFlow 2.16.2/Keras 3.4.1 environment:

- Full parameter inventory (5,031,701) and input/output shapes.
- Finite bounded output and attention normalization.
- An intervention on the first decoder query with encoder memory fixed;
  disabling the query kernel removes the effect (negative control).
- Finite small updates, changed weights, optimizer count, constraints and
  rejection of the wrong input shape.
- Separate MAE/MSE loss identities and rejection of an omitted loss choice.
- Fresh-process save/reload with identical weights, predictions, attention
  and optimizer iteration count.

The serialization refactor preserves the original constructed initialization,
output and attention exactly. These are software witnesses, not evidence of
useful reconstruction, generalization, convergence or target-data timing.
Architecture alternatives and further learning/feedback fixtures remain open.
No research fit, selected repair branch, ensemble, seed sweep, website edit
or publication follows. Any experiment requires its own source justification
and frozen data/metric/budget/stopping contract.
