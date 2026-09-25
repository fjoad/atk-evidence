# AEA implementation fixture envelope

The repair comparison leaves feature-z MAE as the least invasive surviving
interpretation: preserve the paper's featurewise zero-mean/unit-variance
inputs, Sigmoid reconstruction head and threshold context, while changing only
the unspecified reconstruction-error norm from MSE to MAE. This is an
interpretive branch, not a claim about the authors' implementation.

The constructed implementation is `checks/aea_model.py`; it is not connected
to the five paper-facing reproduction files and cannot load CER data. It makes
the missing choices in AEA_SPECIFICATION.md explicit: three native Sigmoid
LSTM encoder cells (500,300,200) and mirrored decoder cells (200,300,500), no
peepholes; additive tanh attention with a decoder query over 48 positions;
mirrored encoder state bridge, zero first reconstruction and free-running
decoder feedback; one Sigmoid reconstruction per time step; MAE objective;
SGD .01; no dropout; MaxNorm 1 on declared kernels; seeded initializers.

Four constructed tests pass in the pinned TensorFlow 2.16.2/Keras 3.4.1
environment: full parameter inventory and shapes (5,031,701), output range/
attention normalization/query dependence, finite small-fixture updates and
separate MAE/MSE objective identities.

This is a capability fixture. It does not establish data performance, timing,
convergence or reconstruction quality, and it does not authorize a research
fit. Exact cell equations, attention state scheduling and other open
alternatives remain separate branches. Before any fit, decide whether the
paper supports changing MSE to MAE and freeze a new data/metric/budget contract.
No branch search, ensemble fit, extra seed or website publication follows.
