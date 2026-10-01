# Both declared interface choices pass the fixed software learning check

The source-supported scalar-Sigmoid reading and the separate scalar-linear
control both learn the normal and reversed synthetic labels. All four fits
finish300 updates at100% held-out accuracy and clipped-probability BCE about
1e-7. These are constructed eight-step software checks, not electricity-data
or poisoning results, and they do not establish a reconstruction mechanism.

The [source addendum](../../SEQUENTIAL_INTERFACE.md) and code were frozen at
73fe054 before fitting. Sigmoid was the numerical candidate before results;
the linear control was not selected by its performance. Only readout activation
changes; it affects both decoder feedback and the downstream GRU input.
All four initial weight hashes equal the preserved full-width ReLU hash.
The failed ReLU implementation/results remain intact and its default reload
behavior is preserved.

| Branch | Labels | Updates | Test accuracy | Test BCE |
|---|---|---:|---:|---:|
| I-SEQ-scalar-sigmoid | Normal | 300 | 100% | 0.00000010 |
| I-SEQ-scalar-sigmoid | Reversed | 300 | 100% | 0.00000010 |
| C-SEQ-scalar-linear | Normal | 300 | 100% | 0.00000010 |
| C-SEQ-scalar-linear | Reversed | 300 | 100% | 0.00000010 |

The numerical BCE floor reflects clipping probabilities to[1e-7,1-1e-7]
for independent reporting; training uses Keras's stable BCE and final
probabilities saturate at0/1. No checkpoint, extra seed or alternate training
setting was selected. The data remain the same32 train/32 fresh test arrays,
eight steps, model/data seeds20260920/20260926, full9,240,802 parameters,
Adam.001 and300-second per-fit guards. These checks ran locally as software
fixtures under the repository's stated exception; large timing/research
work proceeds on Panther.

All artifact/source hashes, initial/final weights, update counts, stored
labels and probabilities pass the fresh-process audit. Independently
calculated accuracy/BCE agree. Loaded configuration retains the selected
readout and reproduces intermediate outputs/attention exactly. A historical
ReLU model loaded with the extended class reproduces its old probability,
intermediate output and attention exactly. See [result.json](result.json),
[artifact_audit.json](artifact_audit.json) and [audit script](audit_saved_fits.py).

Eighteen pinned neural software tests passed. The initializer-compatibility
test was subsequently made portable: compare frozen old/new implementations
on the same CPU rather than require a Mac-specific hash on different hardware.
Its focused recheck passes and does not alter the scientific model. Historical
artifact hashes remain unchanged.

Sigmoid now qualifies for the approved full48-step GPU timing gate, not an
automatic successful paper reproduction. The20-minute V100 allocation will
measure three full variable-batch epochs on4,464 synthetic examples, including
the64-row remainder. A real-data pair still requires the timing and artifact
checks to pass. No GPU or research outcome is claimed by this learning record.
