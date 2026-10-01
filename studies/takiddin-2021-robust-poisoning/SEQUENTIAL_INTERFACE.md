# Decoder-to-GRU interface: source limits and a finite comparison

Recorded 2026-10-01 before the new implementations or learning results.
The user approved source clarification, a minimal separately documented
comparison and conditional continuation toward the bounded ensemble pilot.
The old scalar-ReLU completion and all its failures remain preserved.

## What the source identifies

The target PDF remains SHA256
`03a372fb5ee129799d43a50e6bec8f86fb108e9f8bdc85350e3ffe3e13d4a4ff`.
The complete paper was previously read; complete pages2678,2680,2682,2683
were re-read and visually checked for this decision.

| Location | What it supplies | What it does not settle |
|---|---|---|
| Fig.2 and III-B.1(b), p2678 | Decoder output is called reconstructed output; it feeds the GRUs and is fed back with attention context into the decoder | Scalar readout equation, output width and activation are absent |
| TableII, p2680 | Standalone AEA output activation is Sigmoid | Whether the ensemble preserves that component output setting |
| IV-B, p2682 | The AEA output passes to GRU layers; all components train jointly on classification cross-entropy | No reconstruction loss or separate pretraining; no readout equation |
| IV-C, p2682 | Encoder500/300/200, decoder200/300/500, eight300-unit GRUs, Dense500; Adam, ReLU hidden activation, Sigmoid output, no dropout, constraint1 | Scope of the intermediate output activation; this could refer only to the final classifier output |
| Algorithm1 lines23,35-37, p2683 | Feedback uses x_A; x_A denotes the reconstruction and becomes the first GRU input | No explicit mapping from the final500-wide decoder hidden state to x_A |

Thus the paper cannot uniquely resolve the bridge. A per-time scalar readout
is a reasonable completion of reconstruction under the declared48-step,
one-reading-per-step layout. It is not an explicit printed tensor shape.
Passing the whole decoder-state vector is another possible representation
reading, but would change GRU input width and feedback dimensionality and
would not reconstruct the original scalar without a further map. It stays
open; it is not added to this minimal activation comparison.

## Two choices declared before outputs

1. **I-SEQ-scalar-sigmoid:** retain the AEA component's explicitly reported
   Sigmoid output setting from TableII while applying IV-C's ReLU setting to
   recurrent hidden layers and the classifier hidden layer. The scalar
   projection has one Sigmoid unit per time step. This is textually supported
   but not uniquely required. It preserves9,240,802 parameters and the same
   affine readout weights as the failed ReLU version.
2. **C-SEQ-scalar-linear:** keep the same scalar affine readout without an
   activation. A linear readout can represent signed standardized values and
   tests the clipping explanation. It is a separately labeled control; it is
   not selected as the paper's method if it gives a favorable result.

No other architecture, optimizer, loss, norm, seed or scalar scale is changed.
The activation affects **both decoder feedback and the downstream GRU input**;
this is not an isolated causal intervention on only the GRU's input.
Both models optimize the same repaired BCE and retain the final Sigmoid
classifier. Neither adds reconstruction training or a novelty cutoff.

The original ReLU default and deserialization behavior must remain exact.
All three readouts share identical initial affine/recurrent weights at the
same dimensions and seed; branch identity is stored explicitly in config.

## Fixed software comparison and decision

Use the same32 training/32 fresh test profiles,8 steps, seeds20260920/20260926,
normal/reversed labels and full widths as the September28 check. Four fixed
fits: Sigmoid-normal, Sigmoid-reversed, linear-normal, linear-reversed.
Exactly300 full-batch updates per completed fit, with300-second fit guards
(20 CPU-minutes maximum scheduled fitting plus bounded audit overhead).
The earlier equal-size fits measured about105s each; this guard is declared
before the new outcomes, and no outcome-based extension is permitted.

Both orientations of a branch must complete, reach at least90% held-out
accuracy and BCE<log(2)/2, preserve paired initial weights, finite connected
gradients and valid constraints/persistence. Retain failures, not just the
better branch. No extra seed, early best-epoch selection or branch addition.
Software-only local CPU runs; no CER observations are loaded.

Sigmoid is the candidate for the next numerical stage **before** results:
if it passes, proceed to full48-step constructed V100 timing and then the
approved fixed research pair only after all software/runtime/artifact gates
pass, within the20-minute timing and180-minute pair ceilings. Linear results
remain control evidence; if only linear passes, do not promote it automatically.
If neither passes, report the unresolved validation failure without another
setting change. No claim about the paper's full population or mechanism
follows from this constructed comparison.

The earlier complete-model and source-cell ambiguities remain in
[SEQUENTIAL_ENSEMBLE_PILOT.md](SEQUENTIAL_ENSEMBLE_PILOT.md). That historical
contract is not rewritten. This addendum identifies the new branch and its
selection rule explicitly; subsequent launch records must identify it.
