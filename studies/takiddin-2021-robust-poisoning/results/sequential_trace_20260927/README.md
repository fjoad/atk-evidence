# The constructed reversed-label run collapses after update 2

The fixed replay locates the failure after the **second Adam update**. The
last GRU and classifier hidden layer become entirely zero on both the 32
training and 32 held-out constructed profiles. Earlier GRUs and the AEA
intermediate remain active. Classification gradients to the preceding
network then become zero.

This is a local diagnosis of the 5,082-parameter software fixture, not a
CER/ISET result or evidence that the full 9,240,802-parameter model fails.
The original learning gate remains failed. No GPU timing, research-data
training/scoring, new seed, activation change or website work occurred.

## Replay contract and fidelity

The [plan](../../../../docs/plans/2026-09-27-sequential-collapse-trace.md)
and direct trace script were frozen at **9368a90 before replay**. One
reversed-label prefix was replayed for 50 updates, using the original
32-example full batch, same seed/model/optimizer and `model.fit` path.
There was a 300-second guard; execution including observations took 12.37s.

The initial weights match the original fit exactly. All 50 reported training
losses equal the preserved original history, and observation did not change
any model or optimizer values. The original data/model/result files remain
byte-identical. We are not claiming an independently repeated successful
fit or comparison of full 300-update weight trajectories.

Weights, optimizer variables and train/test activations are preserved at
updates 0 through 50 in the ignored
`data/derived/takiddin-2021-robust-poisoning/sequential-trace-20260927-attempt1`.
[result.json](result.json) identifies source and artifact hashes;
[trace.jsonl](trace.jsonl) preserves all observations. Its compact statistic
columns are minimum, maximum, standard deviation and positive fraction.
Global standard deviation across features is not a measure of between-example
information; the zero-output and per-row probability statements below are
checked directly against the saved arrays.

## What changes during the first two updates

Fractions below count positive activations on the constructed training batch.

| Completed updates | Positive final-GRU entries | Positive hidden-classifier entries | Gradient reaching the earlier network |
|---|---:|---:|---|
| 0 | 50% | 50% | Nonzero |
| 1 | 37.5% | 31.25% | Nonzero |
| 2 | 0% | 0% | Zero |
| 3-50 | 0% | 0% | Zero |

The same zero transition occurs on the held-out constructed profiles.
There is no earlier complete zero output in the AEA intermediate or GRU
layers 1-7 during this prefix. Both last stages go inactive within the same
optimizer update; the trace does not establish that one preceded the other
within that update.

After update 1 the float32 final probabilities are already identical across
the training examples, but upstream gradients still exist. Equal rounded
probabilities must not be mistaken for the later zero-gradient condition.
After update 2 only the output classifier's bias has a nonzero gradient.
Later output-bias movement changes the common prediction, not its ability
to distinguish these balanced classes.

## Local saved-state checks identify bias-driven shutoff

The classifier-hidden affine comparison was specified before replay. Using
its state immediately before/after update 2:

- Updated head parameters produce an all-zero ReLU output even when supplied
  with the previous nonzero GRU inputs.
- Previous head parameters still produce some positive output with the new
  zero GRU inputs, because some previous biases were positive.
- With the new inputs/weights, retaining the old bias restores positive head
  activations. Retaining only the old weights does not. The additive
  input/weight/bias change identity closes to 1.09e-19 numerical error.

These are local activation statements, not a recovered classifier or proof
that restoring a bias would produce a successful learning trajectory.

After observing the simultaneous step 2 transition, a separate **read-only**
GRU8 input/parameter check was specified in the plan. It used only the two
saved states, six block forwards and no optimizer update:

| GRU8 input and parameters | Positive final-state entries |
|---|---:|
| Previous input, previous parameters | 37.5% |
| Previous input, updated parameters | 0% |
| Updated input, previous parameters | 12.5% |
| Updated input, updated parameters | 0% |
| Updated input/parameters, previous candidate bias only | 12.5% |
| Previous input/parameters, updated candidate bias only | 0% |

The candidate-bias change alone is sufficient for this local GRU shutoff
with the old inputs/other parameters, and reverting it restores some
activation with the new inputs/other parameters. With all updated values,
every candidate preactivation in the 8-step sequence is negative: maximum
-3.99923e-5. ReLU clips those candidates to zero; a zero initial hidden state
then remains zero under the GRU recurrence. This independently explains the
native block's zero output on these fixed inputs.

The largest downward candidate-bias change is approximately 7.10e-4, compared
with the preceding state's maximum positive candidate preactivation of
6.63e-5. The head's largest downward bias change is 7.08e-4. These changes
overwhelm the small positive signals in this particular reduced-width
trajectory. This does not identify which change alone explains all later
training behavior, or select a new learning rate, activation or initialization.

An independent float64 reset-before GRU calculation matches all six native
forwards within 8.93e-12. The two unchanged endpoint combinations also match
the saved network states. See [artifact_audit.json](artifact_audit.json) and
the [read-only audit script](audit_saved_trace.py).

## Verification and limits

All 51 snapshots and original-file hashes verify. Fresh forward checks at
updates 0, 1, 2, 50 reproduce saved intermediate/final values exactly; independently
computed gradient zero patterns agree. Metrics are recalculated from all
saved train/test probabilities, and all 50 original losses match exactly.

Three constructed checker tests distinguish initial, transient and persistent
zero outputs; distinguish bias versus input collapse; and reject nonfinite
observations. The repository suite passes 397 tests with 30 environment skips
(427 cases); strict data verification passes the existing ScienceDB branch.
Journal consistency passes read-only. Scientific model/training code and
the old learning results are unchanged.

The failure now has a specific local explanation: early bias updates shut
off ReLU paths in the last GRU and classifier. It is not evidence that the
paper's entire architecture lacks capacity; the earlier complemented-score
witness already represented the reversed answer. Nor does this prefix prove
an infinite-time absorbing state.

The next useful decision is whether the tiny-width instrument created an
unrepresentative failure. A separately bounded constructed check at the
published widths, retaining the stated optimizer and other settings, can
address that before choosing an optimizer/activation repair or more seeds.
That check has not run; this trace does not waive the failed learning gate.
