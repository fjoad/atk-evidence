# Full48-step recurrent-cell comparison

The user approved one source-explicit tanh-cell alternative and constructed
validation under [the contract](../../SEQUENTIAL_RECURRENT_CONTROL.md).
Four fixed software fits compare it with the existing ReLU reference under
normal/reversed labels, with all initial weights/settings otherwise matched.
No CER inputs or research fits. No job yet at this source freeze.

Twenty relevant pinned neural checks pass;405 repository cases pass with45
environment skips. The initial new reload harness mishandled a nested output
list; outputs[0] corrects the harness and legacy/new numerical outputs match.
Both the failure log and passing checks are preserved. Website unchanged.
