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

## Live execution

Submitted once as408764, frozen `6ca174a8e58cd47ccb5af656f1cbfa534c29a156`.
OneV10016GB/4CPUs/16GiB,35min ceiling,360s/fit; no CER data. Check this
existing job/output before any continuation. Four cases remain fixed in
advance; no duplicate or automatic research promotion. Log:
`/export/home/fjoad/seq-cells-transfer-20261002/cells-408764.out`.
