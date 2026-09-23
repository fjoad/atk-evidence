# GRU pair — execution in progress

The user approved the explicit runtime-gate exception. Job **402378** was
submitted once with launcher revision
`46966c7e021e964ff582e1eb56422b75cc6a4ec7`. The scientific files remain
byte-identical to `2d706b103ee03cc705cc3ef3f07718bc3ed7792a`.

This is the original 0%/30% poisoning pilot pair, not a new preparation or
parameter trial. It retains the 8x300 ReLU GRU, 48x1 input, repaired two-class
cross-entropy, seed 20260920, 50 epochs and batch 100. Full settings and source
ambiguities are in ../../GRU_PILOT.md; authorization is in
../../GRU_RUNTIME_EXCEPTION.md. The old 720-second gate remains false.

The allocated node is crimv3mgpu005: one V100-16GB, four CPUs, 16 GiB, a
40-minute wall limit, and unchanged 900-second batch-boundary fit guards.
The launcher saves partial outputs and stops on failure. It automatically
runs the second condition, comparison and artifact audit after a successful
first condition. Do not submit a duplicate while this allocation exists.

Before submission, no account jobs or GRU pilot output were present. All 16
GRU and 8 feed-forward local neural fixtures passed; the main suite had 355
passes and 17 environment-specific skips. Strict data verification passed
the existing CER ScienceDB source branch. The cluster independently verified
the authorization and current scientific-file hashes before fixture execution.
All 16 cluster fixtures then passed in 100.295s. Local re-verification of
launch_authorization.json matches the cluster bytes. The p00 research model
has completed its first full epoch (45 updates, finite loss 0.67069626), with
initial weights and history saved. This establishes that research training
started, not that the final model succeeds or matches the paper.

## Resume and audit

Check job 402378 and its output before any further action. Remote paths:

- Checkout: `/export/home/fjoad/atk-evidence-paper3-gru-approved-20260923`
- Output: `/export/home/fjoad/atk-evidence/data/derived/takiddin-2021-robust-poisoning/gru-pilot-20260923-attempt1`
- Log: `/export/home/fjoad/robust-gru-resume-transfer-jfMFay/slurm-402378.out`
- Preparation: original `setup-20260920-attempt1`, unchanged.

After termination, preserve the Slurm status/log and every output, including
partial or failed results. Copy full artifacts only to ignored local data;
commit small nonrestricted summaries. Re-run the read-only pair audit and
comparison locally and compare bytes with cluster output. Check all histories,
paired initial weights, source/input/output hashes, probability normalization,
reload equality and the 50-epoch/2,250-update completion status. Do not perform
new model inference on the local computer.

There is no completed GRU research result at this checkpoint. Do not interpret
timing, training progress or constructed fixtures as paper-performance evidence.
After the completed/partial result is audited, state the next named scientific
question before another experiment. No website edits or publication occurred.
