# Test the LSTM-SAE inside the paper's reported training time

**Date:** 2026-09-01

**User direction:** do not obtain faster or additional GPUs. Treat the paper's
reported time as part of the claim, infer a reasonable contemporaneous hardware
envelope, explain it publicly, and test what the written LSTM-SAE can attain
inside that envelope.

1. **Complete:** re-read the source timing and implementation statements. Table
   IV reports 183 minutes for full-ISET LSTM-SAE training. The paper names
   Keras Sequential but no hardware, GPU count, epoch count, batch size,
   version, stopping rule, timing boundary, or repetitions.
2. **Complete:** inspect contemporaneous primary infrastructure records. The
   plausible institutional range includes K80 and V100 GPUs. An official Raad2
   guide contains an account prompt closely matching the first author, and the
   Raad2 GPU system uses V100s. This is evidence of access, not proof of the
   unpublished run's device. The fairest single-device test therefore uses one
   V100, never multiple GPUs, H200, or A100.
3. **Complete:** calculate the already measured A16 boundary. Scaling the
   slower 218-second, 32,768-row pilot epoch to 1,500,523 rows gives 166.379
   minutes per epoch. The paper's 183-minute budget accommodates about 1.10
   projected A16 epochs, before full scoring. Thus the declared ten-epoch
   completion cannot fit in the paper's time on the measured A16. This is an
   exact runtime conclusion under the projection, not yet a metric result.
4. **Complete:** freeze and implement one time-capped full-data LSTM-SAE
   attempt: exact clean-reader data and model, seed 20260824, batch 32, one
   V100-16GB, at most 183 minutes of fitting, followed by fresh saved-weight
   reload and complete scoring. Record complete and partial epochs, updates,
   timing, memory, all seven printed-cutoff metrics, both score directions, and
   the complete fixed-score threshold envelope.
5. **In progress:** the frozen runner passed 261 repository tests and strict
   data verification. Commit and sync the frozen attempt, run it once, then
   transfer and audit every artifact.
6. **Pending:** update the current website, README, status, context, evidence
   register, and explanation register. Publish only claims earned by the run.
7. **Pending:** stop for discussion. Do not add a GPU, seed, model, table, or
   longer budget.

The intended conclusion is bounded: whether the reported result arises from
this declared completion within the paper's reported training time on one
plausible contemporaneous GPU. It is not a universal impossibility or a claim
about author intent.
