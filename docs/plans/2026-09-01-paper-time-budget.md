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
2. **Complete:** inspect contemporaneous primary infrastructure records. Texas
   A&M exposed K80 and V100 resources during the relevant period. The paper
   does not identify which resource, if any, it used. The fairest favorable
   single-device proxy therefore uses one V100, never multiple GPUs, H200, or
   A100.
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
5. **Complete:** the frozen runner passed 261 repository tests and strict
   data verification, was committed as `46f0ddd`, and was synchronized to a
   clean Panther checkout. The single authorized execution is Slurm job
   `385632`, started 2026-09-01 15:20:59 Asia/Qatar on one V100-16GB. Its
   configuration artifact passed the initial identity audit. The job completed
   `0:0`; every artifact was transferred and independently audited. See
   `PAPER_TIME_BUDGET_FINDING.md`.
6. **Complete:** after discussing the audited result, publish the full metric
   row, exhaustive cutoff envelope, measured V100 runtime, plausible
   explanations, and conclusion boundary in the current HTML report, homepage,
   and README. The older PDF is labeled as ending on 1 September rather than
   presented as current. All 265 deterministic tests, strict data verification,
   static links, and source-reference checks pass. No new scientific result was
   generated for this step.
7. **Complete:** the required discussion occurred and the user authorized this
   bounded publication. Stop after deployment to choose the next scientific
   question. Do not add a GPU, seed, model, table, or longer budget.

The intended conclusion is bounded: whether the reported result arises from
this declared completion within the paper's reported training time on one
plausible contemporaneous GPU. It is not a universal impossibility or a claim
about author intent.
