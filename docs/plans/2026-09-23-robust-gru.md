# GRU baseline: source, constructed checks, bounded pair

The user approved continuing the research sequence toward the ensemble.
This step covers the GRU baseline, not an automatic sweep. The website is
owned by another session: do not edit or regenerate site pages here.

- [x] Recheck target pp.2679-2681 and read/visually inspect all six pages of
  reference [26], Nabil et al., arXiv:1809.01774 / ICPR2018.
- [x] Freeze the source interpretation, omissions, versions, metrics, inputs,
  runtime gate and stopping rule in GRU_PILOT.md before research outcomes.
- [x] Add the direct model/fit path and constructed fixtures, preserving
  historical source revisions and results.
- [x] Freeze code; run a ten-minute, one-V100 constructed timing preflight
  with no research inputs. Check the predeclared gate before dispatching fits.
- [ ] With the explicit runtime exception below, run only the original p00/p30 pair, same seed,
  50 epochs/batch100 within the bounded allocation. Audit the saved artifacts.
- [ ] Record outcome, limits and the next scientific decision. No publication
  or website work. Do not add seeds/settings in response to the result.

Keep GRU numerical coverage separate from a demonstration of temporal
mechanism. AEA/ARIMA and the two ensembles remain subsequent source-specified
steps; a GRU result does not transfer to them or to another paper.

Source review selected the explicitly labeled native-Keras/table interpretation,
48 time steps by one reading, two Softmax outputs and standard CE repair. The
generic tanh/V-projection notation and missing tensor-shape call remain open
alternatives. The existing neural environment and idle V100 resources are
available on Panther. An initial SSH login timed out before authentication;
the immediate authorized retry succeeded. No job has yet been submitted.

Constructed discovery: the full 8x300 ReLU model can fit two simple constant
sequence classes. Repeated train_on_batch calls emitted retracing warnings,
so the timing instrument uses the same single model.fit/dataset path as the
research runner. This correction precedes hardware timing and any real-data
outcome; no method setting changed. Preserve operational/fixture failures as
software evidence, not paper results.

Thirteen GRU fixtures and all eight existing feed-forward fixtures pass in
the pinned TensorFlow environment. The full 8x300 model fits a simple two-
sequence classification fixture. The earlier feed-forward artifact audit
still matches its saved file byte-for-byte. Main-suite verification totals
352 passes and 17 environment-specific skips (140 study tests plus 229 root
tests); all 21 neural tests passed separately. Strict data verification passes
through the existing verified source branch. No GPU timing or research fit
has run yet.

Scientific contract/code freeze:
`2d706b103ee03cc705cc3ef3f07718bc3ed7792a`. Subsequent status updates do not
change that execution revision. The isolated Panther checkout is
`/export/home/fjoad/atk-evidence-paper3-gru-20260923`; the older main checkout
and original baseline artifacts remain untouched. Existing neural runtime is
reused; no dependency installation is required. All seven AdaBoost fixtures
also passed in their existing isolated environment.

Constructed-only preflight submitted once as job **402376**, output
`gru-preflight-20260923-attempt1`, using the frozen revision above. Research
pair remains unsubmitted until its recorded runtime gate passes. Initial
SSH/session interruptions occurred before submission; no duplicate job.

Preflight completed 0:0 in 2:33 on one V100-PCIE-16GB. All 13 fixtures pass
on GPU; full-model timing gives warmed updates 0.33034-0.34608s. The frozen
slowest-step projection is 778.68199s per 2,250-update fit, above the 720s
launch gate. The verifier correctly blocks real-data loading. Transfer/source/
GPU/norm checks pass, and no research pair was submitted. Result record:
`results/gru_preflight_20260923`.

The user was asked whether to permit an explicit launch-gate exception for
the unchanged pair under the existing 40-minute allocation and 15-minute
fit guards. No response has been received at this checkpoint; do not treat
the gate as passed or start the pair without that decision. Other model
coverage has not begun. Website and earlier evidence remain untouched.

## Authorized continuation

The user has now explicitly approved proceeding with the unchanged model.
GRU_RUNTIME_EXCEPTION.md binds the one existing preflight and raises only the
launch ceiling to 900 seconds. Preserve the original false 720-second gate;
the 40-minute allocation and 900-second fit guards are unchanged. Add focused
authorization/drift checks, freeze the launcher revision, check scheduler and
output state, then submit the pair once and audit it. No new research data
preparation or repeated timing job is needed.

Resume checks pass: all 16 GRU and 8 feed-forward fixtures in the pinned neural
environment; 355 main-suite passes with 17 environment-specific skips; strict
data verification passes the existing source branch. The nine non-neural GRU
checks include refusal of altered identity/scientific sources and projections
over 900s. The approved record passes only the explicit exception path; the
default verifier still refuses it. Scheduler is empty and no GRU pilot output
exists before dispatch. No computational source or original contract changed.

Launcher/exception frozen at `46966c7e021e964ff582e1eb56422b75cc6a4ec7`.
Submitted once as job **402378**, running on crimv3mgpu005 with the exact
40-minute cap, one V100-16GB, four CPUs and 16 GiB. The cluster authorization
checks pass before its constructed tests. Output is
`gru-pilot-20260923-attempt1`; see `results/gru_pilot_20260923` for exact paths.
Do not resubmit. The queued script proceeds through both conditions and audits
without another launch decision, and stops on partial/failure. No completed
research outcome at this checkpoint.

All 16 cluster fixtures passed in 100.295s, local launch-authorization audit
matches cluster bytes, and the p00 model completed its first real-data epoch
(45 updates, finite loss) with initial weights/history preserved. The job
continues; this is a running checkpoint, not a completed research result.

## Partial-result check, September 24

The user asked to check the existing job. Slurm reports FAILED/2:0 after
22:45, matching the intentional partial-fit exit: p00 hit its 900s guard
after 32 full epochs plus seven batches of epoch 33. P30 never started.
Preserve/copy the artifacts and log, verify hashes and saved-score arithmetic
without fitting or running model inference, record the partial outcome and
next decision, then stop. Do not alter the original result's partial status
or loosen the completed-pair auditor. No new allocation, website work or
automatic runtime extension is included in this check.

Read-only partial audit passed: ten consumed inputs, all source/output hashes,
test identities, probability normalization and full saved-score analysis,
initial/final weight hashes, 4,058,702 parameters, serialized optimizer count,
model config (except process-local object IDs), kernel norms and partial
history. Original status remains partial and the completed auditor refuses it.
All artifacts/logs copied; no model inference locally. DR42.99/FA8.78/AUC80.82;
at FA<=6.8 best DR35.75 versus92.4. Loss still improves and p30 is untested.
Stop; next decision is timing calibration and an approved adequate budget for
the unchanged 50-epoch question. No new allocation or automatic continuation.

Final verification: main suite 355 passes/17 environment-specific skips;
strict source-data verification passes; copied original result/history bytes
match. Scientific implementation and original GRU contract are unchanged.
The website consistency check passes without regenerating any page.
