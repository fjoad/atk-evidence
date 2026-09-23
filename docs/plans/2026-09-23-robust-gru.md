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
- [ ] Freeze code; run a ten-minute, one-V100 constructed timing preflight
  with no research inputs. Check the predeclared gate before dispatching fits.
- [ ] If the gate passes, run only the original p00/p30 pair, same seed,
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
