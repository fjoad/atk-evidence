# AEA source decisions and the next discriminating experiment

Date: 2026-09-26. State: source review and experiment design complete.

The user asked to carry out the next steps after the status review: justify
scaling, training loss, anomaly score and cutoff separately, then define a
bounded experiment that advances the ensemble question. The September 25
recheck and withdrawal of MAE promotion remain in force.

1. Read the complete target paper and visually verify its consequential
   pages; consult the accessible attention reference where it bears on an
   actual missing choice. Distinguish the standalone novelty detector from
   the supervised ensemble.
2. Record source-supported choices, executable completions and separate
   controls without selecting an interpretation because its bound passes.
3. Specify the smallest informative next experiment: question, competing
   predictions, exact data/architecture/objective/metrics, software gates,
   resource ceiling and stopping rule. Explicitly identify unresolved gates.
4. Complete any necessary bounded constructed checks, preserve failures,
   update current status/handoff and the explanation register, verify and
   commit the checkpoint.

This source-and-design step does not itself launch new research fits or
data preparation. Preserve historical contracts, results and direct
reproduction files. Website work remains with the other session.

## Completed outcome

- Read and visually inspected every target page; checked reference [23]'s
  complete pp. 425, 426, 428 and 429. Both source PDF hashes match the prior
  specification. Reference [22]'s full-text access limitation remains.
- Wrote [the next experiment](../../studies/takiddin-2021-robust-poisoning/SEQUENTIAL_ENSEMBLE_PILOT.md)
  with separate scaling/objective/score/cutoff decisions, source locators,
  all consequential completions, exact data identities and a bounded paired
  classification experiment. Section IV-C's ensemble settings take priority
  for this interpretation; no standalone MAE loss or0.51 cutoff is inherited.
- Declared concrete constructed learning/query/feedback/reload checks and
  full-model variable-batch timing before research execution. These ensemble
  gates are pending; the existing standalone prototype is not their substitute.
- Kept the statistical attainability and causal mechanism questions open.
  The prior GRU is contextual, not a matched component ablation. Coverage
  of standalone AEA, ARIMA, averaging and full data remains unfinished.

Verification: `bash scripts/test.sh` ran415 cases:392 passed,23 skipped,
and journal consistency passed. The separate pinned TensorFlow invocation
of `tests.test_robust_aea_model` passed all6 existing constructed tests.
These tests verify existing software, not the unimplemented ensemble.
`.venv/bin/python scripts/verify_data.py --strict` passes the recorded
`sciencedb-csv-semantic-equivalence-v1` branch; restricted official copies
remain unavailable. No data substitution was made. Local links and parameter
arithmetic were checked; the proposed9,240,802 parameter count still needs
runtime confirmation. `git diff --check` passes; direct code, previous results,
tests and website have no changes. Only this session's temporary PDF renderings
and extracted text were removed after review.

Next execution stage: implement and validate the specified ensemble, then
review the constructed timing gate before a research-launch checkpoint.
No new model research fit, real-data scoring or cluster allocation occurred.
