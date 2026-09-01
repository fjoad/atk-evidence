# Publish the current evidence, then complete the paper audit

Date: 2026-09-01

User direction: update and publish the full public site first, then complete
the remaining proposed-model, mechanism, and table work discussed on 1
September. This plan extends the governing clean-reader plan; it does not make
historical quarantined runs eligible.

## A. Current-evidence publication - complete

1. Update the README, home page, earlier-study warning, and current reproduction
   report with both completed Sigmoid checks.
2. Replace the stale downloadable PDF with a dated report of the audited
   clean-reader FC-SAE result, conditional Softmax bound, source-assumption
   checks, small fitted Sigmoid result, present table coverage, and limitations.
3. Generate every displayed number from or test it against preserved result
   records. Keep the distinction between a bound, a trained result, and an open
   possibility explicit.
4. Build/test, render and visually inspect the PDF, preview the static site,
   commit, push, wait for GitHub Pages, and verify deployed files byte-for-byte.

No scientific experiment may run during this phase.

Completed in `b1973d9`. All 235 deterministic tests passed (140 study, 95
root/public), strict data verification selected the verified ScienceDB branch,
all five PDF pages were rendered and inspected, Pages run `33448349593`
succeeded, and all 11 deployed files matched the reviewed local bytes.

## B. Exact remaining-paper contract - complete

Return to the frozen complete-paper source map. Reconcile Table-I definitions,
Tables II-V targets, every non-executable instruction, and the quarantined
breadth results. Write a source-located execution contract that declares:

- the four remaining proposed Table-III model configurations;
- model-specific losses, output domains, anomaly scores, directions, thresholds,
  training/stopping settings, validation populations, seeds, and failure rules;
- cheap feasibility and runtime checks before each full run;
- what Table-IV and Table-V cells can be derived from the same saved models and
  what genuinely requires new training;
- the finite Table-II 48-value interpretations, with literal failure retained;
- a matched FC/LSTM mechanism test and a temporal-structure destruction test;
- statistical units, paired effects, smallest effect of interest, uncertainty,
  compute budgets, promotion rules, and stop conditions; and
- exactly which result feeds numerical (`N`), mechanism (`M`), or attainability
  (`A`) findings.

Before cluster submission, compare the contract with available resources and
show the user the exact proposed runs, costs, and remaining ambiguities. This is
the required scientific checkpoint; broad authorization does not permit us to
invent omitted model semantics after observing outcomes.

The proposed contract is now
[`REMAINING_PAPER_CONTRACT.md`](../../studies/atk-2022-deep-autoencoder/REMAINING_PAPER_CONTRACT.md).
It was reconstructed directly from the source PDF and preserves the deferred
decoder, VAE-score, and attention failures. Panther is reachable, the audited
7.1-GiB prepared cache and sole FC-SAE result remain present, no jobs are
queued, and the accessible general partitions remain available. The first
requested execution wave is four separate two-hour feasibility jobs (eight
GPU-hours maximum); passing models then receive one immutable full anchor each,
at no more than 72 GPU-hours per model. The user accepted all five decisions in
Section 12 on 2026-09-01.

## C. Clean-reader Table III proposed-family depth - feasibility instrument in progress

After the checkpoint, run cheap checks and then one watched eligible anchor for
each of LSTM-SAE, FC-VAE, LSTM-VAE, and LSTM-AEA on the same audited data path as
FC-SAE. Preserve every failure. Audit all artifacts and compare all seven
metrics plus the reported family ordering. Add seeds only where the first run
leaves material stochastic uncertainty and the frozen plan promotes them.

The four-model implementation and pilot-only Slurm wrapper are now written.
The old `reproduction/models.py` remains byte-identical because completed
Sigmoid evidence records its source hash; new approved completions live in
`reproduction/remaining_models.py`. Before submission, the complete regression
suite must pass, the code must be committed and pushed, and Panther must be at
that exact clean commit. The first execution remains exactly four capped
operational pilots, not full anchors.

Local verification is complete: 140 study tests and 103 root/public tests pass,
the strict data verifier selects the complete ScienceDB semantic-equivalence
branch, and the historical source hash guard passes. Commit/push and exact
Panther synchronization were required before submission.

Commit `0ca6cc4` was pushed and synchronized. The first four jobs
(`385544`--`385547`) all stopped at the same pre-model checksum gate because
cache metadata omits `table_iv_order.npy`'s checksum. The file's observed hash
matches the eligible anchor record exactly. No model was built or scored. See
`REMAINING_PILOT_FINDING.md`; stop for discussion before the proposed narrow
manifest-source repair and any new immutable attempts.

The user approved that exact repair on 2026-09-01. The repaired gate passed 245
deterministic tests (140 study, 105 root), and strict data verification selected
the complete ScienceDB semantic-equivalence branch. No data, model, training,
scoring, resource, or promotion field changed. It was frozen and synchronized
before new immutable pilot attempts were created.

The repair was frozen at `052ac37`, pushed, and synchronized. New jobs
`385552`--`385555` crossed the input gate and completed their bounded attempts.
FC-VAE passed all gates. LSTM-SAE and LSTM-VAE stopped at the frozen `1e-6`
all-score batch-agreement gate; LSTM-AEA stopped at the 72-hour projection gate
with a 1,879.93-hour conservative estimate. All saved arrays and transferred
artifact hashes passed audit. Stop for discussion before promoting FC-VAE or
changing any failed gate. See `REMAINING_FEASIBILITY_FINDING.md`.

After that discussion, the user approved exactly two score-only recoveries for
the preserved recurrent weights. Jobs `385583` and `385584` completed `0:0`
without training. Printed-cutoff labels and metrics, best balanced accuracy,
and FA-capped outcomes were stable across batches; VAE AUC moved by at most
`0.0000217922` percentage points. Literal transfer of a batch-256 ROC cutoff
can flip one boundary row. The original gate has not been replaced. See
`REMAINING_SCORE_RECOVERY_FINDING.md`; stop for discussion before promotion or
another Phase-C run.

The next discussion authorized costing and one LSTM-SAE anchor only if it fits
the existing ceiling. The A16 evidence projects 417.14 hours for the possible
100 epochs, so it is not eligible on that measured device. One unchanged-method
single-H200 cost pilot is frozen in `LSTM_SAE_ANCHOR_PROMOTION.md`. It is
operational `X`; the full `CR-ISET-LSTMSAE-01` anchor launches only if the new
100-epoch projection and every other gate pass. This is not authorization for
another model or a partial anchor.

The frozen H200 pilot was submitted as job `385602`, but the account's only
QOS is not permitted on the H200 partition. It was canceled while pending with
zero GPU time and produced no model result. Because the required timing gate
was not observed and the A16 projection fails, the conditional full anchor is
not launched. See `H200_COST_FINDING.md`; a new resource or budget decision is
required before Phase C continues.

The user then rejected faster/additional hardware and redirected the audit to
the paper's own Table-IV time claim. The active continuation is
`docs/plans/2026-09-01-paper-time-budget.md`: one V100-16GB, one seed, exact
full LSTM-SAE data/model, and 183 minutes of fitting followed by full scoring.
Contemporaneous records make V100 the strongest specific hardware clue, but do
not establish what the authors used. No other Phase-C model is included.

## D. Mechanism program - pending Table III anchors

Run the predeclared matched FC/LSTM comparison and temporal-structure
intervention. Test the six links in `B > A because Z exploits S`; do not infer
mechanism from the source-configured table alone.

## E. Tables IV and V - pending reusable audited models

Use saved eligible models where the paper's table semantics permit. Run only
the missing data-size fits required by Table IV. For Table V, first establish
the common-model/common-benign identity, then execute only the predeclared
material alternate readings. Do not retrain merely to recreate a changing
false-alarm column unless that interpretation was source-supported and frozen.

## F. Table II - pending ISET program

Retain the literal SGCC 1,034-to-48 failure. Promote only the finite
source-supported executable mappings frozen in Phase B. Do not relabel the
historical three-repair breadth map as clean-reader confirmation.

## G. Three findings and final publication - pending all promoted work

Regenerate Tables II-V from eligible preserved attempts. Report historical
exploration separately. Issue bounded numerical, mechanism, and attainability
findings, including open explanations and failures, then update the HTML and PDF
once more. No allegation about intent follows without separate evidence.
