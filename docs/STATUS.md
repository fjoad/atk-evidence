# ATK Evidence — Current Status

**Last updated:** 2026-08-31

**Branch:** `main`

**Active plan:**
[`plans/2026-08-23-clean-reader-reproduction-rebase.md`](plans/2026-08-23-clean-reader-reproduction-rebase.md)

## Current project state

- **Small fitted Sigmoid check complete; discussion pending:** the user
  requested testing the remaining alternative. See
  [the bounded fit plan](plans/2026-08-31-small-sigmoid-fit.md) and
  [exact setup](../studies/atk-2022-deep-autoencoder/SIGMOID_FIT_CHECK.md).
  Frozen code `cc9af5e` passed 230 tests before execution. CPU job `385198`
  completed `0:0`: 9.22-second pilot, 24.81-second small analysis, 3:52 total
  allocation including startup and inspection. Both heads completed 640
  updates / ten epochs from identical initial weights. On 12,119 held-out
  sampled rows, Sigmoid's maximum DR at FA<=15% is 9.74935% high-error or
  25.39063% reversed, versus 81%. Relaxed rounding also fails. Softmax gives
  8.64258% / 25.52083%. Every cutoff fails for these fitted scores, not every
  possible Sigmoid configuration. Sigmoid calibration MSE improved through
  the last epoch; no long-run plateau is established. The
  [finding](../studies/atk-2022-deep-autoencoder/SIGMOID_FIT_FINDING.md)
  preserves that limitation, every outcome, hashes, and timing. Stop for
  discussion; no full retraining, sweep, public edit, or push is authorized.
  Post-run verification passed 232 deterministic tests (140 study, 92 root)
  and 73 local documentation links. The README, website, reports, and five
  reproduction files are unchanged from `dc37bbe`. Records are local only.

- **Earlier publication and no-training Sigmoid follow-up complete:** after discussing the
  source-assumption findings, the user requested their publication first,
  followed by quick Sigmoid checks without full retraining. Follow
  [the bounded sequence](plans/2026-08-31-source-findings-and-sigmoid-sanity.md)
  within the governing clean-reader plan. The public update passed all 217
  tests and is live at `dc37bbe`: Pages run `33419150100` succeeded and all nine
  checked public files match local bytes. The no-training Sigmoid contract is
  [SIGMOID_SANITY.md](../studies/atk-2022-deep-autoencoder/SIGMOID_SANITY.md);
  all 223 deterministic tests passed before code freeze `9d6c31b`. CPU job
  `385137` completed `0:0`: pilot 4.97 s, full 39.70 s, allocation 2:17. The
  [new local finding](../studies/atk-2022-deep-autoencoder/SIGMOID_SANITY_FINDING.md)
  records every result. On the complete prepared evaluation, minimum Sigmoid
  FA at 0.58 is 29.66640%, so the printed cutoff still fails. With a changed
  cutoff, upper DR at FA<=15% is 85.32587%: the target pair is no longer
  excluded. Reversal is also not excluded. The original-row result reproduces
  exactly within tolerance; the synthetic benign population changes the
  all-cutoff answer. Neither no-training control reaches the target. No model
  was trained. Stop for discussion before publishing these new outcomes or
  launching another experiment; no full training or broad search is authorized.
  Post-run verification passed all 225 deterministic tests (140 study, 85
  root), including eight Sigmoid checks. All checked documentation links
  resolve; public files and original reproduction remain unchanged from
  published `dc37bbe`. New records and code are committed locally only.

- **Source-assumption review complete; discussion pending:** the user approved reviewing
  source support and checking a few load-bearing alternatives, with discussion
  before public updates. The local plan/contract is
  [SOURCE_ASSUMPTION_CHECK.md](../studies/atk-2022-deep-autoencoder/SOURCE_ASSUMPTION_CHECK.md).
  No training, public-report edit, or push is authorized for this round.
  Local code freeze `b76cb02` passed all 212 deterministic tests. CPU job
  `385119` completed `0:0`: pilot 8.20 seconds, full 56.76 seconds, allocation
  3:18. All consumed hashes/identities and geometric checks passed; the prior
  original-row reference bounds were reproduced. The
  [discussion finding](../studies/atk-2022-deep-autoencoder/SOURCE_ASSUMPTION_FINDING.md)
  records the source map and every outcome. At the printed cutoff, Softmax DR
  is bounded by 29.58% current, 29.81% joint-scalar, and 33.96% separate-class
  scaling (rounded upward), all below 81%. Sigmoid-range control substantially
  changes the limits but is not the Table-I FC-SAE and is not a reproduction.
  No training or ADASYN regeneration occurred. No further experiment, push,
  README/site/report change is authorized before discussion.
  Post-run verification passed all 214 deterministic tests (140 study, 74
  root); local documentation links resolve and `git diff --check` passes.
  README, website, and reproduction files are unchanged from `149cde7`.

- **Completed diagnostic round, 2026-08-31:** the user-approved bound,
  useful-information checks, and warranted small control are complete. Both
  CPU jobs exited `0:0` and released their allocations. No model was trained;
  the original data, weights, scores, and numerical result are unchanged.
  See [POST_ANCHOR_FINDING.md](../studies/atk-2022-deep-autoencoder/POST_ANCHOR_FINDING.md).
  - **Conditional attainability (`C/A`):** the label-aware relaxation on all
    8,884,989 prepared rows has maximum balanced ACC 50.92105%, AUC 45.10918%,
    and DR 9.24779% at FA <=15%, versus 83/81/81% reported. This excludes
    all weights/seeds under fixed preparation, Softmax output, and MSE,
    not other source interpretations. Endpoints use outward-padded float64,
    not certified interval arithmetic. Rounding allowance and reversed
    direction also cannot recover the target.
  - **Useful score differences (`C/M`, then adaptive `X/M`):** original-row
    trained-minus-zero ACC is +0.89081 points, conditional customer-bootstrap
    95% interval [0.80454, 0.98117]. On 10,000 source days plus attack siblings,
    within-energy-band AUC is 65.49% trained, 62.18% projection, 55.02% uniform,
    49.74% energy. The latter differences have no confidence interval.
    “Nothing useful learned” is not established; no matched architecture or
    trained-versus-untrained causal comparison was performed.
  - **Cost:** primary code `1175e8d`, job `385090`: pilot 5.00 seconds, full
    analysis 112.92 seconds, allocation 4:16. Adaptive control code `26a42db`,
    job `385091`: analysis 18.03 seconds, allocation 53 seconds. Both frozen
    contracts and every result are preserved with hashes.
  Stop experiments here. The proposed next step is a source-supported map of
  assumptions that can change the bound, not another seed or model family.

- **Public explanation, 2026-08-31:** the user requested the README and website
  be rewritten before any further experiments. The new research report is
  `site/papers/atk-2022-deep-autoencoder/reproduction/index.html`; it connects
  source passages, the exact implementation, a model diagram, all seven
  metrics, completed diagnostics, and proposed next questions. Both previous
  paper pages remain, with historical scope and corrections made explicit.
  Editorial work follows
  [the short publication plan](plans/2026-08-31-readable-research-report.md).
  The local rewrite is complete: all 187 deterministic tests pass, including
  eight new report checks; site assets and local documentation links resolve.
  Four duplicate agent-rule files were removed; research history remains in
  Git and the scientific records. The user explicitly approved publication on
  2026-08-31: the changes and supporting evidence are now on GitHub, Pages
  deployment `33400529269` succeeded, and all public pages/assets match the
  reviewed files byte for byte. See the
  [live report](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/reproduction/).
  The subsequent limited diagnostic round is complete, as recorded above.
  Its report update adds the bound, paired gains, counterevidence, figures,
  and source records; the initial numerical result remains unchanged.
  The follow-up passes all 206 deterministic tests (140 study, 66 root),
  including 14 report checks. New public figure bytes match the cluster
  exports, 64 local Markdown links resolve, and the updated local pages and
  figures return HTTP 200. Follow-up report commit `97c9236` is on GitHub;
  Pages deployment `33407618030` succeeded. All four public HTML pages,
  stylesheet, social preview, model diagram, and two new figures were fetched
  successfully and match the reviewed files byte for byte.

- The clean-reader reproduction rebase is now the governing execution plan.
  Its saved flow begins with preservation/reconciliation, a paper-only read, a
  disposable sandbox, and a new source freeze before assessing the existing
  five-file implementation. The user approved the plan on 2026-08-23. Phase 0
  preservation/reconciliation and Phase 1 paper-only orientation are complete.
  The user authorized Phase 2's bounded disposable discovery sandbox on
  2026-08-24. Its contract, standalone implementation, and cluster wrapper were
  frozen in commit `83dab57`; all 173 deterministic tests pass. Panther job
  `381540` completed the single frozen wave with exit `0:0`, and Phase 2 is
  complete. Its simple-rule, temporal-witness, output-domain, and score-direction
  observations remain exploratory `X` only. Phase 3 is also complete: every
  PDF page was visually re-inspected and the literal failures, exact official
  data identity, primary completion, causal map, and one-attempt stopping rule
  are frozen in
  [`../studies/atk-2022-deep-autoencoder/CLEAN_READER_SPECIFICATION.md`](../studies/atk-2022-deep-autoencoder/CLEAN_READER_SPECIFICATION.md).
  The user approved Checkpoint 1 on 2026-08-24 and required all plausible
  explanations to be preserved and systematically tested. Phase 4 is complete:
  the five-file trace and quarantine are saved in
  [`../studies/atk-2022-deep-autoencoder/CLEAN_READER_FIDELITY.md`](../studies/atk-2022-deep-autoencoder/CLEAN_READER_FIDELITY.md),
  the frozen mismatches were minimally corrected, and all 179 deterministic
  tests pass. The formerly blocked exact-serialization gate remains preserved:
  official metadata shows that the absent 196,316-byte `.tab` is
  Dataverse's archival ingest of an original XLSX. The allocation information
  itself is available in a public ScienceDB CSV and a public GitHub workbook;
  all 6,445 mappings agree under the predeclared blank/zero normalization, and
  all residential reading IDs are covered. The six consumption archives
  already match the official bytes exactly. On 2026-08-30 the user explicitly
  approved the verified `sciencedb-csv-semantic-equivalence-v1` allocation
  branch as a visible `I` completion for the single anchor. The changed source
  gate is frozen in eligible code commit
  `a88d17477ad96b01ffa44a50d8ce051dd8d2b5ca`; 179 deterministic tests and the
  strict local seven-file verifier pass. The sole Panther submission is job
  `384390`, completed with Slurm exit `0:0` on 2026-08-30 at 22:48:39 Qatar
  time after 9:14:27. Phase 6 is complete: the frozen audit and all 65 corrected
  supplemental artifact checks passed. The Table-III ISET FC-SAE row was not
  reproduced in this one frozen `P+I` completion. Checkpoint 2 was subsequently
  approved only for the now-completed bounded diagnostic round above.
  The initial finding and full comparison are in
  [`../studies/atk-2022-deep-autoencoder/CLEAN_READER_FINDING.md`](../studies/atk-2022-deep-autoencoder/CLEAN_READER_FINDING.md).
  The hourly thread heartbeat
  `monitor-clean-reader-anchor` is paused at the user's request; check the job
  only after a new user prompt. No second submission, repair, retry, or
  expansion beyond that completed round is authorized. Broader mechanism and
  attainability work remains unapproved. Prior code, runs, failures, and results remain
  preserved without retroactive reclassification.
  The complete frozen Phase-5 question, code/environment hashes, predictions,
  36-hour one-GPU budget, exact command, stopping rule, and data-gate state are
  in
  [`../studies/atk-2022-deep-autoencoder/CLEAN_READER_ANCHOR_PRERUN.md`](../studies/atk-2022-deep-autoencoder/CLEAN_READER_ANCHOR_PRERUN.md).
  The initial contiguous array samples were fully finite and nonconstant
  where expected; this was a sampled sanity check, not a complete array audit.
  The completed preparation now records 1,500,523 training profiles and
  8,884,989 post-ADASYN test profiles. Measured ADASYN time was 3,890.66 s
  (1:04:51); recorded extraction plus preparation took about 1:09:43;
  fitting took 28,965.90 s (8:02:46) for 28 epochs on a Tesla P100, restoring
  epoch 23; Table-III scoring took 11.12 s. The earlier 07:00--08:00 Aug-31
  ETA is invalidated: it incorrectly reused a benchmark of the larger
  historical test population and transferred training timing across hardware
  and contracts. The saved attempt is
  `seed_20260824_2f483335536c` under the semantic-allocation results directory.
  Its reported metrics are now independently verified with exactly zero
  discrepancy. DR/FA/ACC/AUC/F1 are 25.48/45.13/40.18/39.40/30.09% versus
  81/15/83/81/81% reported. Every saved numeric array was fully scanned with
  no NaN/Inf; customer identities are disjoint; fresh-loaded weights agree
  with 256 saved scores within 1.20e-7. Exact threshold enumeration gives at
  most 50.00072% balanced ACC in the printed direction, 60.21% reversed.
  Scores correlate 0.999253 with zero-reconstruction input energy, while the
  trained ACC gain over that rule is only 1.18 points. These diagnostics do
  not establish mechanism failure or a method-family attainability envelope.
  This full-data numerical anchor is distinct from Phase 2's 60.06-second
  toy/synthetic sandbox (2:25 total job time); neither is relabeled as the other.
- The active competing explanations, their predictions, discriminating tests,
  and status are now durable in
  [`../studies/atk-2022-deep-autoencoder/EXPLANATION_REGISTER.md`](../studies/atk-2022-deep-autoencoder/EXPLANATION_REGISTER.md).
- The Phase-1 discussion has been preserved as six bounded source-only
  findings in
  [`EVIDENCE-AND-LEARNINGS.md`](EVIDENCE-AND-LEARNINGS.md): unusually total
  numerical ordering; non-identification of the credited mechanisms by the
  reported comparisons; a shortcut/triviality hypothesis; literal execution
  blockers; a proved output-domain reconstruction floor; and an explicit
  statement that the formal numerical, mechanism, and attainability findings
  remain open.
- The paper-first minimal-instrument reframe is accepted and recorded in
  [`decisions/2026-08-09-paper-first-minimal-instrument.md`](decisions/2026-08-09-paper-first-minimal-instrument.md).
- Paper 1 is the only active execution target. Paper 2's existing artifact-level
  audit remains preserved but frozen; cross-paper synthesis has not started.
- Panther job `373789` completed successfully in 53:12 from commit `c8c136f`
  on one 16-GB V100. It passed source/preparation gates, trained the frozen
  batch-512 FC-SAE for 74 epochs (best epoch 69), and saved Tables III/full-IV/V
  artifacts. Panther CPU job `373799` measured the selected exact ADASYN
  neighbor implementation, and score-audit job `373800` completed. Premature
  repeated-seed jobs `373803` and `373804` were cancelled before execution
  after the breadth-first correction. One-factor linear-output job `373805`
  completed successfully in 1:05:43 on one 16-GB V100; audit job `373824`
  completed in 24 seconds. Naive Bayes (`373833`), ARIMA (`373836`), one-class
  SVM (`373837`), supervised feed-forward (`373838`), multiclass SVM (`373840`),
  and FC-VAE (`373842`) are complete. The first supervised-LSTM attempt
  (`373839`) trained but failed during oversized-batch scoring; the first three
  proposed recurrent attempts (`373841`, `373843`, `373844`) failed before
  training in diagnostic layer inventory. Those defects were fixed in
  `c735dd9`, and corrected supervised-LSTM job `374310` completed. A second
  pre-training sanity-probe batch leak caused jobs `374311`--`374313` to fail
  before training; it is fixed in `4469a53`. Jobs `374388`--`374390` were then
  rejected by the immutable-attempt guard before execution because their
  configuration still named score batch 512. Explicit score batch 256 makes
  the preserved retry identity distinct. LSTM-SAE job `374391` completed and
  score-audit job `374433` closed its threshold question. LSTM-VAE job `374395`
  completed training but exhausted 16-GB GPU memory during scoring; its weights
  were preserved, and no-gradient fresh-process recovery `374441` scored them
  without retraining. LSTM-AEA job `374396` completed all 100 epochs and full
  scoring. Panther audit jobs `378014` and `378015` closed both score-direction
  questions. Those model-family breadth jobs are complete; Table-IV half and
  three-quarter jobs `378182`--`378191` were active when last observed on
  2026-08-20, and their present state has not been harvested. The Slurm wrapper
  only requests `gpu-all` plus `gpu:1` and executes its arguments.
- Experimental preparation, training, and scoring must run on cluster compute
  nodes. Local work is limited to source reconstruction, code, documentation,
  lightweight inspection, transfer, and monitoring.
- The renewed source freeze is accepted. The five direct files have now been
  traced against the corrected `METHOD.md`; the unfinished score-audit work was
  completed without importing the historical `src/` route.
- All eleven named Table-III model rows now have one registered seed-11 breadth
  result and score audit. The compact model-family map is
  [`../studies/atk-2022-deep-autoencoder/TABLE_III_BREADTH.md`](../studies/atk-2022-deep-autoencoder/TABLE_III_BREADTH.md).
  No repeated-seed depth is authorized before the predeclared one-factor
  population, split, scaling, threshold, and Attack-3 interpretations.
- Table-II seed-11 breadth is now complete. Literal SGCC execution fails because
  the source has 1,034 daily values per customer while the models require 48
  half-hour inputs and no conversion is stated. The executable map covers all
  eleven rows on `last_48`, and all five proposed models plus the feed-forward
  control on `first_48` and `binned_mean_48`. Proposed-model AUC ranges are
  46.31--46.59, 48.86--51.23, and 53.59--54.15 respectively, versus 83--93
  reported. The control AUC is 95.31--96.91 and its closest DR/FA gap is only
  0.80--1.94 points. Exact threshold enumeration leaves every proposed score
  vector at least 33.18 points from its complete reported row. See
  [`../studies/atk-2022-deep-autoencoder/TABLE_II_BREADTH.md`](../studies/atk-2022-deep-autoencoder/TABLE_II_BREADTH.md).
- Table-V common-model/common-benign breadth is complete for all five proposed
  models. Reproduced FA is exactly invariant across attacks for each model, as
  required mathematically, while the paper varies it. Reproduced attack DRs are
  also far below most printed cells. The retrain/resplit interpretations remain
  open. Table-IV half and three-quarter jobs `378182`--`378191` were running at
  the last observation; full-size cells reuse the completed Table-III models.
- Exact threshold enumeration is complete for all five proposed ISET score
  vectors. Even the best threshold leaves a 48.91--60.11-point maximum gap
  across the complete seven-metric Table-III row. This rules out threshold
  choice for those vectors, not hyperparameters that create different scores.
  FC-VAE Table-IV half and three-quarter runs were recovered after a post-score
  target-lookup failure: ACC is 39.44% in both cells versus 79.5% and 86%
  reported. The remaining eight Table-IV jobs were healthy when last observed;
  their final state awaits Panther harvest because the QCRI VPN currently
  requires an interactive OTP login.
- The first material one-factor branch is implemented but not executed:
  `--residential-population seeded_3000` deterministically selects exactly
  3,000 of the 4,225 labeled residential meters while changing no other paper-
  primary preparation choice. The default all-meter cache identity remains
  backward compatible. Full deterministic tests pass (140 study + 33 root).

## Paper 1: verified foundation

- The exact 12-page PDF was independently re-audited on 2026-08-11: SHA-256
  verified, text extracted, and every rendered page visually inspected before
  the prior reconstruction was opened. The corrected source-located executable
  reconstruction is
  [`../studies/atk-2022-deep-autoencoder/METHOD.md`](../studies/atk-2022-deep-autoencoder/METHOD.md).
- The re-audit confirmed the main flow and prior pivotal contradictions, fixed
  the benchmark count (six, not seven), and added omitted VAE-derivation,
  decoder, precision-definition, F1, and common-prevalence inconsistencies.
- A stronger balanced-test arithmetic check is now machine-readable: five
  Table-II and eight Table-III rows cannot satisfy
  `PR = DR / (DR + FA)` even with ±0.5-point rounding, although the paper calls
  the ADASYN output balanced. This is independent of every reproduction choice.
- `P0-ISET-FCSAE` is now labeled precisely as a paper-primary `P+I` executable
  completion. The printed Attack-3 subtraction remains a non-executable source
  outcome rather than being silently called literal.
- The accepted non-executable-source contract requires the literal failure and
  every predeclared reasonable repair to be executed and reported side by side
  with the published target; see
  [`decisions/2026-08-11-non-executable-source-ladder.md`](decisions/2026-08-11-non-executable-source-ladder.md).
- Exact CER/ISET consumption archives are verified. The allocation CSV is the
  explicitly labeled semantic-equivalence branch, not the official `.tab`
  serialization.
- The five direct reproduction files exist under
  `studies/atk-2022-deep-autoencoder/reproduction/` and do not import the
  historical forensic `src/` implementation.
- Full preparation produced 2,251,290 benign profiles, 13,507,740 attacked
  profiles, 1,500,520 B1 training profiles, and the 14,258,510-row `B2+M` test
  population.
- Printed-position default ADASYN is not complete: its exact default full-scale
  neighbor query entails roughly 10.7 trillion first-pass distances. A 16-core
  same-machine benchmark estimates 14.16 wall-hours for both exact neighbor
  searches alone, before synthesis and persistence. This is expensive but
  feasible as an overnight job and gives no basis for claiming that the authors
  could not have run ADASYN. Do not relabel the no-resampling interpretation as
  the printed result.

## Paper 1: current experimental evidence

The SGCC breadth map provides a clean architecture-family contrast. Across all
three one-customer/48-wide readings, proposed-model AUC stays between 46.31%
and 54.15%, while the supervised feed-forward control stays between 95.31% and
96.91%. Within each representation, all proposed score rankings have pairwise
Spearman correlation at least 0.957. FC-SAE and LSTM-AEA raw MSE are almost
identical (Pearson >0.99999998), as are the two VAE score transforms. For these
executed branches, LSTM/VAE/attention changes do not generate the paper's model
hierarchy; they preserve essentially the same weak ranking. These are one-seed
exploratory observations, not initialization-level confidence intervals.

The new compact batch-512 anchor completed:

- method: `I-ADASYN-NONE-ISET-FC-SAE`, seed 11, batch 512;
- reproduced DR/FA/ACC/AUC/F1:
  26.18 / 58.22 / 33.98 / 31.04 / 40.46%;
- reported DR/FA/ACC/AUC/F1: 81 / 15 / 83 / 81 / 81%;
- fit/total time: 45:07 / 47:41 inside the pipeline; and
- Table-V FA is exactly 57.9152% for all six attacks, as required by the frozen
  common-model/common-benign interpretation but unlike the paper's varying FA.

This is a completed exploratory no-resampling anchor, not printed-ADASYN `P0`
and not yet a confirmatory verdict. Its score-distribution/reload audit is now
complete: an oracle threshold in the paper direction reaches only 50.00%
balanced ACC; reversing direction reaches 66.26%; and the trained score ranking
is 0.99946-correlated with the zero-reconstruction control. The committed
result records are
[`../studies/atk-2022-deep-autoencoder/results/compact_route_fc_sae_seed11_batch512_20260811.json`](../studies/atk-2022-deep-autoencoder/results/compact_route_fc_sae_seed11_batch512_20260811.json)
and
[`../studies/atk-2022-deep-autoencoder/results/iset_fc_sae_seed11_score_audit_20260811.json`](../studies/atk-2022-deep-autoencoder/results/iset_fc_sae_seed11_score_audit_20260811.json).

The one-factor linear-output control also completed:

- method: `C-OUTPUT-LINEAR-ISET-FC-SAE`, seed 11, batch 512;
- only change from the anchor: final output activation Softmax to linear;
- reproduced DR/FA/ACC/AUC/F1:
  12.32 / 30.78 / 40.77 / 28.14 / 21.61%;
- paper-direction oracle ACC is only 50.04%; reversing direction reaches 67.56%;
- benign mean error is 0.537 versus malicious 0.281, so the average ordering is
  opposite the paper's decision rule;
- trained versus zero-reconstruction score correlation falls from 0.99946 to
  0.82089, showing that the activation changes the learned score materially but
  still does not produce the reported separation; and
- Table-V FA is exactly 30.0696% for every attack under the common all-benign
  evaluation population, again unlike the paper's varying FA cells.

This closes output activation *alone* as a sufficient explanation for the
baseline gap. It remains a one-seed corrected control, not a paper-level verdict.
The committed summary is
[`../studies/atk-2022-deep-autoencoder/results/iset_fc_sae_linear_seed11_score_audit_20260811.json`](../studies/atk-2022-deep-autoencoder/results/iset_fc_sae_linear_seed11_score_audit_20260811.json).

The benchmark reuse check found no preserved ISET/Table-III benchmark attempt
in either the committed summaries or Panther's attempt manifests. Existing
benchmark results are SGCC/Table II and cannot fill the current ISET breadth
rows. Minimal ISET Naive Bayes job `373833` completed on Panther. It uses
Gaussian Naive Bayes, all original all-customer `B+M`, and an exact seeded 2:1
row split. The omitted supervised ADASYN step is explicit in method
`I-SUPERVISED-ADASYN-NONE-ISET-NAIVE-BAYES`; this is not the printed branch.
It reproduced DR/FA/ACC/F1/AUC = 88.78/44.53/72.12/90.50/79.17%, versus
73/18/77.5/73/70% reported. This completion does not reproduce the reported
operating point.

ARIMA job `373836` then completed in 1m02s using the predeclared smallest
pooled `(1,1,0)` residual-MSE completion on all B1 and original `B2+M` rows.
It reproduced DR/FA/ACC/F1/AUC = 21.48/57.20/32.14/34.46/24.72%, versus
86/12/87/86/87% reported. It is a large non-match with reversed/worse-than-
chance ranking, but the paper omits `p`, fit unit, and score, so it is not a
claim over all ARIMA implementations.

One-class SVM job `373837` completed in 1m04s using the explicit
`kernel=sigmoid, gamma=scale, nu=0.5` repair with a 12,000-row training cap and
30,000-row test cap. It reproduced DR/FA/ACC/F1/AUC =
91.87/50.94/70.47/94.35/79.67%, versus 90/9/90.5/89.5/87% reported. Its DR is
near the paper only at a false-alarm rate above 50%; this bounded diagnostic
does not reproduce the reported operating point and cannot fill the full cell.

Score-audit jobs `373854` and `373855` completed. Oracle-threshold analysis
separates the failures: Naive Bayes is primarily an omitted operating-point
issue (oracle ACC 74.74%, closest reported DR/FA gap 5.00 points); pooled ARIMA
is a fundamental paper-direction failure (oracle ACC 50.00%, reversed 69.74%,
closest gap 56.56 points); capped one-class SVM has useful ranking but cannot
reach the claimed high-DR/low-FA corner (oracle ACC 73.86%, closest gap 18.31
points). These are exploratory score diagnostics, not repeated-seed inference.

Supervised feed-forward job `373838` completed in 1:29:56 using the printed
five 500-unit ReLU hidden layers and Adamax, with the predeclared two-class
Softmax/categorical completion and no supervised ADASYN. At the ordinary 0.5
cutoff it reproduced DR/FA/ACC/F1/AUC =
96.41/23.72/86.35/96.24/97.05%, versus 90/11/89.5/89.5/88% reported. Score
audit job `374255` found that a threshold of 0.824 reaches DR=91.83% and
FA=9.17%, within 1.83 points of the reported pair; the best balanced threshold
reaches 91.66% ACC. This is therefore a threshold-procedure ambiguity with a
strong learned ranking, not a fundamental separation failure. The paper does
not specify a supervised threshold-selection rule, and this branch still omits
pre-split ADASYN.

Multiclass SVM job `373840` completed in 2m27s using the explicit seven-class
sigmoid/scale repair and deterministic 30,000-row train/test caps. Fixed
DR/FA/ACC/F1/AUC = 85.94/55.67/65.14/88.04/73.06%, versus
91/8/91.5/90.5/89% reported. Audit job `374302` found a best balanced ACC of
71.14%; the closest threshold remains 23.44 points from the reported DR/FA
pair. This bounded completion does not reproduce the operating point.

FC-VAE job `373842` completed in 6m14s and restored epoch-2 weights after seven
epochs. Under the predeclared missing-score completion
`exp(-0.5 * profile MSE)`, fixed DR/FA/ACC/F1/AUC =
11.51/32.62/39.45/20.32/30.13%, versus 88/11/88.5/88.5/85% reported. Audit job
`374303` found only 50.00% oracle ACC in the paper's low-probability direction;
reversing direction reaches 66.70%. Malicious mean probability 0.750 exceeds
benign 0.567, and the trained score correlates 0.99957 with the corresponding
zero-reconstruction control. This is a strong failure of the registered VAE
score completion, while the source's missing probability definition keeps
other materially distinct completions open.

The recurrent failures are preserved as evidence, not silently discarded.
Job `373839` saved trained weights before its 8,192-row scoring batch exhausted
the V100. Jobs `373841`, `373843`, and `373844` stopped before training because
the diagnostic inventory assumed one tensor per layer output. The repair only
records multi-output shapes and reduces recurrent inference batches to 512; it
does not alter a model, optimizer, training batch, data row, score, or metric.

Corrected supervised-LSTM job `374310` completed in 6:54:51 after six epochs,
restoring epoch-1 weights. Every test score is exactly 1.0, yielding
DR/FA/ACC/F1/AUC = 100/100/50/92.32/50%, versus
90.5/10/90/90/89% reported. Audit job `374387` confirms that both score
directions have oracle ACC 50%; the closest threshold is 90 points from the
reported DR/FA pair. This registered one-seed completion collapsed completely
and is not a threshold-selection failure.

LSTM-SAE job `374391` completed in 6:23:28 after 25 epochs, restoring epoch-20
weights. Fixed DR/FA/ACC/F1/AUC =
14.78/40.96/36.91/25.25/33.09%, versus 85/13/86/85/82% reported. Audit job
`374433` found paper-direction oracle ACC 50.004%; its closest point to the
reported DR/FA pair remains 47.11 points away. Reversing direction reaches
64.38% ACC. Benign mean reconstruction error 1.087 exceeds malicious 0.519,
and the trained score is 0.97495-correlated with zero reconstruction. This is a
fundamental wrong-direction failure for the registered one-seed no-test-ADASYN
completion, not a threshold mismatch.

LSTM-VAE job `374395` trained for 23 epochs and preserved its best epoch-18
weights before inference exhausted the 16-GB allocation. Recovery job `374441`
loaded those exact weights in a new no-gradient process and scored the original
14,258,510-row test set in 37:03 without retraining. Fixed
DR/FA/ACC/F1/AUC = 10.02/25.79/42.11/17.98/29.83%, versus
91/7/92/91/86% reported. Audit `378014` found paper-direction oracle ACC
50.002%; the closest paper-direction point remains 58.48 points from the
reported DR/FA pair. Malicious mean reconstruction probability 0.783 exceeds
benign 0.630 although the paper declares lower probability anomalous; reversing
direction reaches only 66.93% ACC. The trained score is 0.93379-correlated with
zero reconstruction. This is a wrong-direction failure for the registered
fixed-unit-probability completion, not a threshold mismatch.

LSTM-AEA job `374396` completed all 100 epochs in 43:41:50, with the minimum
training loss at epoch 100. Fixed DR/FA/ACC/F1/AUC =
25.43/58.22/33.60/39.53/29.93%, versus 94/5/94.5/93.5/90% reported. Audit
`378015` found paper-direction oracle ACC 50.002%; its closest point remains
60.11 points from the reported DR/FA pair. Benign mean MSE 1.286 exceeds
malicious 0.645 although the paper declares higher error anomalous; reversing
direction reaches only 66.52% ACC. The trained score is 0.97843-correlated with
zero reconstruction. This is a wrong-direction failure for the registered
attention completion, not a threshold mismatch.

None of the eleven registered fixed operating points reproduces its complete
printed metric pattern. Supervised feed-forward is nevertheless a strong
positive control; it and Naive Bayes expose an omitted threshold-selection
procedure. The other nine score vectors remain materially far from the
reported DR/FA corner under threshold adjustment in their registered direction.
These remain one-seed exploratory branches and
do not execute printed ADASYN or eliminate every source-supported completion.

One full compact-route cluster result exists:

- table/model: Table III, FC-SAE;
- method: `I-ADASYN-NONE-ISET-FC-SAE`;
- seed/batch: 11 / 32;
- population: exact original pre-ADASYN `B2+M` rows;
- epochs: 29;
- reproduced DR/FA/ACC/AUC/F1:
  26.44 / 58.51 / 33.97 / 31.03 / 40.78%; and
- reported DR/FA/ACC/AUC: 81 / 15 / 83 / 81%.

This is one exploratory batch-32 sensitivity, not a reproduction verdict and not
the batch-512 primary branch. Its full score arrays, weights, history, predictions,
and hashes are local. Its score/eligibility audit is unfinished.

## Exact next action

1. Discuss the new, locally saved Sigmoid findings: the printed 0.58 cutoff
   still fails, but the complete-evaluation bound no longer excludes the target
   with a changed cutoff. The original-row limit must not be generalized across
   the synthetic benign evaluation population.
2. Decide whether a tiny properly trained Sigmoid comparison could distinguish
   the remaining explanations. No learned model has yet realized the bound's
   opening; failure of two label-blind rules does not rule training out.
3. Stop experimental execution and public updates pending discussion. No
   further training, seed, branch, report edit, or push is approved.

The former next action—one-factor population/split/scaling/threshold/Attack-3
execution followed by repeated seeds—is superseded. It may return only if the
new plan promotes it after the trusted clean-reader anchor.

## Not on the critical path

- the historical 921-configuration branch matrix;
- new scheduling, manifest, DDP, or workflow infrastructure;
- Paper 2 execution;
- website or LaTeX polishing;
- the corrected preferred detector; and
- cross-paper conclusions.
