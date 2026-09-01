# Explanation register

**Created:** 2026-08-24

**Status:** Active; update whenever an observation creates, weakens, separates,
or closes a materially plausible explanation

**Boundary:** These are competing explanations, not findings about author
intent. `N` asks whether numbers reproduce, `M` asks what capability causes an
effect, and `A` asks whether a reported target lies inside a declared empirical
envelope. One result rarely closes all three.

## Operating rule

For every material result:

1. preserve the result before interpretation;
2. identify which explanations predicted it;
3. identify which incompatible predictions were actually tested;
4. update status without deleting alternatives;
5. predeclare the smallest next test that separates surviving explanations;
6. stop when a bounded finding is earned rather than testing forever.

Statuses are `OPEN`, `SUPPORTED`, `WEAKENED`, `CONTRADICTED`, or `CLOSED WITHIN
DECLARED SPACE`. “Supported” never means uniquely identified unless competing
explanations were separated.

## Primary explanations raised by the temporal sandbox

### E1 — recurrent optimization or underfitting

**Explanation:** The LSTM architecture or its training procedure is harder to
optimize, so its theoretical temporal capability never becomes usable.

**Present evidence:** `SUPPORTED AS A SANDBOX POSSIBILITY`. The sandbox LSTM
had finite updates but ended with training MSE 0.944 versus 0.345 for the dense
AE and performed worse on block disruption.

**Predictions:** LSTM training/validation loss stays materially worse; more
appropriate optimization improves fitting before it improves temporal
behavior; comparisons disappear when fitting quality is matched.

**Discriminating tests:** inspect source-configuration learning curves and
gradients; require matched benign fitting success; then vary only predeclared
optimization budget or optimizer as `C`, never as reproduction. Test multiple
frozen seeds only after one complete anchor.

**Current operational update, 2026-09-01:** the two-epoch exact-cache pilot
improved objective from 1.5899 to 1.4972 and therefore does not show a plateau.
Its A16 timing projects to 42.79 hours for ten epochs and 417.14 hours for the
100-epoch ceiling. The unchanged-method H200 cost pilot did not execute:
Slurm job `385602` was held because the account QOS is not permitted on the
partition and was canceled with zero GPU time. Affordability on H200 therefore
remains unmeasured; this does not update the optimization explanation.

**Subsequent time-budget test:** the user rejected newer/additional hardware.
The paper reports 183 minutes for full-ISET LSTM-SAE training. Contemporaneous
records support a K80-to-V100 access range, with the strongest specific clue
favoring V100. One full-data V100 fit capped at exactly that training budget is
pending. Its learning curve can update E1 only within that declared envelope;
it cannot establish author hardware or universal optimization behavior.

**Feeds:** `N` implementation/failure interpretation, `M` matched-fitting test,
`A` learning-curve envelope.

### E2 — theoretical capability without practical advantage

**Explanation:** Recurrence can represent the relevant temporal function but
its inductive bias supplies no material finite-data or finite-compute advantage
for this task.

**Present evidence:** `OPEN`. The sandbox did not show an LSTM advantage, but
underfitting prevents separation from E1.

**Predictions:** After both models fit benign data adequately, recurrent and
dense models have similar paired rankings and metrics; additional recurrence-
specific compute does not yield a stable advantage.

**Discriminating tests:** matched-fitting FC/LSTM comparison; paired seeds and
partitions; data-efficiency and capacity curves; predeclared smallest effect of
interest.

**Feeds:** `M` and `A`.

### E3 — dense models already exploit fixed-position temporal structure

**Explanation:** A dense AE receives 48 ordered coordinates and can learn
relationships among specific time positions. LSTM supplies parameter sharing
and a sequential inductive bias, not exclusive access to temporal information.

**Present evidence:** `SUPPORTED AS A REPRESENTATIONAL FACT`; `OPEN` as the
explanation of paper performance. The sandbox dense AE detected block
disruption at AUC 0.950.

**Predictions:** Dense behavior changes when order or time-position identity is
destroyed; dense and LSTM models can both learn fixed-alignment witnesses;
LSTM advantages, if any, appear mainly under shifts, limited data, variable
length, or transfer where its inductive bias matters.

**Discriminating tests:** normal-order versus globally permuted-coordinate
training; within-row shuffling; shifted/localized-event generalization; learned
score sensitivity by time position; matched-capacity comparisons.

**Feeds:** `M`.

### E4 — the evaluated task does not require the claimed temporal capability

**Explanation:** Simple marginal changes or fixed-position cues solve most
attacks, so temporal modeling is unnecessary even if it is available.

**Present evidence:** `SUPPORTED IN TOY DATA ONLY`. One-dimensional statistics
gave AUC 0.993–1.000 on toy attacks 1–5; reversal defeated multiset-only rules.
The 2026-08-31 FC-SAE diagnostics do not establish exact-data task triviality:
trained scores improve on the tested simple controls within energy bands, and
no recurrence/attention comparison was performed. Small aggregate gains alone
cannot establish that the claimed temporal capability is unnecessary.

**Predictions:** Exact-data simple rules approach elaborate models on attacks
1–5; performance falls most on reversal; destroying temporal order has little
effect on aggregate results dominated by shortcut attacks.

**Discriminating tests:** exact-data zero-parameter/one-feature rules through
the identical split, threshold, and metric path; per-attack results; reversal;
structure destruction while preserving marginals.

**Feeds:** `M` triviality floor and later `A` context.

### E5 — full-configuration confounding produces the ordering

**Explanation:** LSTM versus FC also changes depth, width, activation, output
domain, optimizer, dropout, and possibly training behavior. Any advantage may
come from those joint differences rather than recurrence.

**Present evidence:** `SUPPORTED AS A SOURCE-DESIGN LIMITATION`; `OPEN` as the
cause of any numerical gap. Table I does not provide a recurrence-only
comparison.

**Predictions:** The paper-configured models may differ while a matched
recurrence-only comparison shrinks, disappears, or reverses; individual
activation/output/optimizer changes explain substantial score variation.

**Discriminating tests:** first reproduce complete source configurations; then
run matched heads, optimizer, dropout, depth/capacity, stopping, and scoring
with recurrence as the isolated factor. Change one declared factor at a time.

**Feeds:** `M`; source-supported branches may also feed `N`.

### E6 — a materially different undocumented procedure was used

**Explanation:** The implementation that produced the table may contain data,
model, training, scoring, threshold, or selection details absent from or
different from the publication.

**Present evidence:** `OPEN AND NON-IDENTIFIABLE FROM NON-REPRODUCTION ALONE`.
No author code is available. Source omissions make this plausible but do not
show that such a procedure exists.

**Predictions:** Reasonable source-supported completions consistently miss the
target while some non-source procedure could match it; published clarification
or code would reveal consequential differences.

**Discriminating tests:** complete clean-reader reproduction across a finite
predeclared completion space; provenance requests or later author artifacts;
compare any disclosed procedure without inferring it from target proximity.

**Feeds:** bounded `N` conclusion and residual uncertainty. It can remain open
indefinitely and must never become an accusation by elimination.

## Additional materially plausible explanations

### E7 — output-domain geometry dominates reconstruction scores

Softmax/sigmoid outputs cannot exactly reconstruct general standardized
profiles. **Current status:** fixed-preparation Softmax/MSE attainability limit
`VERIFIED CONDITIONALLY`; “nothing useful learned” `WEAKENED, NOT ESTABLISHED`.
The 2026-08-31 label-aware relaxation bounds balanced ACC below 50.93% and DR
at 9.25% when FA <=15%, across every allowed reconstruction. This is a
float64 evaluation with outward padding, not certified interval arithmetic.
Changing the preparation/output/score can escape the bound.
Despite global trained/energy correlation 0.999253, original-row trained-minus-
zero ACC is +0.89081 points [0.80454, 0.98117]. In the small adaptive control,
within-energy AUC is 65.49% trained, 62.18% projection, 55.02% uniform, and
49.74% energy. The latter differences have no uncertainty estimate and do not
isolate learning causally. Geometry constrains performance without making all
scores equivalent. See [the finding](POST_ANCHOR_FINDING.md).

### E8 — score direction, reduction, or threshold semantics differ

The VAE inequality may be a typo; MSE may use sum versus mean; threshold
construction is not executable; and a printed constant may have been applied
under a different scale. **Current status:** source ambiguity `VERIFIED`, cause
`OPEN`. On the audited clean-reader score vector, threshold choice alone in
the printed direction cannot exceed 50.00072% balanced ACC; reversing direction
cannot exceed 60.21%, versus 83% reported. The subsequent domain relaxation
closes the stronger fixed-preparation Softmax/MSE rescue for any weights:
best balanced ACC is below 50.93% in the printed direction, 64.56% reversed.
A positive constant sum-versus-mean rescaling cannot alter rankings or an
all-cutoff ROC region; other score definitions or preprocessing remain open.
Preserve literal and anomaly-consistent directions, explicit score
reductions, printed threshold, and finite source-supported alternatives.

### E9 — preprocessing, leakage, resampling, or identity creates the effect

Joint pre-split scaling, test-set ADASYN, customer/day identity, attack
population, and validation leakage can alter apparent performance. **Current
status:** source procedure/ambiguity `VERIFIED`, causal contribution `OPEN`.
The clean-reader printed-position ADASYN executed, preserving all original
test rows and adding 3,629,620 benign rows. Its integer allocation yielded
4,380,387 benign versus 4,504,602 malicious rows, not exact class equality.
Customer and row identities passed the artifact audit; this does not establish
that the entire preprocessing protocol is scientifically unbiased.
If promoted at Checkpoint 2, compare corrected train-fitted and untouched-test
controls one factor at a time without replacing the completed numerical row.

### E10 — data identity or representation differs

The exact customer subset, date range, missing/DST policy, SGCC mapping, and
allocation serialization may differ. **Current status:** `OPEN`; the six exact
official consumption bytes are available, the exact allocation `.tab` remains
absent, and the independently cross-checked semantic allocation branch was
approved before its anchor outcome on 2026-08-30. Test only source-supported
identities and never choose the one nearest the target after seeing results.

### E11 — our reproduction implementation is wrong

A clean-reader non-match may be caused by a parser, attack, split, model,
training, score, or metric defect in our code. **Current status:** `WEAKENED FOR
THE AUDITED CLEAN-READER CHAIN, NOT ELIMINATED IN PRINCIPLE`. Phase 4 corrected
identified mismatches; all 179 deterministic tests pass. The frozen exact-data
audit regenerates every metric/count with zero discrepancy. Full array scans,
customer/feature identities, stopping replay, finite weights, and fresh-load
sample scores passed. A supplemental checker last-chunk slicing bug was
preserved and corrected without altering reproduction code or artifacts.
Remaining source-reading mistakes or untested implementation defects remain
possible; the audit is not proof of every conceivable semantic detail.

### E12 — the sandbox witness is weak or non-transferable

The cyclic toy data, architecture, single seed, or reconstruction objective may
fail to expose temporal capability that matters on ISET. **Current status:**
`SUPPORTED AS A LIMITATION`. Do not retry adaptively. A later formal witness
must have competing predictions, matched fitting, and a declared effect before
execution.

### E13 — finite-sample variation or result-selection effects

Unreported seeds, partitions, failed runs, hyperparameter trials, rounding, or
selection could create a tidy table even without misconduct. **Current status:**
seed/weight/optimizer rescue `CLOSED WITHIN FIXED PREPARATION/SOFTMAX/MSE SPACE`
by the output-domain bound, including target rounding. Variation involving
other prepared inputs or scores remains `OPEN`; the source reports no
dispersion or run count. Do not repeat seeds to answer the closed conditional
question. No inference about the authors' selection history follows.

### E14 — reporting or transcription error

Some inequalities, labels, formulas, or table cells may be typographical or
copied incorrectly while the underlying experiment was ordinary. **Current
status:** `OPEN`; the VAE direction is a prime candidate. Test internal
arithmetic identities, both coherent orientations, and any later erratum or
author clarification. A typo can explain one inconsistency but cannot be used
as a blanket repair for the entire target pattern.

### E15 — reported targets are outside ordinary attainable behavior

Even after source-supported completions, seeds, capacity, training, and
threshold choices, the target may remain far outside observed plateaus.
**Current status:** `CLOSED WITHIN FIXED PREPARATION/SOFTMAX/MSE SPACE` by an
analytic domain relaxation, not by extrapolating the failed seed. Even an
independent label-aware reconstruction per input cannot reach the target;
original-row and reversed-direction controls also miss it. This is stronger
than empirical implausibility within that fixed space. Other source-supported
preparations, output domains, and scores remain `OPEN`; no broader empirical
envelope or years-of-search estimate has been measured. Empirical saturation
would still not establish universal impossibility.

The recurrent two-epoch score recovery adds no plateau: its fixed pilot scores
remain far below the target and batch arithmetic does not rescue them, but the
training objective was still improving. The A16 full-anchor projection exceeds
the approved budget. The predeclared H200 cost pilot could not enter the
partition because the account lacks an allowed QOS, so no new learning curve
was produced and the full anchor was not promoted.

## Ordered test map

**Initial 2026-08-31 checkpoint:** the initial audited result and full metrics are in
[CLEAN_READER_FINDING.md](CLEAN_READER_FINDING.md). E1–E6 and E12–E14 were not
newly discriminated by a recurrent comparison or repeated seeds. Do not treat
the first numerical finding as their elimination.

1. **Phase 4 fidelity (`E11`):** prove or correct the five-file route against
   the frozen specification.
2. **Phase 5–6 numerical anchor (`E6`, `E8`–`E11`):** run once, audit completely,
   and compare the full FC-SAE row.
3. **Checkpoint 2:** use the anchor to decide whether additional `N`, `M`, or
   `A` information is worth its cost.
4. **Exact-data triviality floor (`E4`, `E7`, `E9`):** simple rules and domain
   floors through the same held-out path.
5. **Source-configured FC/LSTM numerical comparison (`E1`, `E5`, `E6`):** test
   whether the published ordering appears at all.
6. **Matched mechanism comparison (`E1`–`E5`, `E12`):** equalize fitting,
   capacity, heads, optimizer, and score; manipulate temporal structure.
7. **Finite numerical/attainability depth (`E13`, `E15`):** only if promoted,
   run frozen seeds/partitions/axes and preserve distributions and failures.

No later result may delete an explanation from this register. Close it with the
tested space, evidence, and remaining escape conditions stated explicitly.

## Completed post-anchor update — 2026-08-31

The user approved the bounded sequence; primary code `1175e8d` and adaptive
control code `26a42db` ran as CPU jobs `385090` and `385091`. Both completed
without training. [The saved finding](POST_ANCHOR_FINDING.md) links the
contracts, all results, and hashes. E7/E8/E13/E15 now have the conditional
closures above. E4 is not upgraded to an exact-data triviality finding, and
E1–E6/E9–E12/E14 are not eliminated. No author procedure is inferred.

Stop the diagnostic round. The next proposed work is a source-supported
assumption map: which changes actually leave the bounded input/output/score
space, and which are corrections rather than reproduction? No broader
experiment is automatically promoted by this update.

## Source-assumption follow-up — 2026-08-31, awaiting discussion

The user approved the source review and capped no-training checks but required
discussion before public updates. Code `b76cb02`, CPU job `385119`, completed
the full original-row analysis in 56.76 seconds. See
[SOURCE_ASSUMPTION_FINDING.md](SOURCE_ASSUMPTION_FINDING.md).

- **E7, output geometry:** source support for FC-SAE Softmax and high-error
  MSE is verified again, not inferred from our code. A Sigmoid cube control
  changes the original-row ACC ceiling from 50.16% to 80.84% (upward-rounded
  limits). It still fails the combined target in the printed direction;
  with reversal the relaxation does not exclude DR>=81%, FA<=15%. Thus the
  restrictive combination matters; no universal bounded-output or learned-
  mechanism conclusion follows.
- **E8, score units:** SSE and RMSE are monotone transforms of MSE and cannot
  change the existing all-cutoff ROC region on the complete fixed evaluation.
  This does not rule out genuinely different scoring procedures.
- **E9, normalization:** the fitted scope is not explicitly specified. Two
  further interpretations were tested: joint scalar and weaker separate-class
  feature-wise scaling. Their optimistic DR at 0.58 is bounded by 29.81% and
  33.96%, versus 81%. These fixed-cutoff rescues are closed for the same attack
  population, even if benign-only ADASYN is regenerated. Other original-row
  all-cutoff limits do NOT automatically transfer to regenerated ADASYN.
- **E10/E11/E14:** no newly discovered source/code mismatch explains away
  Softmax or MSE. Other populations, interpretations, untested defects, and
  possible reporting errors remain open; no author implementation is inferred.
- **E15:** the fixed-cutoff bound covers three explicit normalization readings,
  not every reasonable preparation or an infinite hyperparameter search.
  No new `N` reproduction or matched `M` comparison was performed.

Experiments are stopped. Findings are saved locally; no README/site/report
update or push is authorized before discussion.

## Sigmoid follow-up — 2026-08-31, awaiting discussion

After the preceding findings were discussed and published, the user approved
a quick Sigmoid investigation. Code `9d6c31b`, job `385137`, completed the full
no-training check in 39.70 seconds. See [the finding](SIGMOID_SANITY_FINDING.md).

- **E7, output geometry:** Sigmoid + printed 0.58 still fails on the complete
  prepared evaluation (minimum FA 29.66640%). Unlike the original-row check,
  Sigmoid + a changed high-error cutoff is not excluded: the complete-data
  bound gives DR up to 85.32587% at FA<=15%. Reversal is open too. Do not
  generalize the Softmax exclusion or the original-row Sigmoid exclusion.
- **E8, threshold/scoring:** a cutoff change cannot rescue fixed Softmax
  scores or the fixed-input Softmax range, but may matter after changing the
  output range. The label-aware Sigmoid cutoff near 1.24143 is not a trained
  model threshold or evidence of author behavior.
- **E9, evaluation population:** unchanged synthetic benign rows change the
  population over which FA is computed and change the all-cutoff Sigmoid
  answer. The original-row bounds reproduce; no original result is invalidated.
- **E4/E15, triviality and attainability:** two label-blind reconstructions
  do not reach the target. The larger label-aware ceiling is not achieved by
  these controls, and no trained Sigmoid model has been tested. The learned
  alternative remains unresolved; no zero-useful-work claim follows.
- **E10/E11/E14:** no author implementation or intent is identified. The
  result keeps a changed-output/changed-cutoff explanation open without
  asserting that it occurred or matches the complete seven-metric row.

All new records remain local until discussed. No training, new branch, or
publication is automatically promoted by this completed check.

## Small trained Sigmoid follow-up — 2026-08-31, awaiting discussion

The user subsequently requested testing the remaining alternative. The
[paired-fit setup](SIGMOID_FIT_CHECK.md) was frozen in `cc9af5e`; CPU job
`385198` completed the ten-epoch pair in a 24.81-second analysis. See
[SIGMOID_FIT_FINDING.md](SIGMOID_FIT_FINDING.md). This is exploratory `X/A`.

- **E7, output geometry:** genuinely training the Sigmoid head does not
  realize the permissive bound under the small declared setup. This contrasts
  two actual fitted models, not just allowed output sets. It does not show
  that every Sigmoid model fails or identify a claimed architectural mechanism.
- **E8, cutoff/direction:** closed for the two selected score vectors on the
  12,119-row evaluation sample. Max DR at FA<=15% is Softmax 8.64258%, Sigmoid
  9.74935%; reversing gives 25.52083% and 25.39063%. No cutoff reaches 81%,
  including relaxed rounding. Different fitted scores remain an escape.
- **E6/E13/E15, fitting and attainability:** both completed 640 updates and
  ten epochs without hitting the budget. Sigmoid's best benign-calibration
  checkpoint is epoch 10; loss continued improving after a marked epoch-5/6
  change. The brief budget is therefore not a demonstrated long-run plateau.
  Larger-data/longer-budget fits remain open, not promoted automatically.
- **E4/useful work:** no zero-useful-work or task-triviality conclusion.
  Sigmoid learned to reconstruct better while high-error AUC declined; reversed
  AUC improved. Those distinct observations need preservation, not a blanket
  claim that training did nothing. There is no matched temporal ablation here.
- **E9–E11/E14:** data/preparation were not changed. No author implementation,
  hidden procedure, reporting mistake, or intent is identified by this failure.

The finite fitted-model cutoff rescue is closed; the broader Sigmoid
alternative remains unresolved. All outcomes are local. Stop for discussion
before publication or any additional experiment.
