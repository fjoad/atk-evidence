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
profiles. **Current status:** structural floor `VERIFIED`; dominance on named
data `OPEN`. Test exact projection floors, score correlation with those floors,
and a linear-output `C` control after the primary reproduction.

### E8 — score direction, reduction, or threshold semantics differ

The VAE inequality may be a typo; MSE may use sum versus mean; threshold
construction is not executable; and a printed constant may have been applied
under a different scale. **Current status:** source ambiguity `VERIFIED`, cause
`OPEN`. Preserve literal and anomaly-consistent directions, explicit score
reductions, printed threshold, and finite source-supported alternatives.

### E9 — preprocessing, leakage, resampling, or identity creates the effect

Joint pre-split scaling, test-set ADASYN, customer/day identity, attack
population, and validation leakage can alter apparent performance. **Current
status:** source procedure/ambiguity `VERIFIED`, causal contribution `OPEN`.
First reproduce it literally; later compare corrected train-fitted,
customer-disjoint, untouched-test controls one factor at a time.

### E10 — data identity or representation differs

The exact customer subset, date range, missing/DST policy, SGCC mapping, and
allocation serialization may differ. **Current status:** `OPEN`; the six exact
official consumption bytes are available, the exact allocation `.tab` remains
absent, and the independently cross-checked semantic allocation branch was
approved before its anchor outcome on 2026-08-30. Test only source-supported
identities and never choose the one nearest the target after seeing results.

### E11 — our reproduction implementation is wrong

A clean-reader non-match may be caused by a parser, attack, split, model,
training, score, or metric defect in our code. **Current status:** `WEAKENED AT
THE STATIC/TINY-TEST LEVEL, OPEN FOR EXACT EXECUTION`. Phase 4 corrected every
identified mismatch, 178 deterministic tests pass, eligible scoring reloads
persisted weights, and the anchor audit independently regenerates metrics.
Exact preparation and artifact inspection can still expose a defect before any
gap is interpreted.

### E12 — the sandbox witness is weak or non-transferable

The cyclic toy data, architecture, single seed, or reconstruction objective may
fail to expose temporal capability that matters on ISET. **Current status:**
`SUPPORTED AS A LIMITATION`. Do not retry adaptively. A later formal witness
must have competing predictions, matched fitting, and a declared effect before
execution.

### E13 — finite-sample variation or result-selection effects

Unreported seeds, partitions, failed runs, hyperparameter trials, rounding, or
selection could create a tidy table even without misconduct. **Current status:**
`OPEN`; the source reports no dispersion or run count. After one anchor, use
predeclared repetitions and preserve all runs if `N` or `A` depth is promoted.

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
**Current status:** `OPEN`; no clean-reader envelope exists. Only a later
predeclared `A` program can support a conditional implausibility conclusion.
Empirical saturation is not universal impossibility.

## Ordered test map

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
