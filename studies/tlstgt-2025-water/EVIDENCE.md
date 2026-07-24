# TL-STGT reproduction — evidence and corrections

> **Read [`README.md`](README.md) first** — it carries the audit framing this document assumes.

Durable causal record for the water-distribution paper (EUSIPCO 2025). Format per
project discipline: former belief → evidence → root cause → current conclusion →
confidence and uncertainty. Labels: **VERIFIED** (measured, reproducible),
**OBSERVED** (measured once), **INFERRED**, **HYPOTHESIS**, **INVALIDATED**, **OPEN**.

> **EXECUTION PROVENANCE (2026-07-24).** The user's execution policy — originally
> written for Paper 1, widened this day to cover *every* experiment — requires all
> preparation, training, and scoring to run on the cluster compute nodes, never on the
> local Mac, and makes locally-produced results **ineligible as experimental
> evidence**. Every number in C1–C4 and C6 below was produced locally before that
> widening and is therefore **PROVISIONAL** pending cluster re-execution
> (the recorded cluster jobs, seed 0). C5 is unaffected: it is arithmetic on
> the paper's own printed numbers and involves no execution.
>
> The cluster environment differs from local (torch 2.13.0+cpu, sklearn 1.9.0,
> lightgbm 4.7.0, numpy 2.5.1, pandas 3.0.5) in a separate CPU-only venv at
> the study's own virtualenv, deliberately isolated from
> `~/atk-evidence/.venv` so Paper 1's environment provenance is not perturbed.
> Each job runs `test_detect.py` as a preflight and logs SHA-256 of code and data.
> Small numeric differences from the local numbers are expected; conclusions that
> flip would themselves be a finding.

---

## C1. RETRACTED — "our faithful implementation lands ~46 F1 points below the paper"

**Former belief.** Runs through 2026-07-23 gave STGT-31 F1 ≈ 30.2 / ACC 60.4 / DR 17.9
against the paper's 76.3 / 84.8 / 74.7, and the robustness sweep found no
interpretation that closed the gap except at 19–32% false alarms. This was being
treated as evidence about the paper.

**Evidence that overturned it.** Boundary-divergence probing (2026-07-24) showed
FFNN/LSTM/TGCN/STGT/TL-STGT producing *byte-identical* F1/ACC/DR at every graph
size — impossible for genuinely different architectures. Instrumenting each
pipeline boundary localized the collapse and exposed a defect in our own detector
(C3), not in the paper.

**Current conclusion. [VERIFIED]** The reported gap was **substantially our own
bug**. With the threshold defect fixed, STGT-31 (σ branch, S=1) reaches
**F1 70.0 / ACC 74.2 / DR 62.9** against the paper's 76.3 / 84.8 / 74.7 — roughly
6 F1 points apart, not 46. The earlier "cannot get close" claim is **retracted**.

**Uncertainty.** S=1 is *not* paper-literal (the paper prescribes batch averaging;
at its implied S=32 we get F1 63.8 / ACC 62.8). Our F1 = 70.0 also carries
FA = 15.4%, which the paper never reports. So "we now roughly reproduce it" would
equally overstate the evidence. Full corrected table pending (C5).

---

## C2. The paper's eq-4 attack magnitude is scale-incoherent [VERIFIED]

**Evidence.** C-Town benign per-sensor standard deviations span a 6,600× range:
J280 = 0.0063, PU2 = 41.7. Equation 4 adds an absolute δ ∈ [−5, 5] to raw
readings, so the *same* δ is:

| sensor | benign sd | δ=5 in sigma units |
|---|---|---|
| J280 | 0.0063 | **788.5 σ** |
| T6 | 0.180 | 27.8 σ |
| PU8 | 17.5 | 0.29 σ |
| PU2 | 41.7 | 0.12 σ |

A single 788σ perturbation contributes ~6.2×10⁵ to ξ² against a benign χ²₃₁
baseline of ~31.

**Effect on the residual. [VERIFIED]** During a manipulation attack the residual
`observed − predicted` is dominated by the injected perturbation (~788σ) rather
than by model error (~1σ), so in that extreme tail the residual *equals the
attack*, independent of architecture. Measured: predictions differ substantially
across models (corr 0.45–0.72), yet residuals agree at corr 0.9997, scores at
0.999995, batch-means at 1.000000.

**CORRECTED ATTRIBUTION. [VERIFIED 2026-07-24, later same day]** I first recorded
this scale-incoherence as *the* cause of the identical-model collapse. That was
wrong. The full corrected run shows models differing widely **in the raw branch**,
which still contains the 788σ J280 problem — e.g. at 10 nodes, S=1: FFNN F1 72.1
vs TGCN 44.6. So δ scale-incoherence alone does **not** produce identical metrics.
The collapse was caused by the inflated threshold (C3), which admitted *only* the
extreme model-independent tail into the decision; fixing the threshold alone
resolved it. C2 remains a real and serious property of the paper's attack spec —
it is simply not what made our metrics identical.

**Consequence.** Replay and DoS are temporally consistent (perturbation median
0.075σ, p99 2.7σ) and are essentially invisible: **DR = 0.0% for replay, 0.0% for
DoS, 62.5% for manipulation — identical in every model.** Detection answers "did a
manipulation hit a low-variance sensor?", a question with no dependence on the
model.

**Note.** J280 (788σ) and T6 (27.8σ) appear **only in the 31-node graph**, which is
where the paper claims its largest margin.

**Handling.** Paper-literal track keeps raw δ. A documented branch
(`delta_scale="sigma"`, same [−5,5] grid read in per-sensor benign σ) is run
alongside — **not** a repair of the literal track. Under it, architectures finally
separate (decision agreement 82–91%, was 100%).

---

## C3. OUR BUG (fixed) — threshold selected over contaminated "normal" samples [VERIFIED]

**Defect.** `detect.evaluate_forecaster` selected threshold samples with
`val_labels == 0`, which labels by the *current timestep only*. Two contaminations
followed: (a) a sample's 10-step history window can still contain attack readings
(80 of 435 validation normals); (b) `batch_mean` averages the trailing S samples,
so a normal sample within S steps after an attack block inherits the attack's
score. `data.py` already computed `win_clean` and used it for `train_benign`, so
this was an internal inconsistency.

**Measured impact. [VERIFIED]** Threshold inflated ~8,256× in a controlled
synthetic case; realized false-alarm rate **0.4% against a 5.0% target** on real
data, and **0.00% against 5%** in the regression test.

**Fix.** `detect.clean_batch_mask` restricts threshold selection to wholly-normal
batches (no attack label and no dirty history window anywhere in the trailing S).
Wired through `data.py` → `run.py` / `rob.py`. Regression suite `test_detect.py`,
**3 test functions pass**; calibration now realizes 0.81 / 4.99 / 10.88% against
1 / 5 / 10% targets.

**Note.** Fixing this did *not* fix the model collapse (C2 did) — it corrected the
operating point, which is what invalidated C1.

---

## C4. The paper's batch averaging manufactures false alarms [VERIFIED]

**Evidence.** Detection FA tracks the batch size S, not model quality. STGT-31,
σ branch, 60-step attack operations:

| S | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| FA % | 15.4 | 15.4 | 16.0 | 18.9 | 24.8 | 42.3 | 64.9 |
| ACC % | 74.2 | 73.9 | 74.1 | 72.6 | 70.3 | 62.8 | 56.7 |

**Root cause. [VERIFIED]** A trailing mean over S smears each attack block's score
across the following S normal samples, so FA ≈ S / block-length (32/60 ≈ 53%
predicted, 42.3% measured). Accuracy decreases monotonically in S: the paper's own
prescribed averaging actively degrades its proposed detector.

**Open. [OPEN]** The paper says "flag the batch," which admits a second reading —
non-overlapping batches with one decision each — that would not smear across block
boundaries. Our trailing per-sample mean preserves a per-sample balanced metric and
is documented (D3 AMBIG). Both readings deserve reporting.

---

## C5. STANDING — the reported table is arithmetically impossible [VERIFIED]

Independent of any implementation, because it uses only the paper's own numbers on
its own stated 50/50 balanced split, where ACC = (DR + SP)/2 and SP ≤ 100%.

Verified by re-running `metric_consistency_check.py` (2026-07-24):

- **24 of 27 cells violate the ceiling ACC ≤ (DR+100)/2**, i.e. require
  specificity > 100%. Within the 9 graph/proposed cells specifically, **7 of 9**
  violate it: TGCN-10 (DR 51.8, ACC 79.3) → SP = 106.8%; TGCN-31 (54.8, 80.4) →
  106.0%; STGT-20 (55.6, 83.7) → 111.8%; TL-STGT-20 (59.6, 86.1) → **112.6%**.
- **27 of 27 cells fail the stricter joint DR + F1 → ACC check**, including the
  three that clear the ceiling. STGT-31: DR 74.7 with F1 76.3 forces SP = 78.9%
  and ACC = **76.8%**, not the reported 84.8%. TL-STGT-31: implied ACC **80.4%**
  vs reported 87.1%. LSTM-31: implied **10.0%** vs reported 64.7%.
- **The implied prevalences contradict the stated split.** The single attack
  fraction that would reconcile each row ranges from 13.6% to 33.0% and disagrees
  row to row — none of them 50%.

**This is the load-bearing finding.** It survives every correction above because it
never depended on our code — it uses only the paper's own three numbers per cell
on its own stated balanced split.

---

## C6. The paper's principal result PATTERN does not reproduce [VERIFIED]

Corrected full run, exact Figure-1 graphs, raw (paper-literal) δ. The paper's three
central claims, quantified in F1 (10→20→31):

| claim | paper | ours (S=1) | ours (S=32) |
|---|---|---|---|
| STGT improves with graph size | 52.3→59.4→76.3 (**+24.0**) | 70.8→67.6→70.8 (**+0.0**) | 64.2→58.6→66.1 (+1.9) |
| TL-STGT improves with graph size | 54.8→65.1→79.7 (**+24.9**) | 70.9→69.1→70.4 (**−0.5**) | 63.2→57.7→64.7 (+1.5) |
| STGT beats FFNN | **+31.4 / +35.3 / +40.2** | **−1.3 / −3.9 / −0.3** | +0.2 / −3.6 / +2.3 |
| TL-STGT beats STGT | +2.5 / +5.7 / +3.4 | +0.1 / +1.5 / −0.4 | −1.0 / −0.9 / −1.4 |

**Conclusion.** Individual magnitudes are approachable — our STGT-31 reaches F1 70.8
against the reported 76.3 — but **the pattern that constitutes the paper's
contribution does not appear at all**:

1. **No size trend.** STGT is flat (+0.0) where the paper reports +24.0. Notably
   TGCN *does* improve with size in our run (+24.8), so the pipeline is capable of
   showing the effect; STGT simply saturates near F1 70 already at 10 nodes.
2. **No advantage over a plain feedforward net.** The paper's headline margin of
   +31 to +40 F1 over FFNN becomes **−1.3 to −0.3** in ours. STGT never beats FFNN
   at any size at S=1.
3. **No transfer-learning benefit.** TL-STGT ≈ STGT everywhere, and is slightly
   *worse* at the paper's own batch size.

**Collateral observation. [OBSERVED]** The paper's baselines are implausibly weak:
it reports FFNN at F1 20.9/24.1/36.1, while our FFNN — same stated architecture
(5 layers, 500 units, tanh, Adam) — reaches 72.1/71.5/71.1. Under-performing
baselines are what create the reported margin.

**Uncertainty.** Our numbers carry realized FA of 6.8–17.8% (S=1), which the paper
never reports, so the comparison is not like-for-like on the operating point. Single
seed (0); seed variation not yet characterized. σ-branch results pending.

---

## C7. Zero-parameter rules match or beat every trained model [VERIFIED]

Breadth pass + aimed depth pass, 31 nodes, cluster jobs 355305–355307. All
detectors run through the identical scoring → threshold → metric path.
**Comparisons are at a matched false-alarm rate of 3.7%**, which the earlier
reporting did not do — trained models had been quoted at FA 12–15% against a
trivial detector at 3.7%, inflating their apparent advantage.

| detector | params | AUC | DR | replay | DoS | manip |
|---|---|---|---|---|---|---|
| ZSCORE | 0 | 0.635–0.640 | 31.0–31.2 | 5.6–6.1 | **0.0** | 100 |
| PERSIST | 0 | 0.512–0.515 | 30.5–31.0 | 4.4–5.6 | **0.0** | 100 |
| FFNN | ~1.3M | 0.776–0.783 | 42.9–57.1 | 0.6–1.7 | 49.2–97.5 | 100 |
| STGT | ~0.4M | 0.772–0.779 | **31.7–56.9** | 2.2–3.3 | **5.8–95.8** | 100 |
| stuck-sensor rule | **0** | **0.987** (DoS) | — | — | **93.3** | — |
| composite (z ∨ stuck) | **0** | — | **60.5** @ FA 7.9 | 12.2 | 93.3 | 100 |

**Findings.**

1. **The only thing training buys is DoS detection.** On replay the trained
   models are *worse* than a zero-parameter rule (0.6–3.3% vs 4.4–6.1%); on
   manipulation everything ties at 100%. Every point of the trained models'
   advantage sits in the DoS column.
2. **DoS is trivially detectable by the right rule, and invisible to the paper's
   method.** "Has the reading stopped changing" separates DoS from normal at
   **AUC 0.987** (frozen fraction 0.936 vs 0.234). The paper detects via one-step
   forecast residual, which is *structurally blind* to DoS: fed frozen history a
   forecaster predicts the frozen value, so the residual vanishes. The
   zero-parameter forecast-residual detectors score exactly **0.0%** on DoS.
3. **A composite of two trivial rules matches every trained model.** z-score ∨
   stuck gives **F1 71.5 / ACC 76.9 / DR 60.5 at FA 7.9%**, versus our trained
   models at F1 ≈ 70 and *higher* FA — with zero parameters and no seed variance.
4. **The proposed model is the least stable.** STGT's DoS detection swings
   5.8% → 95.8% across three seeds (FFNN 49.2 → 97.5). The trivial rule is
   deterministic at 93.3%.
5. **Nobody detects replay.** Best observed is the zero-parameter z-score at
   6.1%.

**Quantitative consequence for the paper's DR. [VERIFIED]** The test mix is
≈180 replay / 120 DoS / 120 manipulation (42.9% / 28.6% / 28.6%). Perfect
manipulation *plus* perfect DoS caps DR at **57.1%**. The paper reports
**DR = 74.7%**, which therefore requires detecting ≈**41% of replay**. The best
replay detection observed by anything tested here — trivial or trained — is
**6.1%**.

**Bearing on the paper's stated mechanism.** The paper justifies its architecture
by claiming spatial/temporal structure is too hard for an FFNN. Structure does
exist (spatial |corr| 0.18–0.24, temporal lag-1 0.65–0.74 on attack-free
windows), so the premise is not empty — but exploiting it is not what separates
the models. The task decomposes into one class needing no model, one class needing
a two-line rule the paper's method cannot express, and one class nobody detects.

---

## C8. Forensics on the printed numbers [VERIFIED]

`../water-paper-analysis/forensics.py`. All tests operate on Table I alone and
involve no execution.

1. **Attainability. 0 of 27 cells** are realisable from *any* integer confusion
   matrix on a balanced test set, for every per-class size tried
   (A ∈ {70 … 700}; the paper's own description implies A = 70). The test was
   validated against 15 constructed confusion matrices and accepted all 15, so
   the rejection is not an artefact of the checker.
2. **Direction of error is systematic, not random.** Of the 25 cells where the
   joint DR+F1→ACC identity is evaluable, **25 report an accuracy ABOVE what
   their own DR and F1 imply, and 0 below**. Mean inflation **+30.6 points**
   (range +6.7 to +54.7). Sign test vs chance: **p = 6.0 × 10⁻⁸**. Random
   transcription or rounding error would scatter in both directions.
3. **Monotonicity.** F1 and DR increase strictly with graph size for **9/9**
   models; ACC for only 5/9. Under a no-effect null, P = (1/6)⁹ ≈ 1 × 10⁻⁷ per
   column — but a genuine strong size effect produces the same pattern, so this
   is weak on its own.
4. **Terminal digits** are non-uniform (χ²=19.9, 9 df, p<0.05; digit "1" appears
   17 times against an expected 8.1). Low power at n=81; treat as suggestive only.

**What this establishes.** The printed numbers cannot have been produced by
running the described experiment and computing the stated metrics. The deviation
is one-directional and large, which excludes isolated typos, rounding, and random
transcription error.

**What it does NOT establish. [OPEN]** How the numbers arose. The pattern
— high ACC with low DR — is also exactly what computing accuracy on a *naturally
imbalanced* test set would produce, contradicting the stated 50/50 split. But
that explanation does not fully work either: the single attack prevalence that
would reconcile each row varies from **13.6% to 33.0%** across rows, so no single
test-set composition explains the table. Remaining possibilities include metrics
computed on differently-constructed sets per row, values imported from another
experiment, or fabrication. **These cannot be distinguished from the artefact
alone, and this record does not assert intent.**

---

## C9. Replay is undetectable in principle; FFNN is not capacity-limited [VERIFIED]

cluster jobs 355309 / 355312, 31 nodes.

**Replay.** Nearest-neighbour distance from each held-out test window to a bank
of 2,900 attack-free *training* windows:

| class | median distance | AUC vs normal |
|---|---|---|
| normal | 9.92 | — |
| **replay** | **9.75** | **0.455 (chance)** |
| DoS | 16.31 | 0.917 |
| manipulation | 1422.40 | 0.988 |

Replay is **indistinguishable from benign data** — by construction, since eq. 2
replays genuine past readings, so a replayed window *is* a real benign window.
Manipulation and DoS are highly separable. The replay ceiling is therefore a
property of the **data**, not merely of the paper's detector: no architecture can
exceed it.

*(First run of this test was invalid — the reference bank included the evaluated
normals, which matched themselves at distance 0 for 56% of cases and inverted the
comparison. Bank rebuilt from training windows only.)*

**Consequence for the reported DR.** With replay undetectable, DR is capped at
approximately `1 − (replay fraction)`. Under our attack mix (43% replay / 29% DoS
/ 29% manipulation) that ceiling is **57.1%**, against a reported STGT DR of
**74.7%** and TL-STGT **76.8%**. **Caveat:** the paper never states its attack
mix, so the exact ceiling is not pinned — with less replay the ceiling rises. The
robust claim is conditional: any mix containing a substantial replay share caps DR
well below the reported figures.

**Capacity.** FFNN across a **272× parameter range**, DR at matched FA 3.7%:

| params | 19,681 | 54,431 | 1,173,031 | 5,347,031 |
|---|---|---|---|---|
| AUC | 0.790 | 0.785 | 0.761 | 0.787 |
| DR | 56.7 | 56.9 | 41.9 | 56.2 |

Flat. A 2-layer, 19.7k-parameter network matches a 5.3M-parameter one and matches
STGT (AUC 0.78). The paper's stated justification — that this structure is "too
difficult for an FFNN" — **fails on its own terms**: the FFNN is not
capacity-limited, so capacity was never the missing ingredient.

---

## C10. Exhaustive protocol search — no protocol reproduces the table [VERIFIED]

`src/protocol_search.py`. Impossibility under the protocol a paper *describes*
is escapable: a defender can argue the description was inaccurate. This tests the
stronger claim. Under H0 ("the reported triple arose from a real confusion matrix
under some protocol"), the search covers **67,326 protocols per cell**: test-set
sizes 20–1000, class prevalences 2–98%, and six metric-definition variants
(standard, balanced accuracy, DR-as-precision, negative-class F1, macro F1,
DR-as-specificity).

| Result | |
|---|---|
| Cells unreachable under **every** protocol searched | **20 / 27** |
| Best single protocol explains | **1 cell of 27** |
| Protocols explaining the whole table | **0** |

For those 20 cells P(observed | H0) is exactly zero, not small. The joint
constraint is decisive: a paper has one protocol, not one per row. The searcher
was validated by constructing confusion matrices at several sizes and
prevalences, rounding their metrics, and confirming recovery in every case.

**Limit. [OPEN]** This does not distinguish deliberate fabrication from
catastrophic process failure — placeholder values never replaced, a table carried
from a draft. Both produce numbers that were never measured. That distinction
requires evidence outside the artefact, which this audit does not have.

---

## C11. Graph topology and transfer learning DO help — partially confirming the paper [VERIFIED]

Reported because it disconfirms this audit's own expectation. Two seeds, F1.

**Graph topology carries signal.** Shuffling the C-Town adjacency (same edges,
permuted labels) or removing edges entirely degrades performance:

| model | real | shuffled | identity |
|---|---|---|---|
| STGT @ 10 nodes | **57.9** | 51.9 | 44.9 |
| TL-STGT @ 31 nodes | **63.6** | 45.2 | 44.7 |

Up to 18 F1 points. The graph is **not** decorative — this part of the paper's
premise holds, contrary to what this audit expected.

**Transfer learning shows a large gain** at 31 nodes: TL-STGT over STGT is
**+13.3 to +21.9** across seeds and both batch sizes.

**But both readings are conditioned by C12, below**, and the transfer comparison
is **budget-confounded**: TL-STGT is pretrained on the 10- and 20-node graphs, so
it has seen roughly 3× the training of scratch STGT. A matched-budget comparison
has not been run. **[OPEN]**

---

## C12. The paper's literal training prescription costs ~25 F1 [VERIFIED]

The paper states every split carries equal samples per class, so training data is
50% attack-corrupted. Training instead on benign-only windows (the standard
anomaly-detection reading) changes results dramatically. F1 at 31 nodes, S=1:

| model | balanced 50/50 (literal) | benign-only |
|---|---|---|
| FFNN | 44.1 | 70.3 |
| LSTM | 46.0 | 71.2 |
| TGCN | 44.6 | 70.8 |
| STGT | 46.1 | 67.8 |
| **TL-STGT** | **68.0** | 70.9 |

Two consequences:

1. **Under the paper's own literal reading, every trained model (44–46) scores
   below the zero-parameter composite detector (71.5, C7).**
2. **The transfer-learning gain in C11 is largely an artefact of this.** TL-STGT
   is far more robust to attack-corrupted training data than the others; under
   benign training its advantage collapses from +21.9 to **+3.1**. So transfer
   learning is buying robustness to a bad training prescription, not better
   detection.

**Reporting rule adopted.** Every number in this study must state its
configuration. An unattributed F1 is meaningless when the training-set reading
alone moves it 25 points. An earlier draft of the report and site quoted
benign-training numbers without saying so; corrected.

---

## Status of conclusions

| Claim | Label |
|---|---|
| Reported Table I cannot be produced by any correct implementation | **VERIFIED** (C5) |
| Zero-parameter rules match/beat every trained model at matched FA | **VERIFIED** (C7) |
| The paper's detector is structurally blind to DoS, which a 2-line rule catches at AUC 0.987 | **VERIFIED** (C7) |
| Reported DR 74.7% requires ~41% replay detection; best observed is 6.1% | **VERIFIED** (C7) |
| Principal result *pattern* (size trend, margin over baselines, TL gain) does not reproduce | **VERIFIED** (C6) |
| Paper's eq-4 δ is scale-incoherent across sensors | **VERIFIED** (C2) |
| Paper's batch averaging degrades its own detector | **VERIFIED** (C4) |
| "Our faithful run lands ~46 F1 points below the paper" | **INVALIDATED** (C1) |
| "δ scale-incoherence caused the identical-model collapse" | **INVALIDATED** (C2, corrected attribution — the threshold bug did) |
| σ-branch full table | **OPEN** — running |
| Seed sensitivity | **VERIFIED** (C1 note) — 5 seeds: TGCN spans 23.7 F1, STGT 11.8; single-seed numbers are not quotable |
