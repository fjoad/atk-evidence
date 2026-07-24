# Ambiguity register — TL-STGT (EUSIPCO 2025)

> **Read [`README.md`](README.md) first** — it carries the audit framing this document assumes.

**Purpose.** The thesis is not merely "the paper does not reproduce as written."
It is stronger: **no plausible reading of the paper produces its reported result
pattern.** To support that, every place the paper is silent, ambiguous, or
self-contradictory is enumerated here, each with the reasonable interpretations a
competent implementer could choose. The experiment then sweeps the cross-product
and reports whether *any* point in that space reproduces the paper.

A branch belongs here only if a competent, good-faith implementer could genuinely
read the paper that way. Interpretations invented to be favourable, or that
contradict an explicit statement, are excluded and noted as such.

Status of each axis: **SWEPT** (in the interpretation matrix), **FIXED**
(resolved by the paper itself, no branch legitimate), or **PENDING**.

---

## Axis 1 — Training-set class balance  ·  SWEPT  ·  paper CONTRADICTS ITSELF

- **Paper says (Sec. II):** 80/10/10 train/val/test "with equal number of samples
  per class in each set." That places 50% attack samples in the *training* set.
- **Paper also says (Sec. V-A):** detection compares forecasts against ground
  truth and fits a Gaussian to the error — a design that presumes the model
  learned *normal* behaviour.
- **Why ambiguous:** a forecaster trained on attack-corrupted targets learns to
  predict the attack, destroying the residual signal the detector depends on. The
  two statements cannot both be honoured naively.
- **Interpretations:**
  - `train=balanced` — literal: train on the stated 50/50 split.
  - `train=benign` — train only on windows with no attack readings (standard
    anomaly-detection practice; what the detector design implies).

## Axis 2 — Units of the manipulation offset δ  ·  SWEPT  ·  paper SILENT

- **Paper says (eq. 4):** `X^m = X^b + δ`, with `−5 ≤ δ ≤ 5` in steps of 0.2,
  "selected based on experimental tuning." No units are given.
- **Why ambiguous:** C-Town sensors span a 6,600× scale range (benign sd 0.0063
  on J280 to 41.7 on PU2). Read as an absolute offset, one δ is a 788σ shove on
  J280 and a 0.12σ nudge on PU2 — an attack of wildly different severity per
  sensor. Read as relative, it is uniform.
- **Interpretations:**
  - `delta=raw` — literal: absolute offset on the reading.
  - `delta=sigma` — same [−5,5] grid in per-sensor benign standard deviations.

## Axis 3 — Detection batch size S  ·  SWEPT  ·  paper UNDERSPECIFIED

- **Paper says (Sec. V-A):** average the squared Mahalanobis distance "over
  consecutive batches" of size S and flag the batch above a threshold. S is
  identified with the batch size; Sec. V-B gives no detection-specific S.
- **Why ambiguous:** S is never given a detection value, and averaging over S
  smears an attack block's score across the following S normal samples, so the
  choice materially changes the false-alarm rate.
- **Interpretations:**
  - `S=32` — the training batch size, the only S the paper names.
  - `S=1` — no averaging; per-sample decisions.

## Axis 4 — Threshold selection  ·  SWEPT  ·  paper UNDERSPECIFIED

- **Paper says (Sec. V-A):** the threshold is "determined based on the model's
  performance using the validation set." No rule is given.
- **Why ambiguous:** "performance" admits at least two standard readings, and
  they land in very different operating points on a balanced set.
- **Interpretations:**
  - `thr=fa5` — fix a ~5% false-alarm rate on normal validation batches.
  - `thr=maxf1` — choose the threshold maximising F1 on the validation set
    (the most literal reading of "performance", and the one most favourable to
    the paper's headline numbers).

## Axis 5 — Transformer placement  ·  SWEPT  ·  paper CONTRADICTS ITSELF

- **Paper says (Fig. 2):** graph conv → GRU → global max-pool → dense →
  transformer. Pooling before the transformer collapses the time axis.
- **Paper also says (Sec. III):** the transformer captures "longer-range
  *temporal* dependencies" — impossible if time was already pooled away.
- **Interpretations:**
  - `tf=nodes` — as drawn: pool first, transformer attends over nodes.
  - `tf=time` — as described: transformer attends over time, then pool.

## Axis 6 — Error distribution for the Mahalanobis fit  ·  SWEPT  ·  paper SILENT

- **Paper says (Sec. V-A):** μ is the mean error and φ the covariance "of E",
  where E is the error matrix. It does not say whether E is all samples or
  normal-only.
- **Interpretations:**
  - `fit=normal` — fit on benign errors (the standard reading; a covariance
    contaminated by attacks inflates the very distances used for detection).
  - `fit=all` — fit on all errors, literally "of E".

## Axis 7 — Sliding-window length W  ·  PENDING  ·  paper SILENT

- **Paper says:** input `X ∈ R^{Z×S×|V|}`; predicts all node features for a
  timestamp. No window length is stated; temporal context is implicitly the
  batch of S consecutive readings.
- **Interpretations:** `W ∈ {5, 10, 20}`. Currently fixed at 10. Unavoidable —
  some window must be chosen to run at all — so it is a sensitivity axis rather
  than a competing reading.

## Axis 8 — Attack operation duration  ·  PENDING  ·  paper SILENT

- **Paper says:** 1,400 hours "evenly split between normal and attack
  operations". The word *operations* implies contiguous episodes; no length given.
- **Interpretations:** block ∈ {15, 60, 120}. Currently fixed at 60. Interacts
  with Axis 3 (FA ≈ S / block).

## Axis 9 — Replay offset Δt  ·  PENDING  ·  paper SILENT

- **Paper says (eq. 2):** `X^m_t = X^b_{t−Δt}`, Δt "the difference between the
  current and the replayed time step". No value.
- **Interpretations:** random per episode (current) vs a fixed Δt.

## Axis 10 — Detection method for the shallow models  ·  PENDING  ·  paper SILENT

- **Paper says:** SVM/RF/LGBM are compared in Table I, but the Mahalanobis
  residual procedure is described only for models that forecast. Nothing says how
  SVM/RF/LGBM decide.
- **Interpretations:** supervised classification on the observed reading
  (current) vs classification on a residual feature.

---

## FIXED — not legitimate branches

- **Graph node sets.** The paper *prints* the exact 10/20/31-node graphs in
  Figure 1. Transcribed verbatim; no reduction is recomputed. Note eq. 1
  (betweenness) would select a *different* 10-node set (keeping J422 over PU8),
  so the figure — not the named criterion — governs. Any branch here would
  contradict an explicit figure.
- **Metric definitions.** F1, ACC, DR are given explicitly and used as printed.
- **Model hyperparameters.** Sec. V-B states them; used as given.
- **Attack equations.** Eqs. 2–4 are explicit; only their *units* (Axis 2),
  *duration* (Axis 8) and *offset* (Axis 9) are open.

---

## Interpretation matrix

Core swept axes: 1 × 2 × 3 × 4 × 5 × 6 = **2 × 2 × 2 × 2 × 2 × 2 = 64
configurations**, each covering 9 models × 3 graph sizes, repeated over seeds.

The claim under test: **no configuration in this space reproduces the paper's
reported pattern** — the monotone size improvement, the +31 to +40 F1 margin of
STGT over FFNN, and the transfer-learning gain. Reporting must show the full
distribution over the space, not a selected point.

**Pre-registration note.** This register is written *before* the matrix is run,
so no interpretation can be chosen after seeing which is favourable. Axes marked
PENDING are declared now and either swept later or reported as untested scope
limits — they must not be quietly dropped.
