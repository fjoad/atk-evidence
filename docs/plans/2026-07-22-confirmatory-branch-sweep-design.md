# Paper 1 Confirmatory Branch Sweep — Design Note

**Goal:** When this design is later frozen into the confirmatory contract, the
project will have tested, within a finite pre-registered space, essentially
every SGCC/evaluation interpretation the authors could plausibly have used —
and either shown that none produces the reported Table II pattern, or found
and reported the one that does.

**Status:** Historical draft, partially superseded on 2026-07-23 by
`studies/atk-2022-deep-autoencoder/BRANCH_COVERAGE_CONTRACT.md`. No runs are
authorized by this document.

**2026-07-23 correction:** The exact target paper is the sole authority for the
paper-consistent family. Precursors, contemporaneous literature, and toolchain
defaults may motivate separately labeled external-author-implementation
possibilities, but cannot add methods to or rescue the paper-consistent
reproduction claim.

**Provenance:** Drafted 2026-07-22 by Claude (Fable) from a direct discussion
with the user, alongside the exploratory branch work by the parallel session
(Sol). User directive: "we should do all the things they could have possibly
done, and still show that none of it works."

## Context

The exploratory SGCC Table II results (classical 12/12 cells, FC-SAE 3/3,
LSTM-VAE 1/3, supervised feed-forward 3/3) show every anomaly-detector row far
below its reported pattern with AUC near 50%, while the supervised deep
benchmark lands near its reported row. The standing objection to any single
non-reproduction is "the authors may have done something else," and for SGCC
the paper genuinely does not define the model input (A01). This design answers
that objection by bounded exhaustion rather than by one registered guess. It
corresponds to steps 7–8 of STATUS ("freeze a confirmatory contract, execute
preregistered runs").

## Design

### 1. Externally anchored branch enumeration

A paper-consistent interpretation is admissible only when anchored to a
specific phrase, contradiction, equation, algorithm, figure, or omission in
the exact paper. Precursors, contemporaneous SGCC literature, and toolchain
defaults belong only to the separately labeled `X` track. Anything outside the
paper-derived branch lattice and frozen corrected controls is recorded as out
of scope, not silently searched.

### 2. Candidate dimensions (to be finalized at freeze)

- **SGCC representation (A01 family):** full 1,034-day sequence (current
  primary); 48-day windows (stride 48; stride 1 as budget permits); first-48
  and last-48 days; downsampling 1,034 → 48 (binned means); weekly reshape
  variants used by the SGCC literature.
- **Missing data (A02):** within-customer interpolation + train-benign
  medians; zero fill; literature-standard row-filtering variants (e.g., drop
  customers above a missingness threshold — also addresses the paper's
  "~40,000 customers" vs. the released 42,372).
- **Normalization scope (A07 family):** joint feature-wise before split
  (current primary); per-class; per-customer/per-row.
- **Score aggregation (new, windowed branches only):** per-row scoring vs.
  per-customer aggregation (max / mean) before thresholding; label inheritance
  for windows of malicious customers.
- **Training budget (A18 family):** epochs {10, 30, 100}; batch {32, 512};
  stated optimizer defaults. Screened, not exhaustively crossed.
- **VAE score definitions (A11):** reconstruction MSE; MSE+KL surrogate.
- **Threshold rules:** printed thresholds (paper-literal primary); Youden and
  validation-FA variants (labeled diagnostics); test-ROC-derived thresholds
  (leakage branch — mechanism demonstration only, never presented as
  paper-literal).

### 3. The AUC screening funnel

**2026-08-18 correction:** the former draft incorrectly used balanced accuracy
as a lower bound on ROC-AUC. For a monotone ROC curve containing a point
`(FPR=f, TPR=t)`, the general lower bound is `AUC >= t * (1-f)`: the curve may
remain at zero until `f`, rise vertically to `t`, and remain at `t` until an
endpoint jump at FPR 1. The reported Table-II points therefore imply lower
bounds from 0.7138 (FC-SAE) to 0.9216 (LSTM-AEA), and their printed AUCs do not
violate those bounds. This relationship is a useful necessary condition, not a
source-level contradiction.

- **Stage 1 (screen):** for every admissible branch: short training, reduced
  sample, three seeds, measure anomaly-score AUC. AUC is invariant to monotone
  score rescaling. A branch whose full-run AUC is below the exact lower bound
  required by the reported DR/FA point cannot contain that operating point in
  the registered score direction. A short/reduced run is only a promotion
  screen, however, and cannot by itself prove that the full branch is dead.
- **Stage 2 (full treatment):** only branches whose screening AUC approaches
  the required bar (promotion rule frozen in advance, e.g., mean AUC ≥ 0.80)
  graduate to the complete paper-literal pipeline: full training, frozen
  seeds, printed thresholds, all seven metrics, immutable attempts.

### 4. Controls

- **Positive control:** the supervised feed-forward (already near its reported
  row, AUC 92–95%) demonstrates the pipeline extracts signal when signal is
  extractable.
- **Negative control:** label-shuffled runs must produce AUC ≈ 50, proving no
  leakage inflates screening results.

### 5. Mechanism demonstration (constructive half)

Beyond "nothing works," identify the *minimal deviation that reproduces their
numbers*:

- Test-ROC-derived thresholds (what the precursors' written protocols
  describe) applied to otherwise paper-literal scores.
- Per-attack-column threshold reselection as the candidate fingerprint for
  Table V's structurally impossible varying FA.

If a small set of improper choices closely reconstructs the reported tables,
the report can state the procedure that generates these numbers and contrast
it with the procedure described.

### 6. Pre-registered expectations (recorded now to prevent hindsight bias)

- SGCC anomaly rows: expected not to reproduce in any admissible branch.
- ISET Tables III/V (synthetic attacks): partial reproduction is genuinely
  plausible and would be reported with equal prominence.
- A branch that reproduces the pattern weakens or falsifies the Paper 1
  hypothesis and is reported plainly per VISION.

### 7. Honest residue (explicit exclusions)

Even a complete sweep cannot cover: an undisclosed dataset subset beyond the
literature-standard filters; bugs in the authors' own code; procedures
appearing in none of the three anchors. This residue is itself the finding —
if reproducing the table requires information absent from the paper, its
precursors, and its field's practice, the paper has failed reproducibility by
definition, and the sweep quantifies how far beyond the text a reader must go.

## Freeze checklist (before any execution)

- [ ] Literature-collection pass complete; anchor-2 citation list attached.
- [ ] Final branch list with one-line plausibility rationale per branch.
- [ ] Screening bars derived per model row from the reported (DR, FA) cells.
- [ ] Promotion rule, per-branch budget, seeds, and stopping rule fixed.
- [ ] Statistical treatment fixed (per-branch CIs across seeds; multiplicity
      handling; no post-hoc branch additions or removals).
- [ ] New ambiguity-register entries for every dimension not already covered.
- [ ] Compute plan sized (screening runs are small; the cluster only for
      survivors and full-treatment cells).

## CHECKPOINT: user review

The freeze itself is a major-tier action: the user approves the final branch
list, bars, budgets, and exclusions before the first screening run.

## Verification (at execution time, not now)

- [ ] Every screening result recorded, including dead branches.
- [ ] Controls behave as required (positive ≥ bar's neighborhood; shuffled ≈ 0.5).
- [ ] Evidence ledger updated with sweep outcomes and any surviving branch.
