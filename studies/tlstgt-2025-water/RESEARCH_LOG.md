# Attack detection in water networks

Updated: 2026-09-23

Notes on *Graph Transfer Learning-Based Attack Detection in Cyber-Physical
Water Distribution Systems*, Ahasan, Joad, Atat, Thompson, Serpedin and
Takiddin, EUSIPCO (2025).

**Disclosure:** Faaiz Joad, a maintainer of these notes, is a co-author of this
paper. This investigation is not independent of the authors. We record our
own mistakes and the limits of each comparison.

This account follows the [evidence log](EVIDENCE.md), including the problems
we found, the checks they prompted and the corrections that changed our view.
No water-network experiment was rerun for this page.

**Earlier investigation, awaiting reassessment.** These results were obtained
under an earlier workflow. This page reorganizes the existing record and its
corrections; it does not certify the older conclusions anew. A fresh review
and any reruns are separate future work, after the current poisoning study.

## What the paper claims

The paper combines graph structure, temporal modeling and transfer learning
to detect attacks on water-system sensors. It reports improving performance
on larger graphs and an advantage over ordinary machine-learning baselines.
The proposed benefit is not merely high accuracy: it is that these components
help exploit the network's spatial and temporal structure.

## Starting hypothesis

We asked whether the reported numerical pattern followed from the stated
procedure, and whether simple rules could explain performance attributed to
the architecture. An early expectation that graph topology would add little
turned out to be wrong in the tested comparisons. That change is part of the
investigation, not an exception to hide.

There is also a data boundary. Our benign source is the canonical DeepH2O
C-Town series, not the authors' unpublished 1,400-hour run. Attack duration,
replay offset, window length and other omissions require declared assumptions.
These implementations therefore cannot be treated as exact recovery of the
authors' original experiment.

## 24 July — Our threshold bug changed the apparent failure

We initially described a roughly 46-F1-point shortfall. A later pipeline check
found different models producing identical decisions, which led us to a defect
in our own threshold selection. Samples labeled normal at the current time
could still contain attack-corrupted history or a trailing batch containing
an attack. Using them as normal calibration examples inflated the threshold.

The approximately 46-point-shortfall claim was retracted. Corrected exploratory
STGT results were much closer to the reported F1, but had different false-alarm
rates and did not reproduce the complete pattern. The threshold inflation was
about 8,000-fold in a constructed check; it was not evidence against the paper.

**Provenance:** the evidence log marks the early local experimental numbers
in C1–C4 and C6 as provisional under the later cluster-only execution rule.
This notebook does not promote them into eligible cluster results. The static
paper-arithmetic checks do not depend on that execution issue.

We also corrected an earlier attribution of the identical-model collapse to
attack scaling alone. The scale problem remained worth inspecting, but the
threshold defect caused the observed collapse. See [corrections C1–C4](EVIDENCE.md).

## 24 July and the later arithmetic checks — Do the printed metrics fit?

On a balanced binary test set, accuracy is the average of detection and
specificity. Because specificity cannot exceed 100%, ordinary accuracy cannot
exceed `(detection + 100%) / 2` under that balance assumption.

The saved check found 24 of 27 reported model/graph combinations above that
ceiling, and all 27 failed the stricter joint detection/F1/accuracy check under
the stated interpretation. For example, STGT at 31 nodes reports detection
74.7%, F1 76.3% and accuracy 84.8%; the first two imply about 76.8% accuracy
on a balanced population, not 84.8%.

We then asked whether another metric convention or class balance could explain
the table. A finite search checked 67,326 combinations per cell: **229 selected
test sizes**, 49 prevalences and six definitions. No one combination explained
the whole table; **20 of 27** cells had no match anywhere in that search.
This was not a search of every conceivable protocol.

The [saved search output](results/protocol_search.txt) and
[arithmetic record C5/C8/C10](EVIDENCE.md) retain the definitions and checks.
These inconsistencies identify a reporting/protocol problem under those
assumptions, not whether its cause was deliberate or accidental.

## Later checks — Establishing simple comparisons first

Cluster breadth checks separated the attack types. A z-score rule detected
the large manipulation attacks well. A stuck-sensor rule detected 93.3% of
denial-of-service examples in the recorded setup. Replay remained difficult
for the tested methods.

A composite of the two simple rules achieved F1 71.5, accuracy 76.9% and
detection 60.5% at 7.9% false alarms. It has no fitted parameters, but that does
not remove uncertainty from the sampled data. Some trained-model comparisons
used a common 3.7% false-alarm limit; the composite's 7.9% point must not be
presented as if it used that same limit. Full configurations are in
[C7 of the evidence record](EVIDENCE.md).

The question became what the complex model added beyond these simple checks,
not whether any nonzero detection alone demonstrated its mechanism.

## Later checks — Capacity, replay and instability

Increasing feed-forward capacity across a 272-fold parameter range did not
improve the tested comparison. At FA=3.7%, detection was 56.7%, 56.9%, 41.9%
and 56.2% across the four sizes. This weakens a capacity-only explanation
inside those settings; it is not a universal capacity bound.

A corrected nearest-neighbor check gave replay-versus-normal AUC 0.455.
Our first reference bank had included examples being evaluated, allowing
self-matches; that error was corrected. Low replay detection and the corrected
distance comparison establish difficulty for these checks, not equality of
every distribution a detector could use.

Across five recorded seeds, TGCN spanned 23.7 F1 points and STGT 11.8. That
variability is another reason not to substitute one attractive run for a
stable result. See [C9 and the later evidence](EVIDENCE.md).

## Later checks — The graph does help

This went against our starting expectation. Shuffling graph labels or
removing edges lowered performance in the recorded comparisons:

| Model and graph | Real topology | Shuffled topology | No edges |
|---|---:|---:|---:|
| STGT, 10 nodes | 57.9 F1 | 51.9 F1 | 44.9 F1 |
| TL-STGT, 31 nodes | 63.6 F1 | 45.2 F1 | 44.7 F1 |

The topology is not decorative in these tests. Transfer learning also showed
gains of 13.3–21.9 F1 at 31 nodes in the recorded configurations. However,
pretraining gave the transferred model more total training; the comparison
does not isolate transfer from training budget. Those supportive findings and
their limits belong together. See [C11](EVIDENCE.md).

## Later checks — Training labels change the comparison

The paper's stated equal-class split places attack examples in training.
Using benign-only training instead produces quite different F1 values:

| Model, 31 nodes, S=1 | Balanced training | Benign-only training |
|---|---:|---:|
| Feed-forward | 44.1 | 70.3 |
| LSTM | 46.0 | 71.2 |
| TGCN | 44.6 | 70.8 |
| STGT | 46.1 | 67.8 |
| TL-STGT | 68.0 | 70.9 |

An earlier draft quoted benign-training figures without making that choice
clear. We corrected it. Under benign-only training, the displayed TL-STGT
advantage over STGT is 3.1 F1 rather than 21.9. Training policy and extra
pretraining budget both matter; these observations do not isolate one cause
for every gain. The [C12 record](EVIDENCE.md) preserves the configurations.

## 31 August — Narrowing our own conclusions

The [wording correction](EVIDENCE.md#public-wording-correction--2026-08-31)
changes how the older results should be read:

- The protocol search is finite, not every possible test size or convention.
- A one-sided-error sign-test calculation assumes independent, equally likely
  signs. Its p-value is not the probability that results were fabricated.
- The 57.1% non-replay share is **not an unconditional detection ceiling**.
  It is what detection would be if all non-replay attacks and no replay attacks
  were caught. A genuine indistinguishability bound would need distributional
  assumptions and must account for false alarms.
- Flat capacity results and failure on frozen history do not prove that all
  capacities or every possible forecaster must fail.

The earlier evidence entries retain their original wording. These corrections
govern the current interpretation; they do not change the saved measurements.

## Records and remaining questions

- [Full evidence log, including retractions and provenance](EVIDENCE.md).
- [Data provenance and limitations](DATA.md), [paper specification](PAPER_SPEC.md), and [ambiguity register](AMBIGUITY_REGISTER.md).
- [Saved results](results/).
- [Earlier detailed website account](../../site/papers/tlstgt-2025-water/earlier-notes.html).
- [Earlier report (PDF)](../../site/reports/tlstgt-2025-water.pdf), read with the later corrections above.

The interpretation sweep and a matched-budget transfer comparison remain
incomplete. The unpublished source run is not recovered. This notebook does
not authorize another experiment or transfer its findings to either
electricity-theft paper.

## Current conclusion

The printed table has arithmetic inconsistencies under the stated balanced
evaluation and remains unexplained by the finite protocol search. That is
distinct from a blanket claim that the architecture cannot work.

The experimental record contains both substantial sensitivity to setup and
evidence that graph structure and transfer can help in particular comparisons.
Our own threshold error and earlier overstatements materially changed the
investigation and are retained prominently. Neither the failed searches nor
the experiments establish author intent or the origin of the published values.
