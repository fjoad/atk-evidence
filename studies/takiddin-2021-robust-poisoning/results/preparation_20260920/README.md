# Preparation check: 20 September 2026

Job 397206 ran frozen commit 30ce6c4 and completed successfully in 2:16 on four
CPU cores, with no GPU. The preparation program took 55.66 seconds; the
allocation also included environment startup and the 13 passing fixture tests.
The maximum sampled RSS reported by Slurm was 284,032 KiB.

All six archives were scanned: 157,992,996 source readings. The 20 selected
customers supplied 10,718 candidate days, of which 10,658 passed the strict
48-slot rule. There were 40 days with extra slots and 60 excluded days in
total. The pilot retained the first 28 complete days of each customer, giving
560 original profiles. This does not establish the authors' 3,000-customer
selection.

| Preparation | Training rows | Test rows | Changed training rows at nominal 30% |
|---|---:|---:|---:|
| Generalized two-class | 4,464 | 2,232 | 675 (15.12%) |
| Generalized novelty | 373 | 6,704 | 108 (28.95%) |
| One customer's two-class | 224 | 112 | 67 (29.91%) |
| One customer's novelty | 18 | 336 | 5 (27.78%) |

Each path ran at both 0% and 30%. Percentages differ because the declared
generalized choice selects customers, the customer-specific choice selects
training examples, and only malicious labels can be flipped. These are
consequences of the recorded interpretations, not identified author behavior.

The generalized two-class split contains 1,306 links from synthetic test rows
to original training parents and 1,186 links in the opposite direction. These
counts are parent links, not distinct rows or customers. The novelty
replacement completion produces 108 shared attack identities across training
and testing at 30% (five in the single-customer example). Both effects remain
visible in the metadata; no silent deduplication or corrected split occurred.

The independent artifact checker verified 224 array hashes, shapes, finite
values, class counts, true/observed label separation, scaling, every synthetic
interpolation, recorded identity overlap, and fixed raw test populations across
poison levels. Full arrays and customer IDs remain under ignored data paths.

Records:

- [summary.json](summary.json): unmodified program summary copied from Panther.
- [artifact_audit.json](artifact_audit.json): checks of transferred saved arrays.
- [execution.json](execution.json): allocation, freeze, scope, and timing.
- [slurm-397206.out](slurm-397206.out): unmodified job output.

No classifier was fitted or scored. This completes the bounded preparation
check; it is not a small-data substitute for any published table.
