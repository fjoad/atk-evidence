# Remaining-model feasibility wave: preflight finding

**Date:** 2026-09-01

**Evidence type:** operational `X`; no numerical (`N`), mechanism (`M`), or
attainability (`A`) evidence.

## Plain-language result

The first four pilot jobs stopped before building a model. The gate required a
checksum for every input file to appear in the prepared cache's own
`metadata.json`. That metadata contains checksums for the prepared arrays but
omits `table_iv_order.npy`. The file itself is present, has the expected shape,
and its observed SHA-256 is exactly the value already recorded by the eligible
FC-SAE anchor.

This is like finding that a sealed item is present and its seal matches the
earlier audited receipt, but the current packing list has no line on which to
compare that seal. The correct response was to stop, not to silently skip the
comparison.

## Exact outcome

All jobs ran frozen commit
`0ca6cc4737d37f5d87a7dc156a1b06ea7ca88730` through the two-hour
pilot-only wrapper on `gpu-short`, one GPU, 16 CPUs, and 96 GiB RAM.

| Job | Model | Node | Elapsed | Exit | Outcome |
|---:|---|---|---:|---:|---|
| 385544 | LSTM-SAE | `crirdcmgpu002` | 1:33 | `1:0` | missing manifest entry |
| 385545 | FC-VAE | `crirdcmgpu002` | 1:33 | `1:0` | missing manifest entry |
| 385546 | LSTM-VAE | `crimv3srv041` | 0:46 | `1:0` | missing manifest entry |
| 385547 | LSTM-AEA | `crimv3srv041` | 0:46 | `1:0` | missing manifest entry |

The common exception was:

```text
ValueError: prepared metadata omits the checksum for table_iv_order.npy
```

Total allocation exposure was 278 GPU-seconds, about 0.077 GPU-hours. No model
was built, no parameter was updated, and no reconstruction or anomaly score was
calculated. Therefore these jobs say nothing about model validity, runtime, or
the paper's reported numbers.

## Checksum reconciliation

- prepared-cache metadata SHA-256:
  `5f3e9d8ea038f8dddede879f73f420a679124cd24a5d2311a2e7e4838a9e869e`;
- committed eligible-anchor result SHA-256:
  `ae07b42ef6c84242ca9b39db8b8828694d6d4df6859abdee090fc0a613a69154`;
- `table_iv_order.npy` SHA-256 recorded by that anchor:
  `f5acf853f2efdcc5e237cdf137cd17590fc0712f8151798de65f5b034f351643`;
- SHA-256 observed directly after the failed wave:
  `f5acf853f2efdcc5e237cdf137cd17590fc0712f8151798de65f5b034f351643`.

The expected and observed file hashes agree exactly. The defect is in the
pilot's manifest-source assumption, not in the saved table order.

## Narrow repair for discussion

Keep the checksum gate. For a prepared file whose cache metadata lacks a hash,
read the expected value from the committed, independently audited clean-reader
anchor result only after verifying that the anchor records the exact current
metadata SHA-256. Compare the actual file bytes to that expected value and stop
on any mismatch. Record which manifest supplied every expected checksum.

This repair does not change data, selection, model, objective, score, budget,
or promotion gates. It requires a new code commit and new immutable pilot
attempts. No retry is authorized or implied by this finding; discuss first.

## Discussion outcome

The user approved this exact narrow repair on 2026-09-01. The approval does not
reclassify or erase jobs 385544--385547. A new tested commit and new immutable
attempt identities are required before resubmission. The repaired resolver
passes 245 deterministic tests (140 study, 105 root), including checks against
the exact committed anchor and archived prepared metadata; strict data
verification selects the complete ScienceDB semantic-equivalence branch.
