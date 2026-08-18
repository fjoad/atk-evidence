# Table V common-model breadth

**Frozen:** 2026-08-18  
**Scope:** seed-11 ISET proposed-model scores, one already-trained Table-III
model, its printed threshold, one common held-out benign population, and each
of the six attack populations in turn.

This is the smallest reading of the paper's statement that the Table-V
experiments use the same model settings and thresholds. It requires no
retraining: the six columns are slices of the preserved Table-III score vector.

## Reproduced DR

| Model | A1 | A2 | A3 | A4 | A5 | A6 | Paper range |
|---|---:|---:|---:|---:|---:|---:|---:|
| FC-SAE | 14.86 | 17.24 | 38.70 | 20.32 | 1.98 | 63.99 | 80–83 |
| LSTM-SAE | 6.17 | 6.98 | 25.67 | 3.34 | 0.27 | 46.26 | 82–90 |
| FC-VAE | 4.03 | 3.11 | 15.59 | 7.59 | 0.42 | 38.35 | 85–93 |
| LSTM-VAE | 2.49 | 2.36 | 14.52 | 1.57 | 0.12 | 39.04 | 88–95 |
| LSTM-AEA | 15.04 | 22.14 | 44.01 | 3.78 | 0.93 | 66.66 | 93–97 |

## Reproduced FA and the structural result

| Model | Reproduced FA for A1–A6 | Paper FA range |
|---|---:|---:|
| FC-SAE | 58.2236 for every attack | 10–19 |
| LSTM-SAE | 40.9635 for every attack | 9–15 |
| FC-VAE | 32.6198 for every attack | 8–12 |
| LSTM-VAE | 25.7930 for every attack | 4.5–8.5 |
| LSTM-AEA | 58.2182 for every attack | 2.5–6.5 |

This invariance is mathematical, not an empirical coincidence. False alarms
are benign examples predicted malicious. If the model, threshold, and benign
test identities are fixed, changing only the malicious attack population
cannot change FP or TN, hence cannot change `FA = FP / (FP + TN)`. The paper's
attack-varying FA columns therefore require an unstated change in model,
threshold, benign identities, or some combination.

The common-model results do not reproduce the reported DR or FA values. This
does not close the paper's ambiguous phrase “multiple experiments.” The
predeclared retrain-per-attack, resplit-benign-per-attack, and combined branches
remain distinct experiments; if run, they must be labeled as assumptions and
cannot replace this structural common-model outcome.

Machine-readable values are in
[`results/iset_table_5_common_model_seed11_20260818.json`](results/iset_table_5_common_model_seed11_20260818.json).

