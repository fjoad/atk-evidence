# Table II seed-11 breadth map

**Frozen:** 2026-08-18  
**Scope:** exact SGCC source, one complete `last_48` row for every named model,
plus all five proposed models on the two surviving 48-wide representation
contrasts. All preparation, training, scoring, and score-vector analysis ran on
Panther GPUs.

## Literal outcome

The source contains 1,034 **daily** values per customer, while every printed
architecture requires 48 **half-hourly** values. The paper supplies no
1,034-to-48 rule and no missing-value rule. Literal Table II is therefore
non-executable. No experiment below is relabeled as literal.

The smallest one-sample-per-customer repairs are:

- `last_48`: most recent 48 days;
- `first_48`: earliest 48 days; and
- `binned_mean_48`: 48 contiguous means spanning all 1,034 days.

All branches preserve the printed ordering: joint benign/malicious scaling,
benign `B1` anomaly training, ADASYN on `B2+M`, supervised ADASYN before a 2:1
split, and the transferred printed thresholds. The exact data retain 42,367
customers after five fully missing rows are removed: 38,755 benign and 3,612
malicious.

## Complete last-48 model breadth

Metrics are percentages. `Exact gap` is the smallest possible largest
absolute discrepancy across DR, FA, SP, PR, ACC, F1, and AUC after enumerating
**every deterministic threshold** over the saved score vector. Thus an exact
gap above two proves that threshold choice alone cannot make that executed run
a close seven-metric match.

| Model | Reproduced DR / FA / ACC / F1 / AUC | Reported DR / FA / ACC / F1 / AUC | Exact gap | Fit |
|---|---:|---:|---:|---:|
| FC-SAE | 3.80 / 0.78 / 51.51 / 7.26 / 48.89 | 83 / 14 / 84.5 / 83 / 83 | 36.53 | 7.8 s |
| LSTM-SAE | 3.80 / 0.78 / 51.51 / 7.26 / 51.23 | 86 / 12 / 87 / 86.5 / 85 | 38.78 | 2.38 min |
| FC-VAE | 2.46 / 0.38 / 51.04 / 4.78 / 48.86 | 90 / 9 / 90.5 / 90.5 / 88 | 44.20 | 8.8 s |
| LSTM-VAE | 2.60 / 0.40 / 51.10 / 5.04 / 49.90 | 93 / 6 / 93.5 / 93 / 90 | 45.54 | 10.41 min |
| LSTM-AEA | 4.05 / 0.84 / 51.61 / 7.71 / 50.11 | 96 / 4 / 96 / 95.5 / 93 | 47.47 | 40.67 min |
| Naive Bayes | 7.86 / 1.56 / 53.15 / 14.37 / 55.60 | 75 / 16 / 79.5 / 77 / 73 | 27.04 | 0.06 s |
| ARIMA | 1.31 / 0.19 / 50.56 / 2.58 / 54.83 | 88 / 10 / 89 / 87 / 88 | 33.84 | 0.07 s |
| Single-class SVM | 62.28 / 50.32 / 55.98 / 57.96 / 60.80 | 91 / 8.5 / 91 / 90 / 89 | 35.11 | 19.1 s |
| Feed forward | 83.14 / 6.50 / 88.32 / 87.71 / 95.31 | 91 / 9.5 / 91 / 90.5 / 89 | 6.31 | 1.35 min |
| LSTM classifier | 49.03 / 23.14 / 62.95 / 57.01 / 69.32 | 91.5 / 9 / 91 / 91 / 90 | 28.26 | 17.82 min |
| Multi-class SVM | 20.79 / 5.13 / 57.83 / 33.03 / 69.51 | 92 / 7.5 / 92 / 91.5 / 90 | 28.84 | 1.99 min |

The feed-forward score is a positive control: a different threshold reaches
within 1.94 points of its reported DR/FA pair. Its seven-metric gap remains
6.31 points because its AUC is higher than printed. The proposed models are not
merely using poor fixed thresholds; their closest complete vectors remain
36.53–47.47 points away.

## SGCC representation contrast

The table below reports threshold-free AUC for the five proposed models. The
feed-forward row is the control.

| Model | First 48 | Last 48 | Binned mean 48 | Paper |
|---|---:|---:|---:|---:|
| FC-SAE | 46.59 | 48.89 | 53.68 | 83 |
| LSTM-SAE | 46.32 | 51.23 | 54.05 | 85 |
| FC-VAE | 46.59 | 48.86 | 53.59 | 88 |
| LSTM-VAE | 46.31 | 49.90 | 54.15 | 90 |
| LSTM-AEA | 46.37 | 50.11 | 53.87 | 93 |
| Feed-forward control | 96.91 | 95.31 | 96.50 | 89 |

The control reaches the reported DR/FA neighborhood under all three readings.
The proposed family remains near chance under all three. Within a representation,
all ten pairwise proposed-model Spearman score correlations are at least 0.957;
most exceed 0.98. FC-SAE and LSTM-AEA raw MSE scores have Pearson correlation
above 0.99999998, while FC-VAE and LSTM-VAE exceed 0.99988 after their common
probability-scale orientation. The architectures are therefore producing
nearly the same customer ranking, not the reported hierarchy.

## Mathematical and statistical boundary

Independent of these runs, the printed tables contain exact arithmetic
problems:

- Table-II Naive Bayes reports DR=PR=75 but F1=77; DR and PR force F1=75.
- The paper calls the ADASYN test output balanced, yet five Table-II rows and
  eight Table-III rows cannot satisfy `PR = DR / (DR + FA)` even with ±0.5
  point rounding.
- Even without assuming perfect balance, the rows in each table imply mutually
  incompatible class prevalences.

These are model-independent contradictions. The experiments establish a
different, bounded result: none of the five proposed models reproduces its row
for seed 11 under early, late, or whole-history 48-wide readings, and no
threshold can rescue any saved score vector. They do **not** prove a statement
over every unreported implementation or every neural initialization. Repeated
seeds and customer-level uncertainty are still required before a statistical
claim over the frozen executable branches.

## Operational failures retained

Initial recurrent scoring attempts exhausted a 16-GB V100 because an inference
batch was too large; the retries changed only scoring batch partitioning.
The first multi-class SVM score adapter assumed a two-dimensional margin even
though binary SVM returns one dimension. Both failures and their corrected
retries remain preserved. They are operational evidence, not scientific model
failures.

Machine-readable results are in
[`results/sgcc_table_2_breadth_seed11_20260818.json`](results/sgcc_table_2_breadth_seed11_20260818.json).

