# Table III seed-11 breadth map

**Frozen:** 2026-08-18

**Scope:** one registered ISET completion per named Table III row before repeated-seed depth

All rows below are exploratory interpretations, not paper-literal ADASYN
executions. The proposed-model rows use the original `B2+M` test population;
the supervised rows also omit the paper-positioned pre-split supervised
ADASYN. SVM rows are explicitly resource capped. Metrics are percentages in
`DR / FA / ACC / AUC` order.

| Paper row | Registered completion | Observed | Reported | Score audit and implication |
|---|---|---:|---:|---|
| Naive Bayes | GaussianNB, seeded 2:1 row split, cutoff 0.5 | 88.78 / 44.53 / 72.12 / 79.17 | 73 / 18 / 77.5 / 70 | Non-match at the fixed cutoff; closest ROC point is 5.00 points from the reported DR/FA pair, so an omitted operating-point rule remains material. |
| ARIMA | pooled ARIMA(1,1,0), residual MSE, threshold 0.58 | 21.48 / 57.20 / 32.14 / 24.72 | 86 / 12 / 87 / 87 | Paper-direction oracle is 50.00% ACC and reversed direction reaches 69.74%; this completion ranks in the wrong direction. |
| One-class SVM | sigmoid/scale, `nu=0.5`, 12k train and 30k test caps | 91.87 / 50.94 / 70.47 / 79.67 | 90 / 9 / 90.5 / 87 | The near-reported DR requires over 50% FA; closest ROC point remains 18.31 points away. |
| Supervised feed-forward | five 500-unit ReLU layers, Adamax, two-class Softmax, cutoff 0.5 | 96.41 / 23.72 / 86.35 / 97.05 | 90 / 11 / 89.5 / 88 | Strong positive control. A retrospective threshold reaches 91.83% DR / 9.17% FA, only 1.83 points from the reported pair; the source omits threshold selection. |
| Supervised LSTM | four 300-unit ReLU LSTMs, sigmoid/BCE completion | 100 / 100 / 50 / 50 | 90.5 / 10 / 90 / 89 | Every probability is 1.0; both score directions have 50% oracle ACC. This completion collapsed. |
| Multiclass SVM | seven-class sigmoid/scale repair, 30k train and test caps | 85.94 / 55.67 / 65.14 / 73.06 | 91 / 8 / 91.5 / 89 | Best balanced ACC is 71.14%; closest threshold remains 23.44 points from the target pair. |
| FC-SAE | printed widths and Softmax output, MSE, threshold 0.58 | 26.18 / 58.22 / 33.98 / 31.04 | 81 / 15 / 83 / 81 | Paper-direction oracle is 50.00% ACC; score correlation with zero reconstruction is 0.99946. |
| LSTM-SAE | mirrored Algorithm-2 states, repeat-latent decoder, MSE, threshold 0.61 | 14.78 / 40.96 / 36.91 / 33.09 | 85 / 13 / 86 / 82 | Wrong-direction ranking; nearest paper-direction DR/FA point is 47.11 points away. |
| FC-VAE | fixed-unit probability `exp(-0.5*MSE)`, threshold 0.43 | 11.51 / 32.62 / 39.45 / 30.13 | 88 / 11 / 88.5 / 85 | Wrong-direction ranking for this probability completion; paper-direction oracle is 50.00% ACC. |
| LSTM-VAE | latent width 300, fixed-unit probability `exp(-0.5*MSE)`, mirrored states, threshold 0.47 | 10.02 / 25.79 / 42.11 / 29.83 | 91 / 7 / 92 / 86 | Wrong-direction ranking; nearest paper-direction DR/FA point is 58.48 points away and reversed-direction ACC is only 66.93%. |
| LSTM-AEA | additive attention over prior queries, concatenated context, mirrored states, MSE, threshold 0.51 | 25.43 / 58.22 / 33.60 / 29.93 | 94 / 5 / 94.5 / 90 | Wrong-direction ranking; nearest paper-direction DR/FA point is 60.11 points away and reversed-direction ACC is only 66.52%. |

## Bounded reading

None of the eleven registered fixed operating points reproduces its complete
printed metric pattern. Supervised feed-forward is nevertheless a strong
positive control; it and Naive Bayes expose a materially omitted decision-
threshold procedure. The other nine score vectors remain materially far from
the reported DR/FA corner under threshold adjustment in their registered
direction. This establishes where the
registered completions diverge. It does **not** establish intent, cover printed
ADASYN, eliminate every source-supported ambiguity, or supply repeated-seed
uncertainty.

The next scientific step is to run the predeclared one-factor population,
split, scaling, threshold-selection, and Attack-3 interpretations. Repeated
seeds and clustered uncertainty follow only for the finite branches that
survive that divergence map.

## Result records

Machine-readable summaries are in [`results/`](results/). The two final rows
are [`iset_lstm_vae_seed11_20260818.json`](results/iset_lstm_vae_seed11_20260818.json)
and [`iset_lstm_aea_seed11_20260818.json`](results/iset_lstm_aea_seed11_20260818.json).
