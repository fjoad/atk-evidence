# Paper 1 baseline execution contract

This directory contains five scientific files and one short Panther wrapper.
The current baseline is the least ambiguous complete lane: ISET/CER, FC-SAE,
seed 11, full benign training data, and the printed Table-I architecture and
threshold.

## What is literal and what is assumed

| Node | Paper instruction | Baseline execution |
|---|---|---|
| Data | ISET/CER residential half-hour readings | Exact six consumption archives plus the explicitly labeled semantic-equivalence allocation CSV |
| Sample | One 48-value daily profile | Exact slots 1–48 once each; incomplete or DST-extra days excluded |
| Attacks | Six equations for every customer | All six on every retained meter-day |
| Attack 3 | `t_f = t_i - t_l` | **Literal failure:** positive duration places the end before the start. **Assumption:** use addition, clip at hour 24, and map each hour to two half-hour slots. |
| Scaling | Normalize benign and malicious data before splitting | One 48-feature standardizer fitted jointly to `B_raw + M_raw` |
| Split | Split benign customers 2:1 | Seeded customer-disjoint split; all days from one meter remain together |
| Anomaly train | `X_TR = B1` | FC-SAE trains only on standardized benign `B1` |
| Anomaly test | `B2 + M`, with `M` from all customers | Preserved exactly, including malicious derivatives of training customers |
| Test ADASYN | Apply ADASYN to the complete test set | **Printed branch remains open:** the selected exact default requires about 10.7 trillion first-pass query/reference pairs and did not complete in the bounded attempt. **Runnable baseline assumption:** `I-ADASYN-NONE` evaluates the exact original `B2+M` rows without resampling. It is not labeled paper-literal. |
| Architecture | `48-400-300-200-100-100-200-300-400-48` | Runtime assertion checks all widths, eight sigmoid hidden layers, Softmax output, dropout 0.4, and 450,448 parameters |
| Training | Adam/MSE; other controls absent | Adam `lr=0.001`, batch 512, at most 100 epochs, shuffled batches, training-loss patience 5 and `min_delta=1e-4`, restore best weights |
| Decision | MSE greater than 0.58 is malicious | Applied directly |
| Metrics | DR, FA, SP, PR, balanced ACC, F1, AUC | Applied directly from one preserved score vector |
| Table V | One attack at a time, already balanced | The same trained model, threshold, all benign rows, and each equal-sized attack matrix; FA must therefore be common across attacks |

Batch 512 is the previously frozen primary execution completion. Batch 32 is
the Keras-default sensitivity and cannot be mixed into its seed summary.

## One command on Panther

From `/export/home/fjoad/atk-evidence`:

```bash
sbatch studies/atk-2022-deep-autoencoder/reproduction/run_baseline.sbatch
```

That job verifies the named source files, prepares the exact runnable baseline,
trains FC-SAE, scores Tables III and V, saves raw scores/predictions/weights and
timings, and regenerates the reported-versus-reproduced CSV/JSON summaries.

The scientific implementation remains the five Python files. The Slurm file is
only a four-command resource wrapper and contains no scientific logic.

## Eligibility boundary

This baseline can answer whether the complete paper method *apart from the
explicitly omitted test-set ADASYN step* produces the reported FC-SAE behavior.
It cannot be called the completed `P0` result. A printed-ADASYN result remains a
separate required attempt; no result from this baseline may silently fill that
cell.
