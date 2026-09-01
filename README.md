# ATK Evidence

We read research papers, implement the methods they describe, and compare our
results with the published ones. When they differ, we try to understand why.

**[Read the studies](https://fjoad.github.io/atk-evidence/)** ·
[Current electricity-theft reproduction](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/reproduction/)

## What we found so far

We rebuilt FC-SAE, the simplest autoencoder in Takiddin et al. (2022), and ran
it on the named Irish electricity data. The first run did not reproduce its
Table III result.

| Measure | Paper | Our run |
|---|---:|---:|
| Attack detection | 81.00% | 25.48% |
| False alarms — lower is better | 15.00% | 45.13% |
| Specificity | 85.00% | 54.87% |
| Precision | 81.00% | 36.73% |
| Balanced accuracy | 83.00% | 40.18% |
| F1 | 81.00% | 30.09% |
| AUC | 81.00% | 39.40% |

This is **one model, one documented interpretation, and one training seed**.
The paper omits some necessary settings, so we recorded our choices before
running it. The saved data and predictions passed our checks. Changing the
cutoff alone cannot recover the target from these scores.

We then asked a stronger question: could *any* reconstruction allowed by this
output layer reach the target? On the same prepared data, even an imaginary
detector that knows every label and chooses each reconstruction in its favor
stays below **50.93% balanced accuracy**, versus 83% reported. At a false-alarm
rate of at most 15%, it can detect at most **9.25%** of attacks, versus 81%.
Changing weights, seeds, or training duration cannot overcome this limit while
the prepared inputs, Softmax output, and mean-squared-error score stay fixed.
Different preprocessing, output layers, or scores are outside this bound.

We checked which restrictions were actually in the paper. Softmax, MSE, and
the 0.58 cutoff are explicit; the normalization statistics are less precise.
Two additional normalization readings still cap detection at **29.81% and
33.96%**, versus 81%. A Sigmoid output range changes the limits substantially:
on the complete evaluation it permits the reported detection/false-alarm pair
at another cutoff, so that range calculation alone cannot exclude the rescue.
See
[which assumptions matter](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/reproduction/#source-assumptions)
and [what Sigmoid changes](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/reproduction/#sigmoid-range).
The full source-assumption check took 57 seconds with no training;
[all outcomes and limitations are saved](studies/atk-2022-deep-autoencoder/SOURCE_ASSUMPTION_FINDING.md).

We then trained one small paired Softmax/Sigmoid control from identical starting
weights. Both completed ten epochs. Even after enumerating every cutoff on the
held-out sampled scores, Sigmoid detected only **9.75%** of attacks at at most
15% false alarms, or **25.39%** when lower error was treated as suspicious,
versus 81% reported. The relaxed rounding target also failed. This proves that
no cutoff rescues that fitted model on that sample. It does **not** prove that
all Sigmoid training procedures fail: calibration loss was still improving at
the final epoch. The [paired finding](studies/atk-2022-deep-autoencoder/SIGMOID_FIT_FINDING.md)
preserves both the failure and that open possibility.

We also tested the paper's LSTM-SAE inside Table IV's reported **183-minute**
full-ISET training time. The paper does not name its hardware or epoch count.
Our predeclared implementation used one V100-16GB, batch 32, one seed, the
full prepared data, and exactly 183 minutes of fitting. It completed 1.268
epoch-equivalents. All 8,884,989 saved scores were finite and passed the
independent count, hash, metric, and AUC audit.

| LSTM-SAE measure | Paper | Our 183-minute run |
|---|---:|---:|
| Attack detection | 85.00% | 16.62% |
| False alarms — lower is better | 13.00% | 31.91% |
| Specificity | 87.00% | 68.09% |
| Precision | 85.00% | 34.87% |
| Balanced accuracy | 86.00% | 42.35% |
| F1 | 85.00% | 22.51% |
| AUC | 82.00% | 40.30% |

We evaluated all 7,036,998 distinct cutoff boundaries. At no more than 13%
false alarms, the best detection was **7.00%** in the paper's score direction
and **23.02%** after favorably reversing it, versus 85%. No cutoff rescues
these fitted scores. One V100 epoch took 143.88 minutes; ten epochs in our
declared completion project to **23.98 hours**, or 7.86 paper-time budgets.
The paper itself does not state ten epochs.

The bounded conclusion is that the reported LSTM-SAE result did not arise from
this declared implementation within the paper's reported time. Its loss was
still decreasing, so this is not a proof of unlimited-time impossibility or a
claim about the unpublished author implementation or intent. The
[paper-time finding](studies/atk-2022-deep-autoencoder/PAPER_TIME_BUDGET_FINDING.md)
and [readable explanation](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/reproduction/#lstm-paper-time)
preserve the full result and its limits.

That does **not** mean the model does nothing useful. On original customer
rows, its balanced-accuracy gain over zero reconstruction is 0.89 percentage
points (conditional 95% interval: 0.80–0.98). It also ranks attacks better than
the tested simple controls within similar-input-magnitude groups. These score
comparisons do not identify the paper's claimed architectural mechanism.

The [readable report](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/reproduction/)
connects the paper's instructions to the actual code, a model diagram, the
complete results, the bound, and possible explanations.
The [initial finding](studies/atk-2022-deep-autoencoder/CLEAN_READER_FINDING.md)
and [follow-up findings and records](studies/atk-2022-deep-autoencoder/POST_ANCHOR_FINDING.md)
preserve the details. The output-domain follow-up required 113 seconds for the
full analysis and 18 seconds for a small no-training control. The later paired
fit required 25 seconds of analysis. Other source interpretations, the
remaining proposed models and tables, and the cause of the published
discrepancy remain open; no conclusion about author intent follows.

## How we work

1. Read the complete paper and write down what it asks the model to do.
2. Try small sanity checks and simple alternatives before expensive training.
3. Implement the stated method, making missing details and interpretations visible.
4. Run the experiment and compare the complete result, keeping failures too.
5. Test explanations for any difference, starting with the cheapest useful check.

We keep three questions separate:

- **Does the result reproduce?**
- **Does the extra architecture help for the reason the paper claims?**
- **Can the described method credibly reach the reported performance?**

An unsuccessful reproduction does not answer the other two by itself. To show
that a component adds little useful work, we need a fair comparison and a
justified definition of a meaningful gain. To claim impossibility, we need a
bound with explicit assumptions—not just a large gap or an extrapolated runtime.

## The two studies

| Paper | Read |
|---|---|
| Takiddin et al., *Deep Autoencoder-Based Anomaly Detection of Electricity Theft Cyberattacks in Smart Grids* (2022) | [Current reproduction](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/reproduction/) · [Earlier method notes](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/) |
| Ahasan et al., *Graph Transfer Learning-Based Attack Detection in Cyber-Physical Water Distribution Systems* (2025) | [Earlier study and corrections](https://fjoad.github.io/atk-evidence/papers/tlstgt-2025-water/) |

**Disclosure:** Faaiz Joad, a maintainer of this project, is a co-author of the
water-network paper. That study is not independent of its authors. Its evidence,
limitations, and corrections are kept visible. We do not infer author intent.

## Inspect or reproduce the work

The scientific code for the current experiment is under
[studies/atk-2022-deep-autoencoder/reproduction/](studies/atk-2022-deep-autoencoder/reproduction/).
It contains the download, data preparation, model, training, and analysis files.
The report links the exact revisions used for the runs.

```bash
git clone https://github.com/fjoad/atk-evidence.git
cd atk-evidence
bash scripts/bootstrap.sh
bash scripts/test.sh
```

See [Getting started](docs/GETTING_STARTED.md) for setup and data access.
Original data and paper PDFs stay outside version control. Public records
include code, file checksums, configurations, summary results, and corrections.

For ongoing work, start with [current status](docs/STATUS.md) and the
[documentation guide](docs/README.md). The [runbook](RUNBOOK.md) describes the
research procedure; it is not required reading for understanding the results.
