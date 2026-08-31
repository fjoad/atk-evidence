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

The trained scores closely follow a simple input-magnitude score, with only a
1.18-percentage-point improvement in balanced accuracy over zero reconstruction.
That suggests a useful next question: what did training add? It does not prove
that the model learned nothing, that another configuration cannot work, or that
the published results were fabricated.

The [readable report](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/reproduction/)
connects the paper's instructions to the actual code, a model diagram, the
complete results, and possible explanations.
The [technical finding and audit records](studies/atk-2022-deep-autoencoder/CLEAN_READER_FINDING.md)
preserve the details. The run completed on 30 August 2026 and was checked on
31 August. No new training follows automatically from it.

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
The report links the exact revision used for the run.

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
