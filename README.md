# ATK Evidence

ATK Evidence is a rigorous, paper-by-paper effort to produce independently
rerunnable evidence for or against published numerical claims. It begins with a
selected corpus by Abdulrahman Takiddin and coauthors, but the repository is
structured for papers in other domains and, eventually, other corpora.

Study 1 is:

> A. Takiddin, M. Ismail, U. Zafar, and E. Serpedin, "Deep Autoencoder-Based
> Anomaly Detection of Electricity Theft Cyberattacks in Smart Grids," IEEE
> Systems Journal, 16(3), 4106-4117, 2022.

The primary experiment is always the **paper-literal reproduction**: implement
what the paper explicitly says, use the named data, and add nothing silently.
When necessary details are missing, reasonable paper-consistent interpretations
are enumerated and documented before results are examined.

The working hypothesis is that the reported numerical results in the selected
papers will not be
reproduced reliably within a predeclared, reasonable space of paper-consistent
implementations and unspecified hyperparameters. This is a falsifiable
hypothesis, not a conclusion or an allegation about intent. Reproduction of the
results would be recorded as disconfirming evidence.

## Start here

```bash
git clone https://github.com/fjoad/atk-evidence.git
cd atk-evidence
bash scripts/bootstrap.sh
```

Then follow [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) to acquire and
verify public data and obtain restricted data through official authorization.
`scripts/verify_data.py` provides the machine-readable data gate.

See [`docs/VISION.md`](docs/VISION.md), [`docs/STATUS.md`](docs/STATUS.md), and
[`docs/EVIDENCE-AND-LEARNINGS.md`](docs/EVIDENCE-AND-LEARNINGS.md).

Raw datasets and publication PDFs are not redistributed. Their provenance,
access conditions, checksums, and acquisition procedures are recorded per study
under [`studies/`](studies/).
