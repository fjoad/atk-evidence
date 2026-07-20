# Electricity-Theft Paper Reproducibility Audit

This repository is a rigorous, paper-by-paper reproduction audit of selected
electricity-theft detection research by Abdulrahman Takiddin and coauthors.
The first target is:

> A. Takiddin, M. Ismail, U. Zafar, and E. Serpedin, "Deep Autoencoder-Based
> Anomaly Detection of Electricity Theft Cyberattacks in Smart Grids," IEEE
> Systems Journal, 16(3), 4106-4117, 2022.

The primary experiment is always the **paper-literal reproduction**: implement
what the paper explicitly says, use the named data, and add nothing silently.
When necessary details are missing, reasonable paper-consistent interpretations
are enumerated and documented before results are examined.

The working hypothesis is that the reported numerical results will not be
reproduced reliably within a predeclared, reasonable space of paper-consistent
implementations and unspecified hyperparameters. This is a falsifiable
hypothesis, not a conclusion or an allegation about intent. Reproduction of the
results would be recorded as disconfirming evidence.

See [`docs/VISION.md`](docs/VISION.md), [`docs/STATUS.md`](docs/STATUS.md), and
[`docs/EVIDENCE-AND-LEARNINGS.md`](docs/EVIDENCE-AND-LEARNINGS.md).

Raw datasets and publication PDFs are not redistributed. Their provenance,
access conditions, and checksums are recorded under `replication/`.

