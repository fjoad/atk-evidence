# Reproducibility log

## 2026-07-20 - Initial audit and acquisition

- Read the complete 12-page journal article and visually checked its figures and tables.
- Confirmed that the article reports timing but no hardware or sufficiently detailed software/training configuration.
- Located the SGCC dataset through the corresponding author's page and its linked repository.
- Cloned repository commit `8db682e65422d24689a61bd044eab7235121c5df`.
- Recorded SHA-256 checksums for all three multipart archive files.
- The built-in macOS unzip produced a CRC failure because it does not support the archive format.
- Installed Homebrew `sevenzip` 26.02.
- Tested the three-volume archive successfully with 7-Zip, then extracted `data.csv`.
- Verified extracted CSV SHA-256: `99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7`.
- Profiled the CSV: 42,372 customers, 1,034 date columns, 8.53% positive labels, 25.64% missing consumption cells.
- Observed that source date columns are lexicographically rather than chronologically ordered.
- Located the official Irish CER/ISET record: DOI `10.7929/ISSDA/BX59EU`.
- Downloaded and checksum-verified its unrestricted manifest and documentation.
- Confirmed through official metadata that the six consumption archives are restricted and total approximately 658 MB.
- Searched this machine for a previously downloaded authorized CER archive; none was found.
- Created a project-local Python 3.12 environment and froze its dependency versions.
- Implemented and unit-tested all six CER attack functions. The controlled bypass implementation interprets the stated duration as hours and corrects the printed subtraction of interval length to addition; both choices are explicitly documented.
- Implemented a machine-readable audit of Tables II and III. The reported precision values do not agree with DR and FA under the stated balanced test-set protocol, and the rows imply different positive-class prevalences.
- Ran a first SGCC customer-profile sanity audit using a fixed 60/20/20 customer split, train-only preprocessing, and a validation-selected 5% false-alarm threshold. This is not an exact paper reproduction because the paper does not define its SGCC 48-value input construction.
- Preliminary SGCC sanity result: simple summary/logistic/PCA models achieved test ROC-AUC values of approximately 0.59-0.70, not saturation. This result concerns the real SGCC customer labels; it does not test the much easier synthetic CER attacks.
- Located and read the two same-author precursor papers containing the basic-AE and VAE experiments. Visually verified their threshold and result tables against the journal article.
- Confirmed exact recurrence of the ISET FC/LSTM basic-AE thresholds and TPR/FPR/AUC values, and exact recurrence of the FC/LSTM VAE thresholds and DR/FA values.
- The ISSCS precursor states that malicious data are used only for testing but derives thresholds from ROC curves; the EUSIPCO precursor explicitly applies ADASYN within the test set and also derives thresholds from ROC curves. This is evidence of test-dependent model selection under the written protocols, not evidence that the numerical measurements were fabricated.
- Neither precursor supplies the missing hardware, epoch, batch-size, seed, repetition, or uncertainty information.

## Local execution environment

- Hardware: Apple MacBook Pro, Apple M1 Max, 10 CPU cores, 32 GPU cores, 64 GB unified memory.
- OS: macOS 26.5.2, build 25F84.
- Experiment environment: a project-local Python virtual environment will be used and package versions frozen after installation.

Hardware serial numbers and device identifiers are intentionally omitted.

## Decisions requiring explicit provenance

- Do not use an unofficial CER mirror unless its legal status is clear and its files match the official MD5 values.
- Never modify raw downloaded files in place.
- Preserve failed extraction artifacts until the verified copy and checksums are recorded; failed artifacts are excluded from all analyses.
- Separate paper-literal reproduction from controlled scientific evaluation.
- For the controlled CER bypass attack, use 4-24 actual hours (8-48 half-hour slots) and `tf = ti + tl`; retain a separate literal implementation only if needed for diagnostic comparison.
