# Paper 1 workflow — readable source reconstruction

The earlier embedded Mermaid graph has been replaced by a standalone,
self-contained document:

## [Open the Paper 1 method map](../../site/papers/atk-2022-deep-autoencoder/index.html)

The page follows the paper from:

1. SGCC and ISET/CER source data;
2. the six ISET attack equations;
3. anomaly and supervised preparation paths;
4. benchmark and autoencoder architectures;
5. hyperparameter and threshold selection; to
6. Tables II–V.

It distinguishes:

- **IN PAPER:** stated or printed explicitly;
- **OUR CHOICE:** a documented interpretation where the paper is silent;
- **IMPOSSIBLE:** a printed operation that is contradictory or not uniquely
  executable; and
- **ALONGSIDE:** a scientifically corrected control that is not presented as
  paper reproduction.

The method map is intentionally compact. The executable, claim-by-claim
classification remains in
[`PAPER_TO_CODE_TRACEABILITY.md`](PAPER_TO_CODE_TRACEABILITY.md), and the
machine-readable interpretation inventory remains in
[`BRANCH_COVERAGE_CONTRACT.md`](BRANCH_COVERAGE_CONTRACT.md).
