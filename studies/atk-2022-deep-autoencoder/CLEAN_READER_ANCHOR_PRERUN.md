# `CR-ISET-FCSAE-01` pre-run contract

**Frozen:** 2026-08-24

**Status:** `BLOCKED` at the exact official allocation data gate; no eligible
preparation, training, or scoring attempt exists

**Evidence question:** numerical reproduction (`N`)

**Implementation semantics:** paper-literal `P` where executable plus the
approved minimal `I` completions in
[`CLEAN_READER_SPECIFICATION.md`](CLEAN_READER_SPECIFICATION.md)

## 1. Exact question and target

Can one competent-reader implementation of the publication's ISET FC-SAE route,
using the named data and the fully frozen completion, recover the complete
Table-III FC-SAE row?

| Metric | DR | FA | SP | PR | ACC | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reported target (%) | 81 | 15 | 85 | 81 | 83 | 81 | 81 |

The anchor is not judged by one favored metric. Phase 6 will compare the entire
row, confusion counts, rankings, score distributions, provenance, and the
measuring chain.

## 2. Competing outcomes frozen before execution

1. The complete row is closely recovered under the frozen route.
2. Exact preparation, training, serialization, or scoring exposes an
   implementation/source defect and the attempt ends as a preserved failure.
3. The route executes, but a materially distinct source-supported completion
   would be required to approach the published row.
4. The route executes and yields a numerical non-match inside this finite
   contract.
5. An undocumented procedure outside the declared space remains possible and
   is not inferred from any non-match.

No numerical tolerance is used to stop or retry the attempt. The initial
finding reports exact differences and remains one-run descriptive evidence,
not a repeated-seed confidence statement.

## 3. Frozen code and environment

The eligible execution commit is full Git commit
`1e428ffddeee400f790c21b69812cf2a1a9e62bb`. The Slurm wrapper refuses a
different checkout.

| Artifact | SHA-256 |
|---|---|
| `requirements-lock.txt` | `813d84ccf6230dd3821e94a6b280da94644b9a5e58046db3c48c035e127b3277` |
| `download_data.py` | `fe09c03a63a71dfab5fae6008c6bb91e4091a504b07b4d38d89bc72ef83ad192` |
| `prepare_data.py` | `bebbed81ae8fe24671c31a83b53d150a93c1dab3d8f8b16b4d869d7caf0ba850` |
| `models.py` | `3515415082b26bb91cb5367effbd1eba4324bf250ec47e799f7fccb3e6df83f0` |
| `run_experiment.py` | `ef13f49788cbc49cde11d39c4422844d6e3e09f4e38c0cee762ea03cf91a6fc1` |
| `analyze_results.py` | `26ffa06ca24396b95dc01336d96c3070be0d0cf417132b69dcc35b19c561d62a` |
| `run_clean_reader_anchor.sbatch` | `d247fa8d42fde436ffb82846d2b1fac74bf0cc0152f20180d16506cb89859544` |
| clean-reader specification | `5eb02149e5e90bbd4139aa143b76dd6a825d69587a626ab301c67b8d4c1eb9f1` |

The Panther `.venv` was checked without importing the heavy neural runtime on
the login node. It matches the lock for the consequential packages: Python
3.12.2, Keras 3.15.0, PyTorch 2.7.1, NumPy 2.5.1, pandas 3.0.3,
scikit-learn 1.9.0, imbalanced-learn 0.14.2, and PyArrow 24.0.0. The runner
enables deterministic PyTorch algorithms with warning-on-unsupported behavior,
disables cuDNN benchmarking, enables deterministic cuDNN, sets the CUDA BLAS
workspace configuration, and records the effective state and device.

## 4. Exact named-data gate

The six available local/Panther consumption archives have the exact official
byte sizes and MD5 identities frozen in the specification. A byte-identical
mirror is not a semantic substitution.

The seventh required input is:

- file: `SME and Residential allocations.tab`;
- bytes: `196316`;
- MD5: `124c10711ab1e7c52cb7317c8f69e42e`.

On 2026-08-24 this file was absent both locally and under Panther's remote
repository, and `ISSDA_API_TOKEN` was absent in both environments. Therefore
the source gate is `BLOCKED`. The available
`SME_and_Residential_allocations.csv` is a separately frozen semantic branch
and is not eligible for this attempt without a new visible `I` decision and
user review.

## 5. Statistical unit and frozen implementation

- Root seed: `20260824`; one run.
- Unit: one strict 48-slot residential customer-day.
- Population: every officially residential meter, complete six-archive
  interval, no imputation/truncation of malformed days.
- Attacks: frozen 1–6 definitions, including duration-first in-day Attack 3.
- Scaling: feature-wise population mean/standard deviation over complete
  benign plus six-attack population, before splitting.
- Split: customer-disjoint 2:1 `B1/B2`; training is benign `B1`; original test
  is benign `B2` plus all six attacks derived only from `B2`.
- Resampling: test-only ADASYN to equality, Euclidean distance, five neighbors,
  root seed; original/synthetic identity preserved.
- Model: `48-400-300-200-100-100-200-300-400-48`, sigmoid hidden layers,
  dropout 0.4 after every hidden layer, Softmax output, MSE, Adam `1e-3`.
- Fit: batch 32; shuffle each epoch; maximum 100; minimum 10; patience 5 at
  improvement `1e-6`; restore lowest training-loss weights; no validation set.
- Score: per-row mean MSE; malicious iff score is strictly greater than 0.58;
  continuous high-score AUC.
- Diagnostic recorded beside, never substituted for, the model score: exact
  per-row MSE infimum over the closed probability simplex.

## 6. Compute and stopping budget

- One Panther `gpu-all` Slurm job.
- At most one GPU, 16 CPU cores, and 96 GiB RAM.
- Wall-time limit: 36 hours.
- Preparation and the one training/scoring attempt execute sequentially in the
  same job.
- The known exact ADASYN cost is explicitly acknowledged; no approximate
  neighbor search is permitted.
- A timeout, OOM, unsupported deterministic operation, data error, or other
  literal operational failure is preserved as the anchor outcome. There is no
  adaptive retry.
- After success or failure, stop. Do not change a completion, seed, threshold,
  model, budget, or control before complete inspection and Checkpoint 2.

## 7. Exact submission and inspection commands

The repository must first be synchronized and checked out exactly at the
eligible commit. Once the official `.tab` passes its size/MD5 gate, the sole
submission command is:

```bash
sbatch --export=ALL,EXPECTED_COMMIT=1e428ffddeee400f790c21b69812cf2a1a9e62bb \
  studies/atk-2022-deep-autoencoder/reproduction/run_clean_reader_anchor.sbatch
```

No submission occurred while this record was written. After the immutable
attempt finishes, Phase 6 runs the fail-closed audit against its `result.json`:

```bash
.venv/bin/python \
  studies/atk-2022-deep-autoencoder/reproduction/analyze_results.py \
  --audit-clean-reader-anchor /absolute/path/to/result.json
```

The audit verifies configuration and source metadata, prepared/run hashes,
required artifacts, independently regenerated predictions and all metrics,
history/stopping counts, score/floor alignment, and a full score audit. The
result then feeds only the bounded initial `N` finding. `M` and `A` remain open.
