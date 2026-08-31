# `CR-ISET-FCSAE-01` pre-run contract

**Frozen:** 2026-08-24; source branch refrozen 2026-08-30 before execution

**Execution status (observed 2026-08-31; frozen contract unchanged):** Panther
job `384390` completed once with exit `0:0` at 22:48:39 Qatar time on
2026-08-30, after 9:14:27, under the approved
`sciencedb-csv-semantic-equivalence-v1` allocation `I` branch. Independent
Phase-6 artifact audit passed on 2026-08-31; the bounded initial non-reproduction
is saved in [CLEAN_READER_FINDING.md](CLEAN_READER_FINDING.md). Checkpoint 2
awaits review, and no second submission is authorized.

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
`a88d17477ad96b01ffa44a50d8ce051dd8d2b5ca`. The Slurm wrapper refuses a
different checkout.

| Artifact | SHA-256 |
|---|---|
| `requirements-lock.txt` | `813d84ccf6230dd3821e94a6b280da94644b9a5e58046db3c48c035e127b3277` |
| `download_data.py` | `fe09c03a63a71dfab5fae6008c6bb91e4091a504b07b4d38d89bc72ef83ad192` |
| `prepare_data.py` | `2daea25d3a4112bec411d061958770a92836855274904f6d342249cfd425a5c7` |
| `models.py` | `3515415082b26bb91cb5367effbd1eba4324bf250ec47e799f7fccb3e6df83f0` |
| `run_experiment.py` | `5a522b5a298998eedf6aca2e18d1f9ec40b8b280c431df56ae0eff7a3b3dd51f` |
| `analyze_results.py` | `ecc7586a23bf9ccbccaee56a8433db38b6059f6b7a737ef11bfa13b78d945023` |
| `run_clean_reader_anchor.sbatch` | `b8e183aad54ff3ab84530be80af3f1a59562d81cd3ff79ad9d2744f8c225b98b` |
| clean-reader specification | `5eb02149e5e90bbd4139aa143b76dd6a825d69587a626ab301c67b8d4c1eb9f1` |

The Panther `.venv` was checked without importing the heavy neural runtime on
the login node. It matches the lock for the consequential packages: Python
3.12.2, Keras 3.15.0, PyTorch 2.7.1, NumPy 2.5.1, pandas 3.0.3,
scikit-learn 1.9.0, imbalanced-learn 0.14.2, and PyArrow 24.0.0. The runner
enables deterministic PyTorch algorithms with warning-on-unsupported behavior,
disables cuDNN benchmarking, enables deterministic cuDNN, sets the CUDA BLAS
workspace configuration, and records the effective state and device.

## 4. Named-data identity and approved serialization branch

The six available local/Panther consumption archives have the exact official
byte sizes and MD5 identities frozen in the specification. A byte-identical
mirror is not a semantic substitution.

The originally frozen seventh input was:

- file: `SME and Residential allocations.tab`;
- bytes: `196316`;
- MD5: `124c10711ab1e7c52cb7317c8f69e42e`.

On 2026-08-24 this file was absent both locally and under Panther's remote
repository, and `ISSDA_API_TOKEN` was absent in both environments. Official
file-ID-808 metadata checked on 2026-08-25 shows that this `.tab` is
Dataverse's archival ingest of an originally uploaded
`SME and Residential allocations.xlsx` (185,480 bytes). The allocation
information itself is not missing: the available ScienceDB CSV and an
independent public GitHub workbook agree on all 6,445 mappings under the
predeclared blank/zero normalization. Therefore the source gate remains
`BLOCKED` for the byte-identical official serialization branch, and that
failure remains preserved.

On 2026-08-30 the user explicitly approved the separately named
`sciencedb-csv-semantic-equivalence-v1` allocation branch for this one anchor.
Its frozen allocation file is:

- file: `SME_and_Residential_allocations.csv`;
- bytes: `112589`;
- MD5: `89263f89253cf56b857079986ae73096`;
- SHA-256:
  `96298be047f34ba91fe281c899b440d2b28747b4f102af6f239dbbd93dd354d4`;
- rows/unique meter IDs: `6445` / `6445`;
- allocation counts: Code 1 `4225`, Code 2 `485`, Code 3 `1735`; and
- canonical normalized five-column SHA-256:
  `b6e5ac79964c991d820e359c4413990e8bb60d8f051f4a7bb795d7bae60516c8`.

The public GitHub workbook independently agrees across all 6,445 semantic
rows after the predeclared inapplicable-zero/blank normalization. The clean
reader code and audit now fail closed unless this exact branch name and every
file verification are present. No other experimental field changed. See
[`docs/decisions/2026-08-30-clean-reader-semantic-allocation-admission.md`](../../docs/decisions/2026-08-30-clean-reader-semantic-allocation-admission.md).

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
eligible commit. Once the approved semantic branch passes its full gate, the
sole submission command is:

```bash
sbatch --export=ALL,EXPECTED_COMMIT=a88d17477ad96b01ffa44a50d8ce051dd8d2b5ca \
  studies/atk-2022-deep-autoencoder/reproduction/run_clean_reader_anchor.sbatch
```

The command above was submitted once on 2026-08-30 as Panther job `384390`.
After the immutable attempt finishes, Phase 6 runs the fail-closed audit
against its `result.json`:

```bash
.venv/bin/python \
  studies/atk-2022-deep-autoencoder/reproduction/analyze_results.py \
  --audit-clean-reader-anchor /absolute/path/to/result.json
```

The audit verifies configuration and source metadata, prepared/run hashes,
required artifacts, independently regenerated predictions and all metrics,
history/stopping counts, score/floor alignment, and a full score audit. The
result then feeds only the bounded initial `N` finding. `M` and `A` remain open.
