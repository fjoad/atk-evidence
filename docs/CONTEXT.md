# Durable facts

Facts that stay true across experiments. Current progress belongs in
[STATUS.md](STATUS.md), and the way we work is in [APPROACH.md](APPROACH.md). The
full history of the earlier process is in `docs/archive/CONTEXT-2026-10-06.md`.
Never write passwords, tokens or keys here.

## Panther cluster

- Off site, connect to the QCRI VPN first. A dropped VPN shows up as both name
  resolution failures and SSH timeouts; reconnect rather than trying the IP.
- Log in with `ssh -i ~/.ssh/araclaw_qcri fjoad@panther.qcri.org` (the local SSH
  config has a matching `Host panther.qcri.org` entry).
- Home is `/export/home/fjoad`. The shared project area is
  `/export/home/fjoad/atk-evidence`. Use `umask 0077` before cloning or copying
  data.
- Before Slurm commands: `source /etc/profile.d/modules.sh && module load slurm`.
- Use the login node only for git, transfers, submission and monitoring. It kills
  heavy work: environment installs and whole-file hashing of large files have both
  been killed. Install environments inside a job, and hash large files in chunks.
- Useful commands: `squeue -u "$USER"`, `sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,AllocTRES`,
  `scancel JOB_ID`.
- A short reference lives in `~/Documents/projects/panther-usage-guide/README.md`.

## Slurm requests that have worked

- CPU work: partition `cpu-all`, 4 CPUs, 16 GiB, 15 minutes for preparation and
  classical models; 1 CPU and 8 GiB for small checks.
- GPU work: `--gres=gpu:v100_16GB:1`, 4 CPUs, 16 GiB, on `gpu-short` (2-hour
  limit) or `gpu-all` for longer jobs.
- Idle GPUs don't guarantee a job starts; the whole CPU/memory request has to fit
  on the node.
- A wrong Slurm output path stops the job before the script runs. Check the log
  directory exists.
- After a disconnect, check `squeue`/`sacct` and the output folder before
  resubmitting; a lost connection doesn't mean the job stopped.

## Remote layout

- Each experiment runs from its own clean checkout under `/export/home/fjoad/`, at a
  fixed commit.
- Shared Python environments: `/export/home/fjoad/atk-evidence/data/environments/`.
  The TensorFlow environment for the poisoning paper is `robust-feed-forward-2.16.2`
  (TensorFlow 2.16.2, Keras 3.4.1).
- Large outputs for the poisoning paper: `/export/home/fjoad/atk-evidence/data/derived/takiddin-2021-robust-poisoning/`,
  one folder per attempt. Compact summaries are copied into the study's `results/`
  folder in git; large copies go to the ignored local `data/derived/`.
- Panther's driver can't run CUDA 13 packages. The PyTorch environment is pinned
  to PyTorch 2.7.1 with CUDA 12.6.
- TensorFlow input pipelines must stay on the CPU when their dataset operations
  have no GPU kernels, even when the model runs on the GPU.

## Local machine

- Apple M1 Max, macOS. The main environment is `.venv`. Create it with
  `bash scripts/bootstrap.sh`, setting `PYTHON_BIN` if Python 3.12 isn't the
  default. If pip is missing, run `ensurepip` first.
- Run tests with `KERAS_TORCH_DEVICE=cpu bash scripts/test.sh`. The macOS GPU
  backend fails on some operations.
- Pinned side environments: `tmp/robust-feedforward-venv/` for the TensorFlow
  models. AdaBoost needs scikit-learn 1.5.2 with NumPy 1.26.4, SciPy 1.13.1,
  joblib 1.4.2 and threadpoolctl 3.5.0. The SVM uses scikit-learn 1.9.0 with NumPy
  2.5.1 and SciPy 1.18.0. Tests that need these environments are skipped by the
  main suite.
- Build jobs started from Claude Code through the Codex companion run in a
  sandbox without network access, so they can't reach Panther on their own.

## Data and papers

- Electricity data (Irish CER smart-meter trial): the ScienceDB copy is in
  `data/raw/cer-sciencedb/`. Its six consumption ZIPs match the official sizes, MD5
  checksums and ZIP checks. Its allocation CSV matches all 6,445 rows of the
  official workbook but is not the official `.tab` file. Official files, if
  obtained, go in `data/raw/cer-authorized/` with their original names.
- SGCC data: `bash scripts/acquire_sgcc.sh`, extracted to
  `data/raw/sgcc-verified/data.csv`. macOS `unzip` fails on this multipart archive;
  7-Zip works.
- Verify data with `.venv/bin/python scripts/verify_data.py --strict`. It accepts
  either complete CER copy.
- Paper PDFs live in `papers/`. Raw data and PDFs are never committed; files are
  identified by checksum and left unchanged.
- Don't search credential stores, browser storage, shell history or wide areas of
  the home folder looking for data access. A past broad search likely triggered
  the institution's endpoint protection.

## Small things that cost time before

- In zsh, `path` is tied to `PATH`; never use it as a loop variable.
- Slurm can report 2 logical CPUs for a 1-CPU request on nodes with two threads per
  core. The task still gets one CPU.
- When code changes after a run, check that run's code against the commit it
  recorded, not today's files.
