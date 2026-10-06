# Running the preparation

Verify existing inputs against official metadata and archive integrity:

```bash
.venv/bin/python studies/takiddin-2021-robust-poisoning/reproduction/download_data.py --online --check-crc
```

Run hand-checkable software fixtures locally:

```bash
.venv/bin/python -m unittest tests.test_robust_preparation -v
```

Real-data preparation refuses to run outside a Slurm allocation. The short
`reproduction/run_preparation_check.sbatch` wrapper takes the checkout,
revision, raw-data directory, Python environment, and new output directory as
explicit environment inputs. It prepares 20 customers and 28 complete days
each at 0% and 30% poisoning on both paths, including one customer-specific
example. Prepared arrays stay in ignored data directories. Existing output
directories are never overwritten.
