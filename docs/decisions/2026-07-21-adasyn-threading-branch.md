# ADASYN thread environment is part of the execution branch

**Date:** 2026-07-21
**Status:** Accepted for the authorized exploratory run

The exact verified SGCC file, frozen config, source code, Python packages, and
random seed do not produce a byte-identical ADASYN cardinality across thread
settings. On the same local machine, `OMP_NUM_THREADS` and `MKL_NUM_THREADS`
set to 1, 2, and 4 produced respectively 77,712, 77,712, and 77,708 supervised
rows after ADASYN; the unconstrained local environment produced 77,710.
the cluster preflights likewise recorded 77,712 rows with two or sixteen threads
and 77,710 with eight threads.

The current the cluster four-GPU branch fixes both variables at 2, matching the
resource probes and producing 77,712 rows, a 51,808-sample outer training
partition, and inner supervised train/validation sizes of 44,036 and 7,772.
The production runner must derive and record these values at runtime rather
than hard-code them, and its sharder must remain correct if a documented
thread-setting branch yields unequal or zero-sample rank tails.

This small cardinality sensitivity is evidence that hardware/runtime details
omitted by the paper affect exact data construction. It is not a plausible
explanation for the observed tens-of-percentage-point gaps in the completed
classical metrics, and it does not itself establish non-reproducibility.
Confirmatory work must freeze the thread environment or explicitly register
multiple branches before outcomes are examined.

Machine-readable observations are in
`studies/atk-2022-deep-autoencoder/results/adasyn_thread_sensitivity.json`.
