# Exploratory Reproduction Assessment

The run is not confirmatory, so these criteria organize comparison without
claiming a preregistered verdict.

- Report every model/seed and mean, standard deviation, minimum, and maximum.
- Compare each metric to the displayed paper precision; record absolute
  percentage-point error.
- A row is a close numerical match only when DR, FA, SP, PR, ACC, F1, and AUC
  are all within 2 percentage points for at least two of three seeds.
- For each saved score vector, enumerate every deterministic threshold and
  minimize the largest absolute gap across all seven metrics. A minimum above
  2 points proves that threshold choice alone cannot make that executed model
  a close match; it does not prove that every possible training run fails.
- Breadth seed 11 locates divergences. Repeated-seed inference begins only for
  the frozen surviving interpretations; it must report every seed and its
  uncertainty rather than selecting the closest run.
- A table pattern is a close match only when every proposed model row is close
  and the paper's complete ordering is retained; one lucky row is insufficient.
- Benchmark results are assessed separately because their algorithms are less
  completely specified.
- Table IV timing is descriptive on our hardware; only accuracy uses the
  two-point comparison band.
- Table V fixed-model FA must be invariant across attacks. Any comparison to
  varying published FA values explicitly records the structural mismatch.
- `BLOCKED_EXACT_DATA` is the only valid value when official inputs fail MD5 or
  access checks.
