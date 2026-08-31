# Small follow-up: is the within-energy signal specific to training?

Date: 2026-08-31

Classification: adaptive exploratory `X/M`, after the frozen primary diagnostic

The user approved small discriminating controls after the first checks. The
completed primary analysis found within-energy-band AUC 65.24% for trained
scores versus 49.63% for input energy. This is counterevidence to score identity,
but does not distinguish learning from deterministic input geometry.

## Question and competing predictions

On identical original rows and identical energy bands, compare trained scores
with a constant uniform Softmax reconstruction and the nearest simplex
reconstruction, neither of which requires training.

- If training's within-band distinction is specific to the fitted model, its
  within-band AUC should exceed the no-training controls materially.
- If output/input geometry explains the distinction, the projection or uniform
  rule may produce a comparable or stronger within-band ranking.
- Either outcome is exploratory and does not identify a causal mechanism or
  establish equivalence. No source interpretation is changed.

## Frozen setup before this probe

- Same immutable source result, input identities, high-error direction, and
  analysis functions as the primary `1175e8d` check.
- Uniformly sample 10,000 distinct original source days without replacement,
  seed `20260831`; include each day's benign profile and all six attack
  derivatives, 70,000 rows total. Exclude synthetic rows.
- Use the same 100 pooled label-blind energy-quantile bins for every score.
  Compute energy, trained, uniform, and projection within-bin AUC, weighted by
  the number of positive/negative pairs. No target-guided direction selection.
- Read saved trained, zero, and projection scores; compute the uniform score
  on these selected inputs only. Verify consumed hashes and sample identities.
- One additional CPU-only allocation, two cores, 4 GiB, three-minute cap,
  no retry, training, parameter search, bootstrap, or new figures. This is the
  conditional third-step control, not a repeat of the primary full-data pass.
- Preserve sample-index checksum, script/source hashes, measurements, runtime,
  and all outcomes. Stop after this probe and report the completed round.

The bound already excludes a seed-only rescue. This control addresses only how
to interpret the nonzero residual information; it cannot remove that bound.
