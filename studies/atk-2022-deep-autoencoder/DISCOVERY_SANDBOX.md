# Phase 2 discovery sandbox

**Authorized:** 2026-08-24

**Status:** First wave complete; stopped after the single frozen job

**Classification:** exploratory `X`; candidate questions touch `N`, `M`, and
`A`, but no sandbox outcome is eligible evidence for any of them

**Source anchor:**
[`CLEAN_READER_ORIENTATION.md`](CLEAN_READER_ORIENTATION.md)

## Boundary

This is a disposable source-derived sandbox, not a reimplementation of the
paper's numerical experiment. It uses no named data, historical project model,
production runner, branch machinery, prior score vector, or prior result. It
cannot select a primary source completion or become confirmation
retrospectively.

Experimental computation must run on a cluster compute node. Local work is
limited to writing the script, syntax/static checks, documentation, transfer,
and result inspection.

## First-wave question

Can the smallest transparent observations distinguish four explanations that
remain conflated by the paper?

1. the synthetic attacks are largely separable through simple statistics;
2. temporal order is the one structure that can require a sequential
   capability;
3. the printed output activations impose a reconstruction-domain floor; and
4. Gaussian reconstruction probability must be low, not high, for anomaly.

## Competing predictions

### X1 — triviality floor

**Minimal setup:** Generate positive 48-step daily profiles from a fixed
asymmetric shape with random amplitude, offset, phase, and noise. Apply toy
versions of the six printed attack descriptions. Fit benign-only median/MAD
calibrations for energy, mean, variance, range, zero count, roughness, and
linear trend. Preserve every feature's AUC and oracle balanced accuracy; do not
select a favorable feature as a paper result.

- If attacks 1--5 contain obvious shortcuts, at least one simple feature will
  strongly separate most of them.
- If the task genuinely requires elaborate representation learning, these
  rules will remain near chance on most attacks.
- Reversal preserves each profile's multiset, mean, variance, range, and total;
  order-insensitive features should therefore tie exactly or nearly exactly.

### X2 — temporal witness

**Minimal setup:** Create benign cyclic shifts of an asymmetric sawtooth-like
sequence. Compare amplitude reduction, block disruption, and reversal. Train
one small undercomplete dense autoencoder and one similarly sized seq2seq LSTM
autoencoder on the same benign training rows with linear reconstruction heads,
one seed, fixed epochs, and no tuning. Linear heads deliberately remove the
known Softmax/sigmoid domain confound so this toy question focuses on temporal
capability.

- If recurrence supplies a task-relevant capability, the LSTM should react
  more strongly than the dense model when local order is necessary, while
  both may react to amplitude changes.
- If both models rank the order-sensitive anomalies similarly, this toy
  benchmark does not demonstrate a recurrence-specific advantage.
- If neither reacts, either the training/setup suppresses the capability or
  the witness is not discriminating; that outcome is a sandbox failure, not
  mechanism-absence evidence.

### X3 — output-domain consequence

**Minimal setup:** Standardize the complete toy profile collection. For benign
and each attack, compute the fraction of coordinates outside each decoder's
range and the exact squared-distance lower bound to the probability simplex
and unit box.

- Softmax and sigmoid lower bounds should be positive whenever standardized
  rows lie outside their output sets.
- The bound may differ between benign and attacked rows; if it does, a
  reconstruction score can be driven by domain geometry rather than learned
  structure.
- This cannot bound DR, FA, AUC, or the published result.

### X4 — VAE score direction

**Minimal setup:** Evaluate fixed-unit Gaussian reconstruction density for an
explicit increasing sequence of reconstruction errors.

- Probability must decrease monotonically with error.
- Therefore low probability is anomaly-consistent; a shared
  greater-than-threshold anomaly rule applied directly to probability points
  in the opposite direction.

## Fixed execution contract

- Script:
  `exploration/phase2_discovery_sandbox.py`.
- Result:
  `exploration/results/phase2_seed_20260824.json`.
- Seed: `20260824`.
- Synthetic sizes: 512 benign training profiles, 256 benign test profiles, 256
  profiles per attack/witness class.
- Sequence width: 48.
- Neural comparison: one dense AE and one seq2seq LSTM AE; fixed architecture,
  25 epochs, batch 64, Adam `1e-3`, MSE, linear outputs.
- Budget: one cluster job, one accelerator at most, 20 minutes wall time, no
  hyperparameter search, no rerun for a more favorable outcome.
- Promotion rule: a question may enter Phase 3 only if the observation differs
  across competing predictions and remains directly relevant to a
  source-located paper claim.
- Stopping rule: stop after this one complete result, inspect every section,
  and decide explicitly whether any failed diagnostic needs redesign. Do not
  proceed automatically to VAE training, attention training, named data, or
  production code.

## Results

Job `381540` completed on one Panther GPU node in 2:25 with exit code `0:0`.
The script reported 60.06 seconds of CUDA execution. The complete JSON has
SHA-256
`cef6e4d18ac765dcd5ba02b79c5deb51eace393c2670bee05e1fd54e577f2da8`.
All observations below remain exploratory `X`; they are not paper
reproduction, mechanism, or attainability evidence.

### X1 — triviality floor

The strongest single predeclared feature for each toy attack was:

| Toy attack | Best feature | AUC | Oracle balanced accuracy |
|---|---:|---:|---:|
| 1 fixed reduction | roughness | 0.993 | 0.965 |
| 2 dynamic reduction | roughness | 1.000 | 1.000 |
| 3 forward bypass | zero count | 1.000 | 1.000 |
| 4 daily mean | variance | 1.000 | 1.000 |
| 5 randomized mean | roughness | 1.000 | 1.000 |
| 6 reversal | linear trend | 0.661 | 0.662 |

For reversal, the pairwise differences in energy, mean, variance, range, and
zero count were at most `2.84e-14`, and their AUCs were approximately 0.5.
Thus this toy generator behaves as predicted: attacks 1--5 expose simple
shortcuts, while reversal defeats multiset-only summaries. This does not show
that the named data have the same separability.

### X2 — temporal witness

| Toy anomaly | Dense AE AUC | Seq2seq LSTM AE AUC |
|---|---:|---:|
| Amplitude reduction | 1.000 | 0.903 |
| Block disruption | 0.950 | 0.521 |
| Reversal | 0.542 | 0.503 |

Both models had finite first gradients and decreasing losses. Their parameter
counts were comparable (2,792 dense; 2,683 recurrent), but final training loss
was 0.345 for the dense model and 0.944 for the recurrent model. The recurrent
model therefore did not demonstrate the proposed temporal advantage: the
dense model detected block disruption while both were effectively at chance
on reversal. Because the recurrent model also fit the benign generator much
less well, this observation does not distinguish an absent capability from an
underfit model or a weak witness. It is a failed mechanism witness, not evidence
that recurrence is ineffective.

### X3 — output-domain consequence

Mean exact reconstruction-MSE lower bounds after the frozen standardization
were:

| Population | Sigmoid/unit-box floor | Softmax/simplex floor |
|---|---:|---:|
| Benign | 0.014 | 0.577 |
| Attack 1 | 1.083 | 1.120 |
| Attack 2 | 1.067 | 1.075 |
| Attack 3 | 2.420 | 2.611 |
| Attack 4 | 0.045 | 0.712 |
| Attack 5 | 1.018 | 1.027 |
| Attack 6 | 0.018 | 0.593 |

The Softmax floor was positive for every row in every population. Mean floors
differed substantially between benign data and toy attacks 1--5, so decoder
range geometry alone can contribute to apparent anomaly separation without a
learned structural representation. Reversal was geometrically similar to
benign data. These are exact bounds for the toy vectors, not bounds on the
paper's DR, FA, AUC, or reported table cells.

### X4 — VAE score direction

For reconstruction errors `[0, 0.25, 0.5, 1, 2, 4]`, fixed-unit Gaussian
relative probability was
`[1, 0.969, 0.882, 0.607, 0.135, 0.000335]`. It decreased strictly, confirming
that low reconstruction probability is anomaly-consistent. This is an
algebraic scoring-direction check, not a trained VAE result.

### Promotion and stop decision

- Carry the simple-rule floor into the source freeze as a candidate formal `M`
  control: exact attacks must be compared with zero-parameter and one-feature
  rules through the identical evaluation path before architectural necessity
  is credited.
- Carry reversal forward as the cleanest source-located temporal witness, but
  do not promote the present dense/LSTM outcome. A later formal `M` test would
  need matched fitting success and a witness whose competing predictions
  isolate temporal capability.
- Freeze preprocessing and decoder output domains explicitly because their
  geometry can affect reconstruction scores before model sophistication does.
- Freeze both literal and anomaly-consistent VAE score orientations because
  the paper's prose and shared threshold rule point in opposite directions.
- Do not run a result-guided temporal retry, variance model, or attention model
  in the sandbox. The first wave has identified the necessary source questions,
  and the predeclared stopping rule requires returning to the paper now.

## Operational log

- 2026-08-24: froze the contract above before implementation and committed the
  standalone script and short Slurm wrapper in `83dab57`.
- 2026-08-24: local static verification passed: Python AST parsing, Slurm
  wrapper shell syntax, forbidden-import inspection, and `git diff --check`.
  The repository's deterministic suite also passed: 140 study tests and 33
  root tests. None of these checks executed the experimental sandbox.
- 2026-08-24: the cluster was reachable and the `panther` host-key fingerprint
  matched the already trusted aliases, but batch authentication failed because
  the local SSH agent reported no identities. The interactive password prompt
  was cancelled without entering a credential. No remote command, checkout
  update, `sbatch` submission, or experimental computation occurred.
- 2026-08-24: the user clarified that Panther uses interactive password
  authentication. The credential was used only in interactive SSH/SCP prompts
  and was not persisted. The clean remote checkout was fast-forwarded to
  `c415c26`; job `381540` then ran exactly once and completed successfully.
- 2026-08-24: the raw JSON and scheduler log were transferred without
  transformation, their checksum was verified locally, and the first-wave
  stopping rule was applied. No follow-up sandbox job was submitted.
