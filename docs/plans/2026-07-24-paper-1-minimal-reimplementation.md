# Paper 1 minimal reimplementation

**Created:** 2026-07-24
**Status:** CHECKPOINT 1 passed; compact tiny route complete; full route at the ADASYN executability branch
**Canonical workflow:** [`../../RUNBOOK.md`](../../RUNBOOK.md)

## Objective

Produce a small, independently readable implementation of Paper 1 and obtain
the first eligible full ISET Table III result before expanding the forensic
branch engine, publication layer, or cluster tooling.

The existing implementation and its artifacts remain preserved. They are audit
inputs, not authorities and not dependencies of the new five-file route.

## Scope of the compact route

Create:

```text
studies/atk-2022-deep-autoencoder/reproduction/
  download_data.py
  prepare_data.py
  models.py
  run_experiment.py
  analyze_results.py
```

The five files must contain the scientific implementation directly. They may
reuse a verified parser or checksum constant only after its behavior is traced
back to the PDF/data source. They must not import the old branch lattice,
ordinary/DDP runners, aggregation framework, or implementation-v1 models.

Soft target: no more than roughly 2,000 non-test lines for the first complete
ISET Table III route.

## Why ISET first

ISET/CER has a paper-defined 48-value daily sample that matches the stated
model input. SGCC supplies 1,034 daily features while the paper specifies
48-neuron inputs and gives no conversion. Therefore:

1. ISET Tables III--V are the first reproduction lane.
2. SGCC Table II follows under separately labeled 1,034-to-48 interpretations.

This sequencing does not remove SGCC from the paper verdict.

## Phase 1 — fresh source freeze

Completed 2026-07-24:
[`../../studies/atk-2022-deep-autoencoder/METHOD.md`](../../studies/atk-2022-deep-autoencoder/METHOD.md)
was written from a fresh text extraction and visual inspection of every page
of the SHA-256-pinned PDF. Only after the source draft was complete was it
cross-checked against the reported-table CSVs and historical traceability
inventory.

The source freeze resolves or visibly branches:

- exact ISET population: stated “around 3,000” versus 4,225 official
  residential meters;
- 48-slot day and DST handling;
- six attack equations, including the invalid Attack-3 endpoint;
- whether attacks from all customers enter the anomaly test set;
- customer versus row splitting;
- normalization population and axis;
- paper-positioned test-set ADASYN;
- complete FC-SAE layers and bottleneck interpretation;
- optimizer, batch, epochs, convergence, and validation omissions;
- printed threshold 0.58 versus the non-executable ROC/IQR derivation;
- metric formulas, especially balanced accuracy labeled ACC;
- Table V experiment identity and false-alarm invariance; and
- Table IV sample and timing meanings.

### CHECKPOINT 1 — source agreement

Before model implementation, present the straight-through ISET/FC-SAE reading
and the complete material ambiguity list. Confirm that the paper, not the old
code or contract, supports every step.

**Checkpoint state:** Passed 2026-07-24. No new reproduction model code was
written before approval; the user then authorized uninterrupted continuation.

## Phase 2 — data route

1. `download_data.py` verifies the seven local CER/ScienceDB branch files and
   documents the official authorized route.
2. `prepare_data.py --tiny` parses a small deterministic meter/day subset and
   applies all six attacks.
3. Hand-check each attack and the exact preparation order.
4. `prepare_data.py --full` builds a new compact-route cache. Never relabel the
   implementation-v1 3.2-GiB cache. Preserve the exact pre-ADASYN `B2+M`
   population even if the paper-positioned resampler is not executable.
5. Record counts, identities, hashes, and elapsed time.

### Gate A

- Exact source checksums pass.
- Daily samples have 48 values.
- All attacks match the frozen source reading.
- Train/test identities and ADASYN placement are measurable.
- Tiny and full modes execute the same transformations.

**Tiny Gate-A state (2026-07-24):** Passed. All six archive MD5/ZIP gates and
the allocation semantic branch verify. Twelve real CER residential meters and
240 strict days exercised the same transformations used by full mode. Attack
equations, customer split, all-customer malicious population, joint scaler,
and printed test ADASYN passed hand-checks.

**Full Gate-A state (2026-07-24):** Passed through the exact printed
pre-ADASYN population: 2,251,290 benign profiles, 1,500,520 B1 profiles,
750,770 B2 profiles, 13,507,740 malicious profiles, and 14,258,510 `B2+M`
rows. The selected imbalanced-learn default uses brute-force neighbors at
48 features, so its first full ADASYN query requires about 10.7 trillion
pairwise distances and did not complete in the bounded preparation attempt.
That is retained as an executability observation. `I-ADASYN-NONE` may proceed
on the preserved exact `B2+M` population; any scalable approximate ADASYN run
must remain separately labeled and may not be substituted for P0.

## Phase 3 — first literal model

Implement FC-SAE first:

- all four printed encoder widths;
- full printed mirrored decoder;
- the source-frozen bottleneck interpretation;
- printed hidden/output activations, dropout, optimizer, and MSE;
- reconstruction-MSE anomaly score;
- printed threshold 0.58 for the first direct Table-I replay.

Expose actual runtime layers and validate the decoder output domain against the
standardized targets.

### Gate B

- Layer inventory matches `METHOD.md`.
- Forward pass, gradient, and one update are finite.
- One tiny epoch completes.
- Scores, direction, confusion counts, and all Table III metrics are correct.
- Zero-reconstruction/input-energy and untrained-model baselines are recorded.

**Tiny Gate-B state (2026-07-24):** Passed for FC-SAE. Runtime inventory is
`48-400-300-200-100-100-200-300-400-48`, with all eight hidden Dropout
placements, sigmoid hidden activations, Softmax output, 450,448 parameters,
finite gradients/updates, paper metric formulas, untrained and zero-output
baselines, and a common-benign Table-V invariance check. See
`studies/atk-2022-deep-autoencoder/results/compact_route_tiny_sanity_20260724.json`.

## Phase 4 — first full eligible result

Run FC-SAE Table III on the cluster compute nodes only:

1. seed 11 full-data exploratory anchor;
2. immediately report DR, FA, specificity, precision, balanced ACC, F1, AUC,
   and all timing components beside the published row;
3. if operationally sound, run seeds 22 and 33 without changing the method.

This is the next critical milestone.

The local Mac is not an execution fallback. A local seed-11 attempt was
interrupted after ten epochs on 2026-07-24 and has no evidentiary status.

### CHECKPOINT 2 — result eligibility

Before adding models, verify that the PDF → `METHOD.md` → five files → prepared
cache → result record chain is complete. If not, quarantine the result and fix
the chain.

## Phase 5 — Tables III--V

After FC-SAE:

1. LSTM-SAE;
2. FC-VAE with the stated reconstruction-probability detector;
3. LSTM-VAE;
4. LSTM-AEA;
5. supervised and classical baselines;
6. Table IV half/three-quarter/full training sizes and measured timings;
7. explicit Table V experiment-identity readings.

Each new model first receives a tiny run, then the full predeclared seeds.

## Phase 6 — material interpretations and controls

Run the straight-through printed anchor first. Then vary material ambiguities
one at a time. Use the existing 921-configuration forensic inventory only as a
coverage checklist after anchor results exist; do not put it on the execution
critical path or run its whole matrix automatically.

Corrected leakage-free controls remain a separate `C` result family and wait
until the paper-consistent Tables III--V reconstruction is complete.

## Phase 7 — SGCC, confirmation, and report

1. Implement the finite SGCC 1,034-to-48 readings in the same five-file route.
2. Complete Table II.
3. Freeze tolerances, seeds, branch list, statistics, compute budget, and
   stopping rule.
4. Execute confirmatory runs.
5. Generate the Paper 1 verdict and finish the LaTeX report.
6. Only then begin the scientifically preferred controlled solution.

## Explicit non-goals before the first full FC-SAE result

- no new workflow/orchestration framework;
- no new branch-lattice machinery;
- no GitHub Pages or LaTeX work;
- no generalized DDP runner;
- no precursor-paper research;
- no controlled “better model”; and
- no deletion or mutation of historical evidence.

## Finish condition

Paper 1 is not complete until Tables II--V have eligible repeated-run evidence,
material paper-consistent interpretations have been assessed within the frozen
space, a bounded verdict has been issued, and the report can be rerun from a
fresh public clone plus authorized local data.
