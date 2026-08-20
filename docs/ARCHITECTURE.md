# ATK Evidence — Scientific Architecture

**Last updated:** 2026-08-20

## Purpose

The repository is a domain-neutral, paper-by-paper evidence system. It asks
three questions that must be answered separately:

1. **Numerical (`N`):** does the method as described recover the reported
   numerical result?
2. **Mechanistic (`M`):** does the evidence identify the claimed reason for an
   advantage—for example, that model `B` beats model `A` because component `Z`
   exploits structure `S`?
3. **Attainability (`A`):** where does the reported target lie relative to the
   declared empirical performance envelope, and does ordinary additional
   search show a credible route to it?

These are evidence questions, not three labels for the same experiment. A
numerical miss does not establish mechanism failure or unattainability. A
mechanism failure does not imply the reported number was never observed. An
empirical plateau is not a proof over every possible implementation.

## Scientific lifecycle

The architecture has two broad modes separated by a source freeze.

### Discovery mode

After an initial end-to-end paper read, a disposable sandbox is used to learn
what the decisive questions are. It may use toy or synthetic data, trivial
rules, the smallest recognizable versions of `A`, `B`, and `B-Z`, output
inspection, and static calculations. Its job is to expose competing
explanations cheaply.

Discovery proceeds breadth-first: many small probes that distinguish possible
stories before one expensive implementation path. It is adaptive and may
generate hypotheses, but its results are not confirmatory evidence and cannot
silently redefine the paper.

### Formal evidence mode

The researcher returns to the complete source, freezes a source-located
executable specification and causal claim map, acquires the exact data, and
builds the smallest transparent instrument that can execute the frozen
reading. Only questions that survive cheap diagnostics are promoted to costly
depth. Formal evidence is then confirmed under predeclared targets, seeds,
partitions, statistics, budgets, and stopping rules.

## Two orthogonal classifications

Every material run or static analysis carries two labels.

### Implementation semantics

- **`P` — paper-literal:** directly executes the printed operation, including
  a literal failure when the operation cannot execute.
- **`I` — reasonable interpretation:** a source-supported completion of a
  material omission or contradiction.
- **`C` — controlled analysis:** a deliberate correction, ablation, positive
  control, synthetic witness, or alternative method used to test an
  explanation.
- **`X` — exploratory:** adaptive discovery whose question or procedure was
  not frozen in advance.

### Evidence question

- **`N` — numerical reproduction**
- **`M` — mechanism identification**
- **`A` — empirical attainability**

The classifications are independent. A run can be `P/N`, `C/M`, or `I/A`.
One artifact can inform more than one question, but each inference must be
stated and bounded separately. `X` work can motivate a formal question; it
cannot be promoted retrospectively into confirmation.

## Evidence layers

1. **Source layer:** complete paper PDFs, official dataset metadata, raw files,
   hashes, and source-located quotations or paraphrases.
2. **Claim layer:** numerical targets, the causal claim map, required
   structures and capabilities, omissions, contradictions, and material
   competing explanations.
3. **Discovery layer:** disposable sandbox code and outputs, toy witnesses,
   trivial baselines, static checks, and the diagnostic breadth map.
4. **Specification layer:** frozen executable reading, reasonable
   interpretations, promoted `N/M/A` questions, and predicted discriminating
   outcomes.
5. **Contract layer:** targets, tolerances, independent units, partitions,
   seeds, statistics, compute envelope, budgets, and stopping rules.
6. **Implementation layer:** the minimal paper-literal instrument and clearly
   separated interpretation and controlled-analysis tracks.
7. **Run layer:** immutable configuration, environment, seed, scores,
   predictions, timing, completion or failure status, and raw outputs.
8. **Assessment layer:** separate numerical, mechanism, and attainability
   analyses against their frozen contracts.
9. **Report layer:** one bounded finding for each evidence question, followed
   by limitations and the combined scientific interpretation.

No layer can substitute for an earlier one. Passing software tests cannot fix
a mistaken paper reading. A sandbox pattern cannot replace a formal run. A
large sweep cannot establish a mechanism unless its outcomes discriminate that
mechanism from alternatives.

## Repository structure

```text
atk-evidence/
  RUNBOOK.md                 # Canonical end-to-end audit workflow
  docs/                      # Vision, architecture, status, evidence, decisions
  papers/                    # Local source PDFs; ignored by Git
  data/                      # Local raw/derived datasets; ignored by Git
  scripts/                   # Environment, verification, and deterministic tests
  studies/
    registry.toml            # Stable cross-domain paper registry
    <study-id>/
      METHOD.md              # Source freeze and executable claim map
      reproduction/          # Minimal paper-facing scientific instrument
        download_data.py
        prepare_data.py
        models.py
        run_experiment.py
        analyze_results.py
      results/               # Machine-readable summaries; large arrays ignored
  reports/
    <study-id>/              # Standalone report source
    synthesis/               # Later cross-paper synthesis
  site/
    index.html               # Public project overview
    papers/<study-id>/       # Self-contained readable paper maps
  AGENTS.md                  # Shared operational contract
```

Each paper receives a registered, self-contained study directory. Reuse is
permitted for mechanical operations, never as a reason to hide paper meaning
inside a general experiment platform. Historical or forensic machinery may be
preserved as evidence, but it is neither scientific authority nor the default
path for a new audit.

## Key interfaces

### Source and claim map

`METHOD.md` records both what was printed and what the audit will test. In
addition to numerical cells, it expresses central explanatory claims in a form
such as:

> `B` outperforms `A` because additional component `Z` captures or exploits
> structure `S`.

The formal program then asks whether `S` exists and matters, whether `A` lacks
the relevant capability, whether `Z` supplies it, whether trained `B` uses it,
and whether that use causally explains the measured advantage. A headline
metric alone cannot identify those links.

### Dataset manifest

Each dataset record identifies the authoritative source, access status,
version, expected files, official checksum when available, local checksum, and
transform provenance. Code consumes verified local paths, never an unrecorded
substitute. Restricted datasets carry access instructions and checksum
verification, never embedded credentials.

### Discovery record

The sandbox record is intentionally lightweight. It preserves the question,
minimal setup, observation, competing explanations affected, and whether a
formal test was promoted. Disposable code may remain ignored or under a clearly
marked exploratory path. It must not become an undocumented second
implementation or be cited as confirmatory evidence.

### Formal evidence contract

Before confirmation, the active plan freezes:

- the `N`, `M`, or `A` question;
- the `P`, `I`, or `C` implementation semantics;
- the paper claim or causal link at issue;
- competing predictions and disconfirming outcomes;
- exact data, independent unit, split, seeds, statistics, and tolerance;
- the finite capacity/search/compute envelope;
- promotion and stopping rules; and
- the report finding the result will change.

### Run record

Every formal run is reconstructable from source revision, data hash,
interpretation, evidence question, hyperparameters, seed, split, environment,
raw scores or predictions, metrics, duration, and completion or failure status.
All attempts remain visible; no best-seed or favorable-branch selection is
allowed.

### Mechanism assessment

Mechanism evidence uses capability-sensitive tests: synthetic witnesses,
structure-preserving and structure-destroying controls, component ablations,
trivial baselines, learned-behavior inspection, and paired effects where
appropriate. These tests must distinguish the claimed explanation from simpler
or incompatible explanations. Model-family coverage by itself is not
mechanism identification.

### Attainability assessment

Attainability is an empirical envelope over declared axes such as reasonable
interpretations, seeds, partitions, capacity, optimization budget, thresholds,
and mechanism-specific controls. The assessment reports best observed values,
distributions, failures, response curves, plateaus, and distance to target.

Static bounds can establish literal impossibility only when their assumptions
exactly match the printed protocol. Otherwise the strongest ordinary result is
bounded implausibility: the target lies far outside the tested envelope and no
observed trend suggests that routine additional search would close the gap.
Extrapolation and compute estimates are contextual evidence, not universal
bounds or claims about what authors could have done.

### Three findings

Each report answers three questions independently:

1. **Numerical finding:** reproduced, partially reproduced, not reproduced, or
   not executable under the declared paper-consistent space.
2. **Mechanism finding:** supported, not identified, contradicted by a
   discriminating control, or not testable from the described experiment.
3. **Attainability finding:** within the observed envelope, outside but
   plausibly trending toward it, far outside with plateauing evidence, or
   literally incompatible with a proved printed constraint.

The combined interpretation states what follows from their intersection and
what remains open. It never infers author intent or an undocumented
implementation.

## Data flow

1. Register the paper and preserve the complete source.
2. Read it end to end and draft numerical and causal claim maps.
3. Use a disposable sandbox and cheap diagnostic breadth to expose competing
   explanations and decisive tests.
4. Return to the source and freeze the executable specification.
5. Acquire, checksum, and inspect the exact named data.
6. Build the smallest transparent paper-literal instrument.
7. Pass deterministic, one-step, triviality, and output-inspection checks.
8. Obtain one eligible anchor and inspect the complete artifact.
9. Promote only unresolved `N/M/A` questions whose expected information gain
   justifies deeper execution.
10. Execute finite numerical branches, mechanism controls, and attainability
    curves as separately identified work.
11. Freeze and run confirmation without post-hoc selection.
12. Analyze uncertainty, failures, causal discrimination, and the empirical
    envelope.
13. Update the durable evidence ledger and issue three bounded findings.
14. Publish the paper report and public evidence map.
15. Only under a later, separate contract, design a better solution to the
    underlying research problem.

## Design decisions

- **Paper-first source authority:** author code is a relevant artifact, not a
  license to replace the printed method silently.
- **Discovery/formal separation:** adaptive play is encouraged while its
  evidentiary status remains explicit.
- **Breadth before depth:** cheap tests of competing explanations prevent
  expensive execution from becoming a substitute for experimental design.
- **Literal/interpretation/controlled separation:** repairs and alternatives
  cannot overwrite the primary paper-literal question.
- **Three independent findings:** numerical agreement, causal explanation, and
  empirical attainability require different evidence.
- **Predeclared finite search:** bounded conclusions remain testable without
  pretending to exhaust an infinite implementation space.
- **Local raw inputs, versioned provenance:** restricted or copyrighted
  material is not redistributed.
- **Per-paper isolation:** conclusions and assumptions do not leak across
  papers.

## Extension rule

Add a paper only after registering its source, claims, datasets, and
independent evidence contract. Add shared infrastructure only after repeated
eligible work demonstrates a concrete mechanical need. Add a scientific
correction only under a named controlled track. Add a report only when its
three findings can be regenerated from preserved evidence.
