# Decision: numerical, mechanistic, and attainability evidence

**Date:** 2026-08-20

**Status:** Accepted as the scientific direction; repository integration and
current-work drift assessment remain pending

**Scope:** Every audited paper; Paper 1 is the first application

## Why this decision exists

The project has treated reproduction mainly as a numerical question: implement
the paper, test reasonable interpretations, and compare reproduced values with
the published table. That question is necessary, but it is not sufficient for a
paper whose contribution is an explanatory claim about why one model succeeds.

A typical paper-level claim has the form:

> Model `B` outperforms model `A` because additional component `Z` captures or
> exploits structure `S` in the data.

The published metric can fail to reproduce while the proposed mechanism remains
plausible. Conversely, the published metric can be correct while the experiment
does not identify the mechanism at all. A task may be solvable by a watermark,
shortcut, one-feature threshold, or other trivial rule even though the paper
attributes success to learning the intended object or structure.

The audit therefore needs three separate conclusions, supported by three
separate evidence programs. More seeds or more interpretation branches make a
numerical reproduction stronger; they do not automatically establish or refute
the claimed mechanism, and they do not by themselves show whether ordinary
additional search could plausibly reach the target.

This decision preserves that distinction before the existing repository,
runbook, active plan, and evidence record are assessed for drift from it.

## The three questions

### 1. Numerical reproduction

**Question:** Does a faithful execution of the method as written reproduce the
reported numerical pattern?

The evidence is the existing paper-first route:

- reconstruct the complete source with locators;
- distinguish printed, interpretive, and corrected tracks;
- acquire and verify the exact named data;
- execute the frozen method and every predeclared materially reasonable
  source-supported completion;
- preserve seeds, failures, timings, scores, identities, and metrics; and
- compare the complete reported pattern rather than selecting one favorable
  cell or metric.

The bounded result is:

> The reported result was or was not reproduced under the declared data,
> source-supported completions, seeds, partitions, tolerances, and compute
> budget.

This conclusion alone does not establish why the result differs, what the
authors actually implemented, whether the claimed component works, or whether a
configuration outside the tested space could match.

### 2. Mechanism identification

**Question:** Does the experiment demonstrate that `Z` gives `B` the claimed
advantage over `A` by exploiting `S`?

The causal sentence decomposes into distinct propositions:

1. structure `S` exists in the evaluated data;
2. `S` is useful for the prediction or detection target;
3. baseline `A` cannot adequately exploit `S` under a fair implementation;
4. component `Z` gives `B` the capability to exploit `S`;
5. the trained `B` actually uses that capability on the paper's data; and
6. using that capability causes the observed improvement over `A`.

A final accuracy table does not identify those propositions. The audit must
test them directly through trivial baselines, capability-sensitive synthetic
tasks, structure and component ablations, fair matched comparisons, and
inspection of what the fitted models actually score or represent.

The central validity question is:

> Is the benchmark capability-discriminating for the paper's claimed
> contribution?

Two architectures can be different globally but effectively equivalent on the
support of a particular dataset. If `H_A` and `H_B` are their hypothesis
classes and the observed data occupy support `S_data`, it may be that

```text
H_A restricted to S_data ≈ H_B restricted to S_data.
```

The extra capability of `B` can be real in theory yet irrelevant to this task.
For example, a nonlinear classifier has a genuine advantage over a linear
classifier on concentric circles but not on two already linearly separable
blobs. Testing only the blobs cannot support the claim that nonlinear structure
caused the result.

The strongest bounded result is not merely that an ablation changes a number.
It identifies which of the six propositions above survived and which did not.

### 3. Attainability

**Question:** Does the described method show a credible empirical route toward
the reported target, or does the target remain far outside the observed
performance envelope?

This is deliberately intermediate between finite non-reproduction and a proof
of mathematical impossibility. It asks whether ordinary additional search is
supported by any observed trend.

The envelope may vary, one predeclared axis at a time or through a separately
frozen search design:

- materially reasonable source-supported completions;
- random seeds and independent partitions;
- model capacity;
- training-set size;
- training duration and stopping behavior;
- optimization settings;
- preprocessing and evaluation choices;
- thresholds, including exact enumeration for a fixed score vector; and
- cumulative compute and search effort.

Relevant observations include:

- learning curves plateau far below the target;
- increasing capacity, data, epochs, or search budget does not close the gap;
- materially different architectures produce nearly identical score rankings;
- the claimed component has negligible, unstable, or wrong-direction effects;
- oracle threshold selection cannot produce the complete reported pattern;
- the best-observed result versus cumulative trials or compute has flattened;
  and
- mechanism-specific synthetic controls provide no missing capability that
  could plausibly explain the target.

The conclusion remains conditional on the declared envelope. A finite search
cannot rule out an arbitrary hidden configuration or a discontinuous improvement
outside the observed region.

## The geometric intuition

The initial intuition resembled a performance surface over model or search
parameters with a reported target plane far above the visible surface. The
surface appears to flatten, and nothing nearby suggests that ordinary extension
will cross the target.

There are two importantly different versions of this picture.

1. **Structural bound.** A proved range, invariant, information limit, or other
   global constraint places the entire surface below the target. The target is
   impossible under the stated assumptions.
2. **Empirical envelope.** Only a finite region is observed. The response is
   stable or saturating far below the target, but an unobserved region could in
   principle behave differently. The target is unsupported or highly
   implausible within the tested envelope, not proved universally impossible.

The project currently seeks the second conclusion unless a genuine structural
proof is available. It must not convert a fitted logarithmic, power-law, or
other scaling curve into a universal bound. Extrapolation can contextualize the
gap, especially under deliberately optimistic assumptions, but remains an
extrapolation.

## Capability witnesses and synthetic playgrounds

Small synthetic distributions are not substitutes for the paper's named data.
They are diagnostic instruments that isolate whether an architecture has the
capability attributed to it.

Useful witness tasks include:

- linearly separable blobs;
- XOR;
- concentric circles;
- interleaving moons or spirals;
- sequences where order is either essential or deliberately irrelevant;
- examples requiring long-range dependence;
- examples where spatial, graph, or contextual structure is necessary; and
- examples with an explicit trivial shortcut.

Interpretation must distinguish capability from relevance:

| Synthetic witness | Paper data | Interpretation |
|---|---|---|
| `B` beats `A` | `B` beats `A` | Capability may be real and relevant; causal attribution still needs ablation |
| `B` beats `A` | no material difference | `B` has the capability, but the paper's task does not require or exercise it |
| no material difference | no material difference | The written implementation may not realize the claimed capability, or training suppresses it |
| no material difference | paper claims a large advantage | The stated mechanism has not been demonstrated and is in serious tension with the implementation |
| both solve both tasks trivially | both solve the paper task | The diagnostics are too easy or the baseline is already at the task ceiling |

Synthetic evidence cannot reveal unpublished author code or prove that no
implementation can achieve a target. It can show that the described causal
explanation leaves no expected footprint even under controlled conditions.

## Discovery sandbox before the formal audit

The investigation should not begin as a large reproduction program. After an
initial end-to-end reading identifies the paper's central numerical and causal
claims, use a small disposable sandbox to discover what actually needs to be
tested.

The sandbox may be a notebook or short script using NumPy and the minimum model
library needed to instantiate the written architectures. It should avoid the
project's production runner, branch machinery, cluster orchestration, reporting
layer, and full dataset unless one of them is essential to the immediate
question.

A useful starting sequence is:

1. write the paper's central claim as `B > A because Z exploits S`;
2. instantiate the smallest recognizable versions of `A`, `B`, and `B` without
   `Z` from the text, making every necessary guess visible;
3. verify shapes, output ranges, one forward/update step, and the ability to
   overfit a hand-sized sample;
4. construct simple geometric witness distributions on which the added
   capability should be irrelevant, useful, and necessary;
5. compare against zero-parameter, linear, random-feature, or other trivial
   rules before interpreting the elaborate model;
6. progressively add or destroy `S` and observe whether either architecture
   notices;
7. inspect per-example outputs, rankings, representations, and failure modes
   rather than looking only at one aggregate metric; and
8. record wall time and rough resource scaling so the likely cost of a formal
   investigation is visible early.

Exploration is allowed to be curious and adaptive. Its purpose is to expose
candidate explanations, reveal whether the benchmark can distinguish the
claimed capabilities, and locate the smallest experiment whose outcomes differ
under those explanations. It is not confirmatory evidence and must not be
presented as though its hypotheses, settings, or stopping rule were frozen in
advance.

The transition out of the sandbox is explicit:

1. state the discriminating question discovered;
2. record the competing explanations and the outcome each predicts;
3. return to the complete paper and exact data to verify that the question and
   proposed implementations are source-relevant;
4. predeclare the eligible interpretations, controls, metrics, statistical
   units, repetitions, compute budget, and stopping rule; and
5. rerun the formal experiment through the preserved evidence path.

Sandbox results may motivate a formal test but cannot select a favorable paper
interpretation after outcomes are seen. A failed toy implementation may expose
a coding mistake rather than a scientific limitation; a successful toy result
shows capability under that controlled distribution, not reproduction on the
paper's data. Preserve useful exploratory notes, but keep their evidentiary
status visibly separate.

## Breadth first means diagnostic breadth

The first breadth pass covers competing explanations, not every paper table
cell, architecture, or ambiguity branch at full scale. It should run many small,
cheap, discriminating checks that each answer one question and return learning
before the next expensive decision.

Diagnostic breadth may include:

- hand-checkable geometry and output-domain probes;
- zero, mean, one-feature, linear, nearest-neighbor, and random-feature rules;
- tiny overfitting and label-permutation checks;
- synthetic capability witnesses of increasing structural difficulty;
- untrained-versus-trained score comparisons;
- structure-preserving and structure-destroying transformations;
- one component removal at a time;
- score-direction and exact-threshold enumeration on small saved vectors;
- small data-size, capacity, and epoch contrasts; and
- rough runtime and memory measurements.

Run independent cheap questions concurrently when useful, but inspect and
interpret their outputs before expanding the next layer. The purpose is a map
of which explanations remain viable, not a large pile of completed jobs.

Only a surviving, source-relevant question promotes to depth:

1. restate the exact question and competing predictions;
2. freeze the full-data implementation and evidence contract;
3. run one watched eligible anchor;
4. verify scores, identities, and intermediate behavior; and
5. add repetitions or a bounded search only when the anchor shows that
   uncertainty about that question remains scientifically material.

Running one expensive seed for every named architecture is model-family
coverage, not the initial diagnostic breadth envisioned here. Likewise, adding
thousands of lines to support branches before cheap probes establish which
questions matter is execution depth disguised as breadth.

The sandbox and first diagnostic map should remain disposable and small. Stop
and reassess when code, orchestration, or documentation grows while the number
of discriminated explanations does not. Prefer a short direct probe that fails
in minutes over a generalized instrument that can execute many uninformative
full runs.

## Triviality floor before architectural ceilings

Before interpreting a complex-model advantage, establish what the task yields
to the simplest fair instruments through the identical evaluation path:

- a constant or zero-parameter rule;
- class prevalence;
- one-feature thresholds;
- nearest centroid or nearest neighbor;
- PCA plus a linear probe, with the caveat that PCA is unsupervised and low
  variance directions may still carry labels;
- logistic regression or a linear SVM;
- random or untrained features plus a linear head; and
- a small supervised positive control when needed to prove that learnable
  signal exists.

If a trivial rule nearly matches the proposed system, the reported number may be
real while the architectural explanation is unsupported. If every elaborate
model ranks samples almost exactly like a zero or energy baseline, the relevant
question becomes what, if anything, training added.

## Mechanism-specific measurements

No single diagnostic establishes mechanism. Use a convergent set whose outcomes
would differ under the competing explanations:

- destroy `S` while preserving labels and the evaluation path;
- remove `Z` while preserving capacity and training as far as possible;
- compare `A` and `B` with paired seeds and identical partitions;
- vary `A`'s capacity to test the claim that it cannot learn the structure;
- inspect score correlations and per-sample ranking agreement;
- inspect effective rank and relevant low-dimensional projections;
- compare learned representations with tools such as CKA or SVCCA when they
  answer a predeclared question;
- examine whether attention, gates, recurrent state, or other credited
  components vary meaningfully rather than remaining constant or inactive;
- test label permutation and feature or structure shuffles; and
- use positive controls to show that the instrument could have detected the
  claimed effect.

These checks must be capability-sensitive. A test on which both mechanisms make
the same prediction cannot distinguish them.

## Statistical treatment of `B - A`

The relevant effect is not two unrelated best runs. Under matched seed `i`, data
partition, and evaluation path, define

```text
Delta_i = metric(B_i) - metric(A_i).
```

Report the paired distribution and an uncertainty interval. Predeclare a
smallest effect of scientific interest. If the complete uncertainty interval
lies inside that negligible region, an equivalence test can support the bounded
claim that `Z` has no material effect under the tested conditions. If the
interval is wide, the result is inconclusive rather than evidence of equality.

Sampling and uncertainty must respect the real independent unit. Multiple rows,
days, attacks, augmentations, or synthetic siblings from one customer or source
record are not independent repetitions. Use cluster-level resampling or another
appropriate dependence-aware analysis.

For a predeclared stochastic trial-generating process, if zero of `n`
independent eligible trials reaches the declared target, the approximate 95%
`rule of three` gives

```text
P(success under that trial process) <~ 3 / n.
```

This bounds the success probability for the declared distribution of trials. It
does not bound the existence of a specially chosen configuration elsewhere in
the hyperparameter space.

## Search and compute plausibility

Every eligible attempt should record wall time, device count and type, aggregate
device-hours, preparation and scoring time, configuration identity, and the
scientific question it answers. An attainability analysis may report:

- cumulative compute already expended;
- best complete-pattern result after `k` attempts or `C` device-hours;
- marginal improvement as capacity, data, duration, or search grows;
- realistic and ideal-parallel wall-clock estimates; and
- linear, log-compute, power-law, and deliberately optimistic extrapolations,
  each labeled as a model rather than a fact.

Compute extrapolation is supporting context, not proof. A response curve may
change outside the observed region.

Publication intervals are especially weak evidence about historical compute.
They do not reveal when an idea arose, when experiments began, the hardware
available, or the degree of parallelism. Unless the paper states incompatible
runtime and hardware facts, report only what the measured workload would cost
under explicit assumptions. Do not speculate that the authors lacked time to
conceive or execute the work.

## Evidence ladder and permissible conclusions

The conclusion language must match the strength of the evidence.

### Level 1: finite numerical non-reproduction

> The reported results were not reproduced under any of the predeclared,
> materially reasonable source-supported completions tested here.

### Level 2: mechanism not identified

> The evaluation did not establish that component `Z` produced model `B`'s
> reported advantage by exploiting structure `S`. The claimed structure,
> capability, use, or causal effect was absent or not distinguished from the
> tested alternatives.

State exactly which link failed. Do not convert an inconclusive mechanism test
into evidence that the mechanism is absent.

### Level 3: empirical attainability finding

> The reported target lies far outside the observed empirical performance
> envelope. Across source-supported interpretations, matched seeds and
> partitions, model capacities, thresholds, training budgets, ablations, and
> mechanism-specific synthetic controls, the additional architectural
> components produced no stable material advantage, and the observed learning
> and search curves plateaued substantially below the published result. These
> findings make the result highly implausible under the method as described,
> while not excluding an undocumented implementation or an untested procedure
> outside the declared space.

### Level 4: structural impossibility

Use only when a proof establishes a global constraint under explicitly stated
assumptions:

> Under assumptions `X`, the described method cannot attain target `T` because
> bound or invariant `Y` limits every admissible execution to `U < T`.

Statistical saturation, a large search, or a fitted scaling curve is not a
structural proof.

## What absence of author code permits

Without an author implementation or provenance evidence, the audit cannot
determine what code actually ran. It must not state:

> This is not what the authors implemented.

It may state:

> The reported results were not produced by our faithful executions of the
> written method or by the finite family of predeclared source-supported
> completions tested here.

The remaining explanations include an undocumented implementation, omitted
data or evaluation operations, an untested configuration, reporting error, or
another unknown cause. The audit does not select among them without evidence.

Likewise, rejection-level peer-review judgments about unclear methods, weak
construct validity, or unsupported contributions do not require a misconduct
finding. Intent and fabrication require a different and much higher evidentiary
threshold.

## Coverage boundary

Do not claim to test the set of “all possible reasonable assumptions.”
`Reasonable` is not a mathematically closed universe. The defensible target is a
finite, documented coverage closure over every materially distinct
source-supported interpretation identified before outcomes are inspected.

Every conclusion must name:

- datasets and populations;
- interpretations and corrected controls;
- seeds and partitions;
- capacity and hyperparameter envelope;
- target pattern and tolerance;
- statistical unit and uncertainty method;
- compute budget and stopping rule;
- failures and excluded attempts; and
- what remains untested.

## Compressed reconstruction of the reasoning

The reasoning that produced this decision should survive even if the original
conversation does not:

1. Exceptionally clean, monotonic, uniformly flattering results create a
   research question, not an accusation.
2. With no author code, the proper response is to read the complete paper and
   build the smallest transparent reimplementation from a frozen source map.
3. Static feasibility checks and trivial baselines precede expensive training.
4. Numerical non-reproduction alone is ambiguous and cannot establish an
   unobserved implementation or intent.
5. The more important explanatory claim is often `B > A because Z exploits S`.
6. A benchmark may be too trivial, shortcut-ridden, or structurally
   insensitive to distinguish `Z` from its absence. A correct number can still
   support the wrong mechanism.
7. Begin with a small disposable discovery sandbox and diagnostic breadth
   pass: play with the written
   architectures, trivial rules, and synthetic geometries until a genuinely
   discriminating question emerges; then freeze and rerun that question through
   the formal evidence path.
8. Synthetic playgrounds provide capability witnesses: they show whether the
   written architectures differ when the claimed capability is actually
   necessary.
9. Real-data ablations and matched comparisons show whether that capability is
   relevant and used on the paper's task.
10. Learning, capacity, threshold, and search curves define an empirical
   performance envelope. Stable saturation far below the target supports
   conditional implausibility, not universal impossibility.
11. The audit therefore needs three separately earned findings: numerical,
    mechanistic, and attainability.

## Immediate direction and non-action

This decision preserves the new scientific direction. It does not yet assert
that Paper 1 has earned any of the three findings, and it does not retroactively
relabel exploratory results.

The next separate task is to audit the current repository documentation,
active goal, plan, and experiments against this decision; identify what remains
valid, what has drifted, and what evidence is missing; then propose a bounded
documentation and goal correction for user approval before changing the
scientific execution contract.
