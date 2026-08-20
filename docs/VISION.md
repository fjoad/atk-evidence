# ATK Evidence — Vision

**Last updated:** 2026-08-20

## Thesis

A published result normally carries more than a number. It also carries an
explanation: that one method outperforms another because some added component
captures structure in the data. ATK Evidence audits both the number and the
explanation, then asks whether the described method shows any credible route to
the reported target.

Every paper therefore receives three separately earned findings:

1. **Numerical reproduction:** does the reported result pattern emerge from the
   method as written and the finite family of predeclared source-supported
   completions?
2. **Mechanism identification:** does the experiment demonstrate that the
   credited component supplies, uses, and benefits from the capability claimed
   to explain the result?
3. **Attainability:** does the target lie inside a credible empirical
   performance envelope, or remain far beyond every observed trend under the
   declared search and compute budget?

These findings are independent. A number may reproduce while its claimed
mechanism remains unidentified. A number may fail to reproduce while the
mechanism remains plausible. A large finite non-match may support empirical
implausibility without proving mathematical impossibility.

## The claim behind the table

The central explanatory claim of a comparative paper is written explicitly as:

> Model `B` outperforms model `A` because component `Z` exploits structure
> `S`.

That sentence contains at least six propositions:

1. `S` exists in the evaluated data;
2. `S` is useful for the target;
3. `A` cannot adequately exploit `S` under a fair implementation;
4. `Z` gives `B` the relevant capability;
5. the trained `B` actually uses that capability; and
6. using it causes the observed advantage.

A table of final metrics does not identify those propositions. A benchmark may
be solvable by a shortcut, watermark, one-feature rule, or other structure that
both models exploit equally. The reported number can then be correct while the
architectural explanation is unsupported.

## Scientific posture

Suspicion starts an investigation; it does not decide its outcome. Exceptionally
clean rankings, monotonic improvements, missing variance, vague methods, or
apparently excessive architectural claims become concrete questions to test.
Every experiment must be capable of weakening the audit's initial doubt.

The project is open to all outcomes:

- a stable reproduction is reported plainly;
- a mechanism-confirming ablation is reported as readily as a null ablation;
- an attainable target is not described as implausible merely because one run
  failed; and
- a stable paper-consistent non-reproduction is a valid bounded finding.

The project does not infer intent, fabrication, or an unobserved implementation
from a paper, absent code, or reproduction failure. Those claims require
different evidence.

### Table patterns are triage, not a verdict

Perfectly ordered improvements, uniformly flattering metrics, missing
variance, tidy increments, and a discussion that anticipates every result can
justify closer inspection. The audit may recompute metric identities, enumerate
feasible confusion matrices, inspect rounding and digit patterns, or evaluate
an ordering under an explicitly stated probability model. Correlated metrics,
shared test sets, and selection effects must not be treated as independent
samples. These checks prioritize questions and can expose internal
contradictions; cosmetic neatness alone is never evidence of intent or a
substitute for the three formal findings.

## Discovery before formal execution

Each paper begins with an end-to-end orientation read and a small disposable
discovery sandbox. The sandbox uses the minimum code and data needed to expose
the central numerical and causal questions. It may inspect author code when
available, instantiate the written architectures, generate simple geometric
witness distributions, test trivial rules, and examine whether claimed
components behave differently at all.

Sandbox work is adaptive and exploratory. It discovers discriminating
questions; it does not produce eligible reproduction or confirmatory evidence.
Before formal execution, the project returns to the complete paper and exact
data, freezes the source-supported interpretations and competing predictions,
and reruns promoted questions through the preserved evidence path.

Author code, when available, is an artifact to dissect after the paper's claims
are understood. It is not an authority that silently rewrites the publication.
When no code exists, the project builds the smallest transparent
reimplementation supported by the text.

## Diagnostic breadth before execution depth

Breadth first means many cheap, question-specific probes across competing
explanations—not one expensive full run for every model or every ambiguity
branch.

The initial map may include:

- static arithmetic and shape checks;
- zero, mean, one-feature, linear, nearest-neighbor, and random-feature rules;
- tiny overfitting and label-permutation checks;
- synthetic capability witnesses;
- structure-preserving and structure-destroying transformations;
- one-component ablations;
- untrained-versus-trained score comparisons;
- score direction and exact-threshold feasibility;
- small data, capacity, epoch, runtime, and memory contrasts; and
- positive controls proving that the instrument can detect the expected effect.

Each probe answers one question and is interpreted before the next expensive
decision. Only a surviving source-relevant uncertainty promotes to full data,
repeated seeds, or a bounded search. Code, orchestration, and documentation must
not grow faster than the number of explanations actually discriminated.

## Two orthogonal classifications

The project uses two different classifications that must not be conflated.

### Implementation track

- `P`: the executable printed procedure, including statistically questionable
  operations;
- `I`: a predeclared materially defensible completion of an omission or
  contradiction;
- `C`: a scientifically corrected control; and
- `X`: adaptive exploratory or externally motivated work whose question or
  procedure was not source-frozen; it cannot count as reproduction or be
  promoted retrospectively into confirmation.

### Evidence question

- `N`: numerical reproduction;
- `M`: mechanism identification; and
- `A`: attainability.

A `C` ablation can answer a mechanism question without becoming a repaired
reproduction. A `P` run can contribute to numerical, mechanism, and
attainability evidence at once, but each inference remains separately bounded.

## Numerical reproduction contract

Formal numerical work:

- reconstructs the complete paper before eligible model execution;
- identifies exact data, populations, sample units, preparation order, models,
  scores, thresholds, metrics, and timing claims;
- preserves literal non-executable outcomes rather than silently fixing them;
- predeclares a finite coverage closure over every materially distinct
  source-supported completion;
- separates corrected controls from paper-consistent evidence;
- preserves every seed, failure, score, identity, timing, and configuration;
  and
- compares the complete reported pattern rather than selecting one metric,
  branch, or lucky run.

The conclusion is always bounded by the declared data, interpretations,
hyperparameter envelope, seeds, partitions, tolerances, compute, and stopping
rule.

## Mechanism-identification contract

Formal mechanism work tests the links in `B > A because Z exploits S`:

- establish whether `S` exists before measuring it on contaminated or
  transformed inputs;
- establish a triviality floor through the identical evaluation path;
- use synthetic witnesses where the capability is irrelevant, useful, and
  necessary;
- destroy `S` while preserving labels and evaluation;
- remove `Z` while matching all other choices as closely as possible;
- vary `A`'s capacity when the paper claims it cannot learn `S`;
- compare `A`, `B`, and `B-Z` with matched seeds and partitions;
- inspect per-example rankings, representations, gates, attention, or other
  credited internal behavior; and
- use equivalence intervals or other predeclared effect criteria rather than
  interpreting an insignificant difference as equality.

A mechanism finding states exactly which link was supported, contradicted, or
left unidentified.

## Attainability contract

Attainability is not inferred from one failed reproduction. It is studied
through a predeclared empirical envelope over scientifically relevant axes such
as data size, capacity, duration, seeds, partitions, thresholds, source-supported
completions, and cumulative compute.

The project records:

- learning and capacity curves;
- best complete-pattern result versus trials and device-hours;
- threshold-feasible operating regions for saved scores;
- marginal improvement from additional data, capacity, duration, or search;
- failures and saturation as well as improvements; and
- realistic and deliberately optimistic extrapolations, visibly labeled as
  models rather than bounds.

Only a proved invariant, range restriction, or other global argument supports a
claim of structural impossibility. Stable saturation far below a target supports
conditional empirical implausibility within the declared envelope.

Publication intervals do not establish how long authors searched or what
hardware they used. Compute comparisons report explicit assumptions and do not
speculate about undocumented history.

## Exact data and minimal instruments

The exact named data are a hard gate for eligible numerical claims. A proxy can
support exploratory discovery only when labeled ineligible for reproduction.
Raw files remain immutable and local; provenance, checksums, access routes,
code, configurations, and summary evidence are versioned.

Each formal paper route uses the smallest readable instrument that exposes the
paper's scientific meaning directly. Shared code performs only stable
mechanical work. A disposable sandbox remains separate from the formal
instrument, and a historical forensic harness never becomes the source
authority.

## Confirmation and reporting

Exploratory results do not become confirmatory retrospectively. Before depth,
freeze the relevant question, predictions, eligible implementations, metrics,
statistical units, repetitions, uncertainty method, compute budget, promotion
rule, and stopping rule.

Every paper report contains three distinct assessments:

1. **Numerical finding:** reproduced, partially reproduced, not reproduced, or
   non-executable within the stated finite contract.
2. **Mechanism finding:** supported, contradicted, not identified, or not
   tested, with the failed or surviving causal links named.
3. **Attainability finding:** inside the observed envelope, outside it but
   unresolved, highly implausible within it, structurally impossible under
   stated assumptions, or not tested.

The combined conclusion cannot be stronger than its weakest necessary evidence
link. Author intent and universal impossibility remain outside the verdict
unless independently established by appropriate evidence.

## Non-goals

- Treating suspicion as evidence.
- Inferring what unpublished code did when no provenance establishes it.
- Claiming that a finite search exhausted every imaginable implementation.
- Improving a method before preserving what the paper actually describes.
- Treating a corrected match as reproduction.
- Using a trivial or shortcut-ridden benchmark to validate an architectural
  mechanism.
- Reporting a best seed, branch, metric, or extrapolation as the result.
- Generalizing one paper's finding to another without an independent audit.
- Turning the audit into a workflow platform whose construction delays
  discriminating evidence.

## Success criteria

- [ ] Every paper has a complete source-located method and causal-claim map.
- [ ] Every paper begins with a recorded discovery sandbox and diagnostic
      breadth map whose outputs remain visibly exploratory.
- [ ] Every numerical target has a finite paper-consistent reproduction
      contract and repeated assessment or explicit non-executable outcome.
- [ ] Every claimed architectural advantage has capability-sensitive witnesses,
      fair baselines, and matched mechanism tests, or is explicitly untested.
- [ ] Every attainability conclusion has a declared envelope, compute budget,
      stopping rule, and uncertainty boundary.
- [ ] Disconfirming and confirming evidence, failures, and corrections receive
      equal prominence.
- [ ] Every paper report issues three separately bounded findings.
- [ ] A fresh public clone can verify every redistributable input and rerun the
      published audit without private paths, credentials, or chat history.
- [ ] A new agent can recover the scientific direction, current state, and next
      discriminating question from shared documents alone.

## Key concepts

**Capability-discriminating benchmark:** a task whose outcomes differ depending
on whether the claimed capability is present and used.

**Discovery sandbox:** a small adaptive exploratory environment used to find
discriminating questions before the formal evidence contract is frozen.

**Diagnostic breadth:** many cheap tests of competing explanations, interpreted
before any survivor receives expensive depth.

**Empirical performance envelope:** the observed relationship between complete
performance and declared data, model, training, evaluation, and compute axes.

**Finite coverage closure:** the documented set of every materially distinct
source-supported interpretation identified before outcomes, never “all possible
reasonable assumptions.”

**Triviality floor:** performance achieved by the simplest fair rule through the
same evaluation path.

**Structural bound:** a proved necessary constraint applying to every admissible
execution under stated assumptions.

## Future scope

Extend the protocol only after independent paper audits show which mechanical
operations are genuinely stable across studies. Cross-paper synthesis compares
recurring evidence patterns only after each paper has earned its own numerical,
mechanistic, and attainability findings.
