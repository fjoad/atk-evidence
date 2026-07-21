# ATK Evidence — Vision

**Last updated:** 2026-07-21

## Thesis

Published numerical claims should be reproducible from the data, algorithms,
and procedures the paper describes. ATK Evidence builds the strongest
independently rerunnable evidence available for or against such claims, one
paper at a time. It begins by testing the hypothesis that selected results by
Abdulrahman Takiddin and coauthors cannot be reproduced reliably within a
predeclared, scientifically reasonable space of paper-consistent
implementations and unspecified hyperparameters. The hypothesis may be
falsified: a stable, legitimate reproduction is a valid and welcome outcome.

## Primary hypothesis

For each paper `P`, beginning with the 2022 deep-autoencoder paper:

> A faithful implementation using the named dataset, the algorithms and
> processing steps explicitly stated by `P`, and documented reasonable
> interpretations of omissions will not reproduce the complete principal
> numerical result pattern within predeclared tolerances across repeated runs.

The finite scientific claim is always bounded by the registered implementation
space, search budget, metrics, tolerances, seeds, and data partitions. The goal
is evidence strong enough to make alternative paper-consistent explanations
increasingly implausible, never a claim of logical impossibility.

## What this project does

- Reconstructs each paper's literal experimental protocol before introducing
  corrected or alternative methodology.
- Separates explicit instructions, omissions, contradictions, and reasonable
  ambiguity branches in a versioned experiment contract.
- Acquires the exact named datasets through authorized sources and records
  provenance, versions, checksums, and access restrictions.
- Runs predeclared small experiments, hyperparameter searches, repeated seeds,
  and statistical tests without selecting favorable runs after the fact.
- Preserves code, configurations, raw scores, logs, and both supporting and
  disconfirming evidence.
- Publishes scripts and explicit authorized-access instructions so an
  independent researcher can recreate every non-redistributed input.
- Produces a standalone LaTeX reproduction/rebuttal report for each paper and a
  combined synthesis across the selected corpus.

## Methodological priority

The **paper-literal track is primary**. It implements only what the paper says.
It must not silently repair questionable preprocessing, change the evaluation,
parameter-match architectures, or substitute a different dataset. When the
paper is ambiguous, each reasonable interpretation is labeled and justified.

A separate **controlled-analysis track** may later test corrected methodology
or causal architectural questions. Its results must never be presented as the
paper-literal reproduction.

## Falsification and reproduction criteria

Before confirmatory runs for a paper, the project will freeze:

- the reported claims and numerical targets being reproduced;
- allowable implementation branches for every material ambiguity;
- the hyperparameter search envelope and computational stopping rule;
- seeds, partitions, statistical tests, uncertainty estimates, and tolerances;
- criteria for a complete reproduction rather than a match to one metric.

If a paper-consistent configuration reproduces the principal results reliably,
the hypothesis for that paper is rejected or weakened and that outcome is
reported plainly. A lucky run, test-set selection, undocumented extra method,
or match to one isolated number is not a complete reproduction.

## Non-goals

- Inferring or alleging author intent from a reproduction failure.
- Claiming that an unlimited and untested parameter space has been exhausted.
- Improving the method before testing what the paper actually reports.
- Cherry-picking seeds, branches, metrics, or attacks that favor the hypothesis.
- Generalizing one paper's verdict to another without independently testing it.
- Redistributing restricted datasets or copyrighted publication PDFs.
- Treating a written argument alone as proof; substantive claims require
  rerunnable artifacts and appropriately powered statistical evidence.

## Success criteria

- [ ] The three-paper core corpus is identified and registered; additional
      papers can be added without changing the evidentiary rules.
- [ ] Every target paper has a frozen paper-literal experiment contract,
      ambiguity register, code, environment lock, and machine-readable runs.
- [ ] Every primary claim receives a repeated-run statistical assessment with
      an explicit reproduction verdict and limitations.
- [ ] Disconfirming results are preserved with the same prominence as evidence
      supporting the hypothesis.
- [ ] Each paper has a reproducible LaTeX report, and a combined report compares
      recurring findings without overstating them.
- [ ] A new Claude or Codex session can recover the exact state and next action
      from Charter documents without relying on chat history.
- [ ] A fresh public clone can bootstrap its environment, acquire or request all
      named data, verify every available checksum, and rerun the published audit.

## Key domain concepts

**Paper-literal reproduction:** An implementation restricted to the named data,
algorithms, processing steps, and evaluation described by the target paper.

**Ambiguity branch:** A documented, reasonable interpretation of a material
detail the paper omits or states inconsistently.

**Reproduction contract:** The frozen targets, branches, search envelope,
statistics, tolerances, and stopping rules established before confirmatory runs.

**Complete result pattern:** The jointly reported principal metrics,
architecture ordering, and relevant per-attack results—not one convenient
number considered in isolation.

**Evidence record:** Versioned artifacts and conclusions labeled VERIFIED,
OBSERVED, INFERRED, HYPOTHESIS, INVALIDATED, or OPEN.

## Future scope

- Extend the same protocol from the initial three-paper corpus to additional
  same-author papers, other domains, and eventually other corpora only after
  each current paper receives its own independent verdict.
- Publish reusable reproduction tooling once the paper-specific pipelines have
  stabilized.
