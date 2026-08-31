# Why this project exists

A researcher reads a paper, follows its method, and asks a simple first
question:

> If I do what the paper says, do I obtain what the paper reports?

Sometimes the answer is yes. Sometimes it is no. A large mismatch is important,
but it does not explain itself. The implementation may be wrong, the paper may
omit a consequential operation, the model may be poorly matched to the task,
or the reported result may contain an error. The purpose of this project is to
turn that uncertainty into testable questions.

## The claim behind a result table

A paper often says more than “model B scored higher than model A.” It says:

> B is better than A because component Z captures structure S.

That explanation requires several things to be true: S must be present and
useful; A must lack something important; Z must supply it; the trained B must
actually use it; and that use must cause the measured advantage.

A final accuracy table cannot establish all of those links. A task may be
solved by a simple shortcut. Both models may rank examples the same way. An
extra component may exist in the code without doing useful work.

## Three questions

Every study therefore separates:

1. **Reproduction:** did the stated method recover the reported numbers?
2. **Explanation:** did the added architecture provide the capability credited
   for the advantage?
3. **Reachability:** within a clearly defined set of data, configurations,
   seeds, and compute, is there a credible route to the target?

These questions need different evidence. A failed reproduction is not proof
that every configuration fails. A reproduced result is not proof of the
proposed explanation. A flat set of experiments can make a target highly
implausible within that set without proving universal impossibility.

## What “useful work” means

For a complex model, the interesting quantity is what it contributes beyond a
simple fair comparison. Training may consume hours while the final score still
follows an input statistic available without learning.

The project asks:

- Does training improve held-out performance over zero, untrained, and simple
  controls?
- Does the added component improve a matched model without it?
- Does that improvement disappear when the claimed structure is removed?
- Is any gain large enough to matter, with uncertainty narrow enough to rule
  out a meaningful effect?

A near-perfect correlation with a simple score is a clue, not proof of equality.
“No useful contribution” requires a predeclared meaningful effect and an
appropriate equivalence or bound.

## How the investigation proceeds

Read the complete paper. Play cheaply with small examples and synthetic
geometries. Return to the source and write an executable, source-located
specification. Acquire the named data. Build the smallest transparent
implementation. Check simple alternatives and saved artifacts. Run one full
anchor. Then deepen only the questions that remain scientifically material.

The project reports every attempt and its limitations. It welcomes results that
contradict the investigator's initial suspicion. It publishes corrections to
its own analysis. It does not infer author intent from a table, missing code, or
a reproduction failure.

The operational details are in [RUNBOOK.md](../RUNBOOK.md). The durable
definitions and evidence boundaries are in the
[three-part evidence frame](decisions/2026-08-20-three-part-evidence-frame.md).
