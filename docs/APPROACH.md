# How this project works

## What we do

We take a published paper whose results we doubt, rebuild the code it
describes, run that code, and compare our numbers with the published ones. We
work through papers one at a time, and the list grows as we go.

Authors often don't publish their code. A paper still has to describe its
method well enough for someone else to rebuild it; that is the point of
publishing a method. So we treat the paper as the specification and do what a
careful outside researcher would do: write the code the paper describes, run the
experiment it describes, and see whether the published results come out.

It is a little like reverse-engineering software. We can't see the original
code, but we can see what it produced (the published tables) and we have a
detailed description of how it works (the paper). We build a version from the
description and check whether it behaves the same way.

## Where we start from

We start from doubt: the results in these papers looked implausible to us. That
doubt is a hypothesis to test, not a conclusion.

The burden is on us. The test has to be fair enough that a skeptical reader, or
the authors themselves, can't wave the outcome away, whichever way it goes. If
our rebuild matches the paper, we say so as plainly as we would report a
mismatch. Being shown wrong is an acceptable outcome; an unfair test is not.

## The best-faith remake

Every paper leaves gaps: settings it doesn't state, details it borrows from
other papers, the occasional typo or contradiction. We fill each gap the way a
reasonable outside researcher would if they wanted to build on the work:

- **Anything unstated:** use the defaults of the software the paper names, such
  as scikit-learn or Keras.
- **An obvious typo:** make the obvious fix.
- **A detail taken from a cited paper:** follow the cited paper.
- **Two readings that conflict:** try both quickly and keep the one that works
  better for the paper.
- **Samples the paper doesn't identify,** such as which customers: choose them
  at random, with the choice fixed in advance.

Every choice goes on one page per paper, `studies/<paper>/ASSUMPTIONS.md`,
before the final run.

We are generous to the paper at every gap on purpose. It makes the test
stronger: if even the most favorable reasonable rebuild can't get close to the
published numbers, "you implemented it wrong" has nowhere to go.

## How the work runs: breadth first, then one frozen run

1. **Read the whole paper.** List its claims, its experiments, every stated
   setting and every gap. Do the free checks first: whether the published
   numbers agree with each other, and whether the stated hardware and training
   time could cover the stated workload. These need no code, so they can be
   done for several papers before rebuilding any of them.
2. **Build the whole remake:** the data preparation and every model the paper
   reports, baselines included. Test the code on small made-up data where the
   right answer is known.
3. **Run everything small.** All models and all conditions on a small slice of
   the real data, with short training. This is where bugs are cheap to catch and
   where we move fast: try things, fix them, rerun. Results at this stage are
   tests of our code, not findings, and they stay internal. This stage also
   measures how long each model takes, so the full run gets a real budget.
4. **Freeze.** Finish the assumptions page and fix the code version. This is the
   main point where the project owner signs off.
5. **Run the paper's experiment once, in full,** at the paper's data and scale.
   If a model can't be trained at that scale within the paper's own stated
   hardware and time, run what fits in that budget and report the gap; that is a
   result in itself. Watch the first output so a bug can't waste days.
6. **Compare** our numbers with the published ones, model by model.
7. **If they're far apart, test the explanations** (next section).
8. **Write it up plainly.**

The rigor goes into steps 4 to 7. Steps 2 and 3 are for building and debugging
quickly, without ceremony.

## When the numbers don't match

A big gap first means looking for a mistake in our own work: the inputs, the
labels, the metric, the direction of the scores. More random seeds don't fix a
wrong setup.

If the gap survives that check, we build a statistical case that the published
result is out of reach for any reasonable version of the method. We can't test
every configuration, so:

- **List the reasonable variations in advance.** These are the gaps in the
  paper that could plausibly change the result, each with the alternatives a
  reasonable researcher might choose.
- **Run each variation a few times** with different random seeds. Measure
  uncertainty using independent units, such as customers rather than days from
  the same customer.
- **Test it like a hypothesis.** Start by assuming that at least one reasonable
  variation reaches the published number, within a set tolerance, under the
  paper's stated budget. If even the most optimistic upper limit across all the
  variations sits far below the published number, reject that assumption.
- **Look at the curves.** Track performance as data, training time and model size
  grow. If different routes all level off at about the same height, well below
  the published number, the gap is unlikely to be a matter of tuning. A curve
  that flattens hasn't necessarily hit a ceiling, so we state exactly what range
  was tested.

Think of standing on a peak and looking across a mountain range. We can't climb
every hill, but if every route we try levels off at the same height, a summit
far above it needs evidence, not hope.

Some checks need no training and can settle a question outright: numbers in a
table that can't all be true at once, or a stated workload that can't fit in the
stated time on the stated hardware.

Two more questions are worth asking along the way, because papers often lean on
them:

- **Is the task as hard as the paper implies?** Run a trivial rule and the
  paper's simplest baselines through the same evaluation. If they do about as
  well as the proposed method, the advantage claimed for it is in doubt.
- **Does the new component do the work credited to it?** Remove it, or destroy
  the structure it is supposed to exploit, and see whether the results notice.

## What we say and don't say

- Report matches as plainly as mismatches.
- Say exactly what was tested, on what data and with what choices. Don't stretch
  a result beyond that.
- Report numbers that are impossible as printed as impossible. How they came to
  be printed is not ours to claim: we never assert intent or fabrication.
- Disclose conflicts of interest. A maintainer of this project co-authored one of
  the papers, the water-network study.

## Who reads the website

Anyone. Write for a reader who has taken one introductory statistics class: plain
words, short sentences, and every technical term explained.

The homepage says what the project is and shows a card for each paper. Each
paper's page follows the same order:

1. What the paper claims.
2. Why we doubted it.
3. What we did.
4. What we found, matches and mismatches alike.
5. Limits: what this does and doesn't show.

Full records, code and earlier journals are linked underneath for anyone who
wants to check our work. Internal process, such as approvals, budgets, job
numbers and test counts, stays off the public pages.

## History

Until October 2026 this project ran a much heavier process, with a written
contract and sign-off for every step and long running status logs. It made
progress slow and the public pages hard to read, so it was retired. Its
documents are kept in `docs/archive/`, `docs/plans/` and `docs/decisions/` as
history, not instructions. The deep-autoencoder and water-network studies were
done under that process and will be redone with this approach.
