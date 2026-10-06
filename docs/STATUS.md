# Status

_Last updated 2026-10-06. Rewrite this page whenever the state changes; don't
append history to it._

## Right now: overhaul done, waiting to publish

The old process has been replaced. [APPROACH.md](APPROACH.md) holds the thinking,
[AGENTS.md](../AGENTS.md) is one page, and this page and
[CONTEXT.md](CONTEXT.md) are short. Old documents are in `docs/archive/`.

The README, each study's README and the website are rewritten in plain
language. The website has a homepage with a card per paper and a summary page
per paper, rendered from the study README. Each full research log moved to
`papers/<id>/journal/`, and old section links forward there. All tests pass.
Nothing has been pushed yet: publishing needs the owner's approval.

Next: research on the poisoning paper, below.

## Papers

### Robust Electricity Theft Detection Against Data Poisoning Attacks (Takiddin et al., 2021): active

What we have so far, all on a 20-customer slice of the data at 0% and 30%
poisoning:
- Random forest, AdaBoost, feed-forward and GRU models rank attacks well. The
  feed-forward model reaches the paper's unpoisoned detection level.
- The SVM is far below the paper.
- The main sequential model, built with ReLU recurrent cells, gave every example
  the same score. A version with tanh cells learns a synthetic test task but
  hasn't been run on real data.
- Free checks done: the printed training loss ignores the label, and three of the
  main model's result rows can't all be true at once.

Next, in order:
1. **Free checks not yet done.**
   - Could the stated RTX 2070 training times cover the full data?
   - Detection plus false alarms is about 100 in every printed row. That could
     come from how the cutoffs were chosen; test it on our saved scores.
2. **Write `ASSUMPTIONS.md`.** Make reasonable-person choices for all nine models.
   The paper conflicts on tanh versus ReLU, so try both in the small run.
3. **Build what's missing:** ARIMA, ensemble averaging, a trained standalone
   attention autoencoder, and the 10% and 20% poisoning levels.
4. **Small run of everything:** about 200 customers, all four poisoning levels,
   short training. Debug.
5. **Freeze, get the owner's sign-off, then the full run.**

### Deep Autoencoder-Based Anomaly Detection (Takiddin et al., 2022): waiting

First attempt finished under the old process; its results are summarized in the
[README](../README.md). It will be redone with the current approach.

### Graph Transfer Learning-Based Attack Detection in Water Distribution Systems (Ahasan et al., 2025): waiting

First attempt finished under the old process. It will be redone with the current
approach. A maintainer co-authored this paper.

## Open question

Build jobs run by Astra through the Codex companion have no network access, so
they can't reach Panther. Either allow network access for those jobs, or the
planning agent submits the cluster jobs.
