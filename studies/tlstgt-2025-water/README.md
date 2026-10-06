# Attack detection in water networks

*Graph Transfer Learning-Based Attack Detection in Cyber-Physical Water
Distribution Systems*, Ahasan, Joad, Atat, Thompson, Serpedin and Takiddin,
EUSIPCO (2025).

**Disclosure:** Faaiz Joad, a maintainer of this project, is a co-author of this
paper. This study is not independent of its authors.

**First attempt.** This study was done under our earlier, heavier process. We
will redo it with our current approach; the results below stand until then.

## What the paper claims

The paper detects attacks on water-system sensors. Its model combines the
layout of the water network (a graph), patterns over time, and transfer
learning from smaller to larger networks. The paper reports that this model
beats ordinary machine-learning methods and improves as the network grows.

## Why we doubted it

The reported numbers looked implausible to us. We checked whether the published
results table is consistent with itself, then rebuilt the method and its
comparisons.

## What we did

- **Checked the table against itself.** On a test set with equal numbers of
  attacks and normal readings, accuracy can't be higher than the average of the
  detection rate and 100%.
- **Searched for a way to explain the table.** We tried 67,326 possible ways of
  measuring each number, covering test sizes, attack proportions and metric
  definitions.
- **Rebuilt the models.** This included simple rules with no training, run
  through the same evaluation. We also tested whether the graph and transfer
  learning actually help.

## What we found

- **The table doesn't add up.** 24 of its 27 accuracy values are higher than a
  balanced test set allows. None of the 67,326 measurement setups reproduces the
  whole table, and 20 of its 27 numbers can't be reached by any of them.
- **Simple rules catch a lot.** Two rules with no training, one for large changes
  and one for stuck sensors, caught many attacks. Replay attacks stayed hard for
  every method we tried.
- **The graph does help.** Against our starting expectation, scrambling or
  removing the network's connections lowered performance. Transfer learning also
  helped in our runs, though it also gave the model extra training.
- **Setup choices matter a lot.** Training on normal data only, instead of the
  paper's 50/50 split, changed scores by about 25 F1 points.
- **We made mistakes along the way.** A threshold bug in our own code briefly
  made the gap look far larger than it was. That claim was retracted, and the
  corrections are kept in the record.

## Limits

- The search covered many measurement setups but not every conceivable one.
- Our data is the standard public version of this water-network simulation, not
  the authors' unpublished run.
- Some early results ran on a local machine rather than the cluster and are
  marked provisional.
- The inconsistencies show the table contains errors. They don't show how it was
  produced, and we make no claim about that.

## Read more

- [Full research log](RESEARCH_LOG.md).
- [Evidence log with every finding, correction and retraction](EVIDENCE.md).
- [Earlier notes](../../site/papers/tlstgt-2025-water/earlier-notes.html) and
  [the earlier report (PDF)](../../site/reports/tlstgt-2025-water.pdf).
- [Data sources](DATA.md) and [running the code](RUNNING.md).
