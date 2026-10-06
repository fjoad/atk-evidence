# Electricity theft detection with poisoned training labels

*Robust Electricity Theft Detection Against Data Poisoning Attacks in Smart
Grids*, Takiddin, Ismail, Zafar and Serpedin, IEEE Transactions on Smart Grid
(2021). [Publication record](https://doi.org/10.1109/TSG.2020.3047864)

**In progress.**

## What the paper claims

Electricity theft detectors learn from examples labeled "normal" or "theft".
The paper studies an attacker who poisons that training data by labeling some
theft examples as normal. It tests seven standard detectors and reports that
all of them get worse as poisoning grows: at 30% poisoning, their detection
rates fall by about 15 percentage points.

The paper's main contribution is a new model that chains three networks: an
attention autoencoder, a recurrent GRU network and a final classifier. The
paper reports that this model barely notices the poisoning. Its detection rate
falls only from 95.2% to 92.2%.

## Why we doubted it

The results looked implausible to us, so we set out to rebuild the experiments
from the paper's description. Reading the paper closely turned up problems
before we wrote any code:

- **The training loss is wrong as printed.** Equation (1) uses log(p) in both of
  its terms, so the label drops out and the loss can't tell theft from normal
  use. The obvious fix is the standard formula, and that's what we use.
- **Three of the main model's result rows can't all be true at once.** No single
  set of test examples gives those detection, false-alarm, precision and
  accuracy numbers together, even allowing for rounding.
- **Key details are missing:** which 3,000 customers were used, exactly how the
  poisoning was applied, and how the three parts of the main model connect.

## What we did

We rebuilt the data preparation the paper describes: the Irish smart-meter data
it names, its six simulated theft patterns, its balancing step and its
poisoning. We then trained the paper's models on a small slice of the data, 20
customers with four weeks each, both without poisoning and with 30% poisoning.
The slice was meant to test our code before scaling up.

Six of the paper's nine models have run this way: random forest, AdaBoost, a
support vector machine (SVM), a feed-forward network, a GRU network and the main
model.

## What we found

- **Most of the simpler models work.** The random forest, AdaBoost, feed-forward
  and GRU models all learned to separate theft from normal use. Without
  poisoning, the feed-forward model matched the paper's detection rate at the
  paper's false-alarm rate. This went against our starting expectation, and we
  report it as plainly as the problems.
- **The SVM falls far short.** At the paper's false-alarm rate it detected 19% of
  attacks, against 89% in the paper, and no choice of cutoff closes the gap.
- **The main model didn't learn.** Built with the activation the paper's text
  specifies (ReLU), it gave every example the same score, which is no better
  than guessing. The paper contradicts itself here: its Algorithm 1 uses tanh
  instead. A tanh version learns a simple test task but hasn't been run on the
  real data yet.
- **Poisoning behaves differently from the paper's tables.** In the paper,
  poisoning lowers detection and raises false alarms by about the same amount
  for every model. On our slice, poisoning made the models raise fewer alarms of
  both kinds, and moving the cutoff recovered much of the lost detection.

## Limits

- Everything above comes from a small slice of the data, 20 customers against the
  paper's 3,000, and covers only 0% and 30% poisoning. The full experiment
  hasn't run yet.
- Three of the paper's nine models haven't run: ARIMA, the standalone attention
  autoencoder and ensemble averaging.
- Where the paper leaves gaps, we had to choose. A different reasonable choice
  could change some results, so every choice is written down.
- Numbers that can't all be true show the paper contains errors. They don't
  show how those numbers were produced, and we make no claim about that.

## Read more

- [Full research log](RESEARCH_LOG.md): every step, including our own mistakes
  and corrections.
- [The source problems in detail](SOURCE_AUDIT_FINDING.md).
- [Results and run records](results/) and [our code](reproduction/).
- [Running the preparation yourself](RUNNING.md).
