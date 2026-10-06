# ATK Evidence

We rebuild the experiments in published papers whose results we doubt, run
them, and compare our numbers with the published ones.

**[Read the studies](https://fjoad.github.io/atk-evidence/)** ·
[How we work](docs/APPROACH.md)

## Why

Many machine-learning papers don't publish their code, but each one describes
its method, and that description should be enough for another researcher to
rebuild it. When a paper's results look implausible, the fair test is to do
exactly that: build what the paper describes, fill any gaps the way a
reasonable researcher would, and see whether the published numbers come out.

We start from doubt, and we try to test it fairly. When our rebuild matches the
paper, we say so. When it doesn't, we look for mistakes on our side first, then
ask whether any reasonable version of the method could reach the published
numbers.

## Papers so far

### Robust Electricity Theft Detection Against Data Poisoning Attacks in Smart Grids

Takiddin et al., *IEEE Transactions on Smart Grid*, 2021. In progress.

- The training loss printed in the paper can't train a classifier: as written,
  it ignores the label.
- Three of the main model's result rows can't all be true at once. No single
  test set gives those numbers together.
- On a small slice of the data, most of the simpler comparison models work
  well, and one reaches the paper's detection level. The paper's main model,
  rebuilt one way, gave every example the same score. We're now testing another
  reading of the paper.

### Deep Autoencoder-Based Anomaly Detection of Electricity Theft Cyberattacks in Smart Grids

Takiddin et al., *IEEE Systems Journal*, 2022. First attempt, to be redone with
our current approach.

- Our rebuild of the paper's simplest model detected 25% of attacks. The paper
  reports 81%.
- With the output layer and error score the paper specifies, a calculation
  shows that no trained model could detect more than about 9% of attacks at the
  paper's false-alarm rate on our prepared data. A different output layer would
  loosen that limit.
- Given the paper's own reported training time, the paper's LSTM model detected
  17% of attacks. The paper reports 85%.

### Graph Transfer Learning-Based Attack Detection in Cyber-Physical Water Distribution Systems

Ahasan et al., *EUSIPCO*, 2025. First attempt, to be redone with our current
approach.

- We checked 67,326 possible ways of measuring each number in the published
  results table. None reproduces the table, and 20 of its 27 numbers can't be
  reached by any of them.
- In our own experiments, the graph and transfer-learning parts did help in
  some comparisons, and the results depended strongly on setup choices.

## Disclosure

Faaiz Joad, a maintainer of this project, is a co-author of the water-network
paper. That study is not independent of its authors, so its evidence, limits
and corrections are kept visible. We make no claims about anyone's intent.

## Check our work

Each paper's code, assumptions and run records are in [`studies/`](studies/).
Original datasets and paper PDFs are not in the repository; see
[Getting started](docs/GETTING_STARTED.md) for data access.

```bash
git clone https://github.com/fjoad/atk-evidence.git
cd atk-evidence
bash scripts/bootstrap.sh
bash scripts/test.sh
```

Contributors and AI agents: start with [AGENTS.md](AGENTS.md).
