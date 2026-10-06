# Electricity theft detection with deep autoencoders

*Deep Autoencoder-Based Anomaly Detection of Electricity Theft Cyberattacks in
Smart Grids*, Takiddin, Ismail, Zafar and Serpedin, IEEE Systems Journal
(2022). [Publication record](https://doi.org/10.1109/JSYST.2021.3136683)

**First attempt.** This study was done under our earlier, heavier process. We
will redo it with our current approach; the results below stand until then.

## What the paper claims

The paper detects electricity theft with autoencoders: networks trained on
normal electricity use to reproduce their input. A day that the network
reproduces badly is flagged as possible theft. The paper compares a simple
fully connected version with recurrent (LSTM) and attention versions on two
public datasets and reports high detection rates. On the Irish smart-meter data,
the simplest model (FC-SAE) detects 81% of attacks at 15% false alarms, and the
LSTM version (LSTM-SAE) detects 85% at 13%.

## Why we doubted it

The reported performance looked implausible to us, so we rebuilt the method
from the paper and ran it on the same Irish data.

## What we did

- We rebuilt the simplest model, FC-SAE, with the data preparation the paper
  describes, and trained it once.
- We asked a stronger question using math rather than more training. With the
  output layer (Softmax) and error score (mean squared error) the paper
  specifies, what is the best any trained model could possibly do on these
  inputs?
- We gave the LSTM version exactly the training time the paper reports, 183
  minutes, on one GPU, and scored the result.

## What we found

| | Paper | Our rebuild |
|---|---:|---:|
| FC-SAE: attacks detected | 81.00% | 25.48% |
| FC-SAE: false alarms | 15.00% | 45.13% |
| LSTM-SAE after 183 minutes: attacks detected | 85.00% | 16.62% |
| LSTM-SAE after 183 minutes: false alarms | 13.00% | 31.91% |

- **The paper's own setup rules the number out.** With Softmax outputs and mean
  squared error on our prepared data, no trained model could detect more than
  9.25% of attacks at 15% false alarms, or reach more than 50.93% balanced
  accuracy. The paper reports 81% and 83%.
- **A different output layer loosens that limit but didn't help in practice.**
  With a Sigmoid output instead, the limit no longer applies. A small Sigmoid
  model we trained still detected only 9.75% of attacks at that false-alarm rate.
- **The models weren't doing nothing.** FC-SAE carried a little useful signal. A
  small contribution can coexist with a large gap.

## Limits

- One training run per model, covering two of the paper's models.
- The 9.25% limit holds for the paper's stated output layer and score on our
  prepared data. Other ways of preparing the data, outputs or scores are outside
  it.
- The LSTM's training loss was still falling when its 183 minutes ran out.
  Longer training wasn't tested. The small Sigmoid model trained for ten epochs
  and was also still improving.
- None of this shows how the published numbers were produced, and we make no
  claim about that.

## Read more

- [Full research log](RESEARCH_LOG.md).
- [Detailed technical report](../../site/papers/atk-2022-deep-autoencoder/reproduction/index.html),
  with every comparison and the calculation behind the limit.
- [Earlier notes](../../site/papers/atk-2022-deep-autoencoder/earlier-notes.html)
  and [all findings and run records](results/).
- [How the rebuild was specified](CLEAN_READER_SPECIFICATION.md).
