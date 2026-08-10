# Paper 1 workflow — one-page source reconstruction

This is the short human view of the paper. It shows the order actually printed;
it does not silently repair the method.

```text
1. DATA
   SGCC: labeled benign/malicious customer histories
   ISET: benign readings every 30 min -> one 48-value customer-day
                                      |
2. ISET ATTACKS                      |
   Apply Eqs. 1-6 to every customer's complete benign matrix
   -> six malicious matrices per customer
                                      |
3. PREPARATION                       v
   merge customers -> normalize B and M before any split
                                      |
          +---------------------------+--------------------------+
          |                                                      |
4A. ANOMALY PATH                                      4B. SUPERVISED PATH
    split benign B 2:1 -> B1/B2                           B+M for all customers
    train only on B1                                     -> ADASYN
    test = B2 + all M                                    -> split 2:1
    -> ADASYN on the test set                            -> normalize
          |                                                      |
          +---------------------------+--------------------------+
                                      |
5. MODEL/SELECTION                    v
   ISET cross-validation -> sequential hyperparameter search
   -> ROC/IQR threshold -> Table-I settings -> train offline
                                      |
6. EVALUATION                         v
   SAE/AEA: reconstruction MSE > threshold -> malicious
   VAE prose: reconstruction probability < threshold -> malicious
   -> DR, FA, SP, PR, balanced ACC, F1, AUC
                                      |
7. REPORTED OUTPUTS                  v
   Table II: SGCC metrics             Table III: ISET metrics
   Table IV: ISET size/time/ACC       Table V: ISET attack-by-attack DR/FA
```

## Where the flow breaks

- **NON-EXECUTABLE:** Attack 3 prints `t_f = t_i - t_l`, so the final time is
  earlier than the initial time.
- **CONTRADICTORY:** the paper says test customers are unseen, then constructs
  `M` from every customer and tests on `B2 + M`.
- **CONTRADICTORY:** VAE prose says low probability is anomalous; the generic
  decision paragraph says probability above the threshold is anomalous.
- **NON-EXECUTABLE:** hyperparameter search records DR/FA using anomaly-model
  `X_TR`, but `X_TR` is defined as benign-only.
- **CONTRADICTORY:** the Table-I layer counts cannot be produced by the printed
  layer-count search range as written.
- **AMBIGUOUS:** one fixed Table-V model, threshold, and benign population must
  have one FA, while the six reported FAs require unstated retraining,
  resampling, or a different benign population.
- **CONTRADICTORY:** Tables II and III do not have one common class prevalence
  consistent with their reported DR/FA/PR rows and the paper's formulas.

## How the audit handles this

- `P`: retain each executable printed operation, even when statistically poor.
- `I`: predeclare a finite reasonable completion for omissions and each
  contradiction; the first executable route is `P0-ISET-FCSAE`.
- `C`: run scientifically corrected controls only after the paper-consistent
  evidence is complete.

The complete, source-located specification is [`METHOD.md`](METHOD.md). The
optional standalone visual page is
[`../../site/papers/atk-2022-deep-autoencoder/index.html`](../../site/papers/atk-2022-deep-autoencoder/index.html).
