# Experiment specification and ambiguities

## What the paper explicitly reports

### Data

- SGCC: approximately 40,000 customers, daily readings over roughly three years, with benign and malicious labels.
- ISET/CER: approximately 3,000 residential meters, readings every 30 minutes for about 1.5 years, approximately 25,000 readings per customer.
- ISET malicious samples are generated from benign daily profiles using six functions:
  1. one random multiplicative factor in [0.1, 0.8] for the full profile;
  2. an independent multiplicative factor in [0.1, 0.8] per reading;
  3. replace a random 4-24 hour interval with zero;
  4. replace the daily profile by its mean;
  5. multiply the daily mean by independent factors in [0.1, 0.8];
  6. reverse the sequence in time.

### Splitting and preprocessing

- The paper says customers are separated between train and test partitions using a 2:1 ratio.
- Anomaly detectors are trained only on benign samples.
- The paper says both classes are normalized to zero mean and unit variance before describing the split.
- ADASYN is applied to the test set to balance classes.
- For supervised models, ADASYN is applied before the train/test split.

### Models

- FC-SAE encoder hidden widths: 400, 300, 200, 100; full mirrored decoder 100,
  200, 300, 400; both a distinct unspecified-width latent projection and a
  last-hidden bottleneck reading; Softmax output; dropout 0.4.
- LSTM-SAE encoder hidden widths: 500, 300; mirrored decoder 300, 500; both a
  distinct unspecified-width latent projection and the Algorithm-2 terminal
  state bottleneck; sigmoid output; dropout 0.2.
- FC-VAE encoder hidden widths: 500, 400, 300, 100; a distinct
  unspecified-width latent distribution; mirrored decoder 100, 300, 400, 500;
  Softmax output; dropout 0.4.
- LSTM-VAE encoder hidden widths: 400, 300; a distinct unspecified-width latent
  distribution; mirrored decoder 300, 400; sigmoid output; dropout 0.
- LSTM-AEA encoder hidden widths: 500, 300, 200; attention; mirrored decoder
  200, 300, 500; both a distinct post-attention latent projection and the
  Algorithm-5 attention-context bottleneck; sigmoid output; dropout 0.
- Hyperparameters are selected by sequential grid search over depth, width, optimizer, dropout, hidden activation, and output activation.

### Thresholds and metrics

- SAE/AEA anomaly score: reconstruction MSE.
- VAE anomaly score: described as reconstruction probability.
- Thresholds are said to be selected from ISET ROC curves using the "median of IQR" after dividing each curve into quartiles.
- Primary reported metrics: DR/TPR, FA/FPR, specificity, precision, balanced accuracy (called accuracy), F1, and ROC-AUC.

### Timing

- Full-ISET training time: 137-193 minutes depending on architecture.
- Reported online decision time: 1-2 seconds.
- No CPU, GPU, RAM, OS, library version, epoch count, batch size, timing protocol, or number of repetitions is reported.

## Provenance of the journal results in earlier papers

Two papers by the same authors contain the component experiments later assembled in the journal article:

- The 2021 ISSCS basic-autoencoder paper reports FC-BAE at TPR 0.81, FPR 0.15, AUC 0.81 and LSTM-BAE at 0.85, 0.13, 0.82. These are exactly the journal article's FC-SAE and LSTM-SAE ISET rows. The architectures and thresholds 0.58 and 0.61 also match.
- The 2020 EUSIPCO VAE paper reports FC-VAE at DR 88%, FA 11% and LSTM-VAE at DR 91%, FA 7%. These are exactly the journal article's ISET DR/FA entries. The architectures and thresholds 0.43 and 0.47 also match. The earlier paper reports `HD = DR - FA`, not ROC-AUC, so its third metric is not comparable to the later AUC entries.

This establishes result reuse/provenance; it does not establish that the numbers are false. It matters because the precursors describe the evaluation more explicitly:

- In the ISSCS paper, a portion of benign training data is reserved for validation and hyperparameter optimization, while the malicious dataset is stated to be used only for testing. The reported thresholds are then obtained from ROC curves, which require both benign and malicious labels. On the written protocol, that makes the labeled test data the only identified source for threshold selection.
- In the EUSIPCO paper, the benign holdout is concatenated with all generated malicious samples to form the test set, ADASYN oversamples the benign class within that test set, and the threshold is computed from the ROC curve. No separate labeled validation set is identified.
- Neither precursor reports hardware, epochs, batch size, seeds, repetitions, or uncertainty.

The journal article says its reported test set is different from the ROC validation set. That statement is difficult to reconcile with the precursor descriptions and the exact recurrence of their thresholds and point estimates. Possible explanations include an incompletely described split or reuse of test-selected results. The paper and available precursors do not distinguish these possibilities.

## Blocking ambiguities for exact reproduction

1. SGCC contains 1,034 daily features, but the architecture section states that the model input has 48 neurons. The paper does not explain how SGCC is transformed into 48-element sequences.
2. The source SGCC dates are not chronologically ordered; the paper does not say whether they were sorted.
3. Missing-value treatment is not described, despite 25.64% missing SGCC consumption cells.
4. It is unclear whether normalization is joint, per class, per customer, or per feature. The described order places normalization before splitting.
5. ROC curves require malicious validation samples, but anomaly-detector `X_TR` is defined as benign-only.
6. "Median of IQR of the ROC curve" is not a reproducible threshold-selection algorithm.
7. The VAE section gives contradictory threshold directions: low reconstruction probability is first called anomalous, but a later common rule declares values above threshold anomalous.
8. The FC and sequential models are not parameter-, depth-, dropout-, optimizer-, or output-activation-matched.
9. Epochs, batch sizes, early stopping, initialization, seeds, and number of runs are absent.
10. Table V reports attack-dependent false-alarm rates although false alarms depend only on benign samples when model, threshold, and benign test set are fixed.
11. Reported DR, FA, and precision values do not correspond to one common balanced test set.
12. The CER data and model input contain 48 half-hour values per day, but the attack equations and Fig. 2 use a 24-hour index. The paper does not say whether readings were aggregated or whether attack durations were converted to half-hour slots.
13. Equation (3) defines the bypass interval endpoint as `tf = ti - tl`; this puts the endpoint before the start. The intended operation appears to require addition.
14. The exact thresholds and point estimates recur from precursor papers whose written protocols place malicious examples only in the test set (ISSCS) or identify no separate labeled validation set (EUSIPCO). The journal's later validation/test distinction is therefore not independently reconstructable.
15. The journal says cross-validation is over `X_TR`, which it elsewhere defines as the benign anomaly-detector training subset, yet says hyperparameters improve DR on a validation set and uses ROC curves to set thresholds. DR and ROC cannot be computed from benign-only data; the labeled validation construction is not given.
16. Table IV describes full ISET `|X_TR|` as 60 million. That is plausibly a count of scalar meter readings, whereas a model input is a vector of 48 readings. The number of training examples, and therefore the reported time per example or epoch, cannot be recovered.

## Non-blocking internal inconsistencies (recorded for the report)

These do not change any implementation branch; the reproduction targets remain
the printed table cells. Registered 2026-07-21, before any Table III--V
execution.

1. The paper's headline improvement ranges disagree across sections: the
   abstract and conclusion state DR 4--21% and FA 4--13%; the contributions
   list states DR 4--8% and FA 4--7%; Section IV states 4--21% and 3.5--12%
   for SGCC and 3--21% and 3--13% for ISET.
2. SGCC malicious behavior is described as reporting "an energy consumption
   value of zero at specified hours," but SGCC has one reading per day. The
   description cannot be executed against the provided data; SGCC labels are
   taken as given and no attack synthesis is applied to SGCC.
3. The paper claims "around 3000 residential units" for ISET, but the official
   allocation table assigns 4,225 residential (Code = 1) meters, all present in
   the consumption archives. No subselection procedure is described; the
   paper-literal branch uses all official residential meters (A03) and records
   the count discrepancy (A27).

## Controlled replication protocol

The controlled experiment will not imitate known-invalid evaluation choices merely to obtain similar numbers. A separate `paper_literal` mode may later reproduce ambiguous choices explicitly, with warnings.

### Partitions

- Fixed customer-level train/validation/test partitions.
- Default ratios: 60%/20%/20%, stratified only where real labels exist.
- Identical held-out customers for all models in a comparison.

### Preprocessing

- Date columns parsed and chronologically sorted.
- Imputation fitted on training customers only.
- Scaling fitted on benign training samples only.
- No ADASYN or other synthesis in validation or test data.

### Model comparisons

1. Sanity baselines: daily/profile summary scores, PCA reconstruction, Isolation Forest, one-class SVM.
2. Capacity-controlled FC-AE and LSTM-AE.
3. Same LSTM-AE with and without attention.
4. Paper-like unmatched architectures as a separate diagnostic.

### Statistical protocol

- At least 10 training seeds for neural models when computationally feasible.
- Store every seed, split identifier, model parameter count, training curve, raw score, and prediction.
- Report mean and standard deviation across seeds.
- Bootstrap held-out customers, rather than treating correlated readings from one customer as independent.
- Primary comparison: ROC-AUC and DR at fixed validation-selected FA rates (1%, 5%, and 10%).
