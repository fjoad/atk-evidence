"""Detection + metrics for the TL-STGT paper.

Paper Section V-A: from residuals E=Y-Yhat (or reconstruction error), fit a benign
Gaussian (mean mu, covariance phi); the per-sample score is the SQUARED Mahalanobis
distance xi^2 = (e-mu)^T phi^-1 (e-mu); the paper then averages xi^2 over consecutive
batches of size S ("mean squared Mahalanobis distance for each batch") and flags a
batch when that mean exceeds a threshold tuned on the validation set. We apply the
S-window trailing mean per sample so the balanced per-sample metric is preserved.
"""
from __future__ import annotations
import numpy as np

BATCH_S = 32   # paper: S = the batch size (Adam batch = 32); documented.


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int); y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
    dr = tp / (tp + fn) if tp + fn else 0.0
    fa = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * dr / (prec + dr) if prec + dr else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn)
    return dict(F1=100 * f1, ACC=100 * acc, DR=100 * dr, FA=100 * fa)


def fit_benign(resid_benign):
    mu = resid_benign.mean(0)
    cov = np.cov(resid_benign, rowvar=False) + 1e-4 * np.eye(resid_benign.shape[1])
    return mu, np.linalg.pinv(cov)


def sq_mahalanobis(resid, mu, cov_inv):
    d = resid - mu
    return np.einsum("ij,jk,ik->i", d, cov_inv, d).clip(min=0)     # xi^2 (no sqrt)


def batch_mean(scores, s=BATCH_S):
    """Trailing mean of the squared distance over the last s samples (the paper's
    'mean squared Mahalanobis distance over consecutive batches')."""
    out = np.empty_like(scores, dtype=float)
    csum = np.concatenate([[0.0], np.cumsum(scores)])
    for i in range(len(scores)):
        lo = max(0, i - s + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i - lo + 1)
    return out


FA_TARGET = 5.0   # threshold set for ~5% false alarms on NORMAL validation data


def clean_batch_mask(labels, win_clean=None, s=BATCH_S):
    """True where the trailing batch of s samples is entirely attack-free.

    The paper thresholds a BATCH mean, so the reference distribution must come
    from batches that are wholly normal. Labelling by the current timestep alone
    is not enough: `batch_mean` averages the trailing s samples, so a normal
    sample just after an attack block inherits the attack's score. Optionally
    also requires each sample's history window to be attack-free (`win_clean`),
    matching how `data.make_datasets` defines `train_benign`.
    """
    ok = np.asarray(labels) == 0
    if win_clean is not None:
        ok = ok & np.asarray(win_clean).astype(bool)
    csum = np.concatenate([[0], np.cumsum(ok.astype(int))])
    out = np.zeros(len(ok), dtype=bool)
    for i in range(len(ok)):
        lo = max(0, i - s + 1)
        out[i] = (csum[i + 1] - csum[lo]) == (i - lo + 1)
    return out


def best_f1_threshold(scores, labels):
    """Threshold maximising F1 on the given (validation) scores.

    Ambiguity Axis 4: the paper says only that the threshold is "determined based
    on the model's performance using the validation set". Maximising F1 is the
    most literal reading of "performance" and the reading most favourable to the
    paper's headline numbers, so it must be swept rather than assumed away.
    """
    scores = np.asarray(scores, float); y = np.asarray(labels).astype(int)
    order = np.argsort(-scores)
    s, ys = scores[order], y[order]
    tp = np.cumsum(ys)                      # flagging the top k scores
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    k = int(np.argmax(f1))
    # threshold strictly between the k-th and (k+1)-th score so exactly k+1 flag
    return (s[k] + s[k + 1]) / 2.0 if k + 1 < len(s) else np.nextafter(s[k], -np.inf)


def evaluate_forecaster(resid_train_benign, resid_val, val_labels,
                        resid_test, test_labels, fa_target=FA_TARGET,
                        val_clean=None, batch_s=BATCH_S, thr_mode="fa5"):
    """Fit benign Gaussian; batch-mean squared MD; set threshold for a fixed FA on
    normal validation data (paper: 'threshold determined ... using the validation
    set'); evaluate on test. A fixed-FA operating point avoids the max-F1
    degenerate 'flag-everything' regime on a balanced set.

    The threshold is taken over wholly-normal BATCHES (see `clean_batch_mask`);
    selecting over every normal-labelled sample inflates it by orders of
    magnitude and drove the realized FA to 0.4% against a 5% target.
    """
    mu, cov_inv = fit_benign(resid_train_benign)
    s_val = batch_mean(sq_mahalanobis(resid_val, mu, cov_inv), s=batch_s)
    s_test = batch_mean(sq_mahalanobis(resid_test, mu, cov_inv), s=batch_s)
    if thr_mode == "maxf1":
        thr = best_f1_threshold(s_val, val_labels)
    else:
        ref = clean_batch_mask(val_labels, val_clean, s=batch_s)
        if not ref.any():                  # degenerate split: fall back, but say so
            ref = np.asarray(val_labels) == 0
        thr = np.percentile(s_val[ref], 100.0 - fa_target) if ref.any() else np.inf
    return _metrics(test_labels, s_test > thr)


def evaluate_classifier(y_true, y_pred):
    return _metrics(y_true, y_pred)
