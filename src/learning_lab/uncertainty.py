"""Finite-sample split conformal inference, calibration, and paired uncertainty."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, softmax
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score


def conformal_quantile(scores, alpha=0.1):
    """Order statistic ceil((n+1)(1-alpha)); infinity when calibration is too small."""
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or not len(s) or not np.isfinite(s).all() or not 0 < alpha < 1:
        raise ValueError("Need finite nonempty 1D scores and 0 < alpha < 1")
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return np.inf if k > len(s) else float(np.partition(s, k - 1)[k - 1])


def regression_interval(pred, calibration_pred, calibration_y, alpha=0.1):
    q = conformal_quantile(np.abs(calibration_y - calibration_pred), alpha)
    return np.asarray(pred) - q, np.asarray(pred) + q


def cqr_interval(lower, upper, cal_lower, cal_upper, cal_y, alpha=0.1):
    """Conformalized quantile regression; permit negative corrections as in CQR."""
    # Quantile crossing is repaired consistently for calibration and prediction.
    lo, hi = np.minimum(lower, upper), np.maximum(lower, upper)
    cl, cu = np.minimum(cal_lower, cal_upper), np.maximum(cal_lower, cal_upper)
    q = conformal_quantile(np.maximum(cl - cal_y, cal_y - cu), alpha)
    # A very negative correction can yield an empty interval; preserve it honestly.
    return lo - q, hi + q


def prediction_sets(probabilities, cal_probabilities, cal_y, alpha=0.1):
    p, cp = np.asarray(probabilities), np.asarray(cal_probabilities)
    q = conformal_quantile(1 - cp[np.arange(len(cal_y)), cal_y], alpha)
    return (1 - p) <= q


def fit_temperature(logits, y):
    # Float64 normalization avoids float32 row-sum warnings in sklearn log_loss.
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(y)

    def objective(log_t):
        p = softmax(z / np.exp(log_t), axis=1)
        return log_loss(y, p, labels=np.arange(z.shape[1]))

    return float(np.exp(minimize_scalar(objective, bounds=(-3, 3), method="bounded").x))


def binary_temperature(p, temperature):
    return expit(logit(np.clip(p, 1e-7, 1 - 1e-7)) / temperature)


def binary_metrics(y, p, threshold=0.5):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y, p >= threshold),
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "log_loss": log_loss(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": reliability(y, p)["ece"],
    }


def reliability(y, p, bins=15):
    ids = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        keep = ids == b
        if keep.any():
            confidence, frequency = float(np.mean(p[keep])), float(np.mean(y[keep]))
            n = int(keep.sum())
            rows.append({"probability": confidence, "frequency": frequency, "count": n})
            ece += n / len(y) * abs(confidence - frequency)
    return {"ece": ece, "bins": rows}


def paired_bootstrap(a, b, seed=0, repeats=1000):
    """CI for a mean paired loss difference a-b, conditional on fitted models."""
    delta = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
    return {
        "difference": float(delta.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "repeats": repeats,
        "unit": "test example",
    }
