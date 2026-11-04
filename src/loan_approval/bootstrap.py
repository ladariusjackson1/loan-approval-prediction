"""Percentile bootstrap intervals for ROC-AUC and for a paired difference.

The interval is the percentile method on resampled rows. It is not a Bayesian
credible interval and it is not a cross-validation standard deviation. A
resample that lands on one class is dropped, because ROC-AUC is undefined there.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_roc_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float | int | None]:
    """Point ROC-AUC and a percentile interval from ``n_boot`` row resamples."""

    actual = np.asarray(y_true).astype(int)
    predicted = np.asarray(scores, dtype=float)
    estimate = float(roc_auc_score(actual, predicted))
    estimates, valid = _resample_scores(actual, [predicted], n_boot=n_boot, seed=seed)
    return _pack(estimate, estimates[:, 0], valid, n_boot=n_boot, alpha=alpha)


def paired_auc_difference(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float | int | bool | None]:
    """ROC-AUC of ``scores_a`` minus ROC-AUC of ``scores_b``, on shared resamples.

    ``includes_zero`` is true when the interval covers zero, false when it does
    not, and null when too many resamples were single-class to quote an interval.
    """

    actual = np.asarray(y_true).astype(int)
    left = np.asarray(scores_a, dtype=float)
    right = np.asarray(scores_b, dtype=float)
    estimate = float(roc_auc_score(actual, left) - roc_auc_score(actual, right))
    estimates, valid = _resample_scores(actual, [left, right], n_boot=n_boot, seed=seed)
    packed = _pack(estimate, estimates[:, 0] - estimates[:, 1], valid, n_boot=n_boot, alpha=alpha)
    low = packed["ci_low"]
    high = packed["ci_high"]
    includes: bool | None = None
    if low is not None and high is not None:
        includes = bool(float(low) <= 0.0 <= float(high))
    packed["includes_zero"] = includes
    return packed


def _resample_scores(
    actual: np.ndarray,
    score_columns: list[np.ndarray],
    *,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(actual.shape[0])
    estimates = np.zeros((n_boot, len(score_columns)), dtype=float)
    valid = np.zeros(n_boot, dtype=bool)
    for draw in range(n_boot):
        index = rng.integers(0, n, size=n)
        sample = actual[index]
        if np.unique(sample).size < 2:
            continue
        for column, scores in enumerate(score_columns):
            estimates[draw, column] = roc_auc_score(sample, scores[index])
        valid[draw] = True
    return estimates, valid


def _pack(
    estimate: float,
    draws: np.ndarray,
    valid: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
) -> dict[str, float | int | None]:
    clean = draws[valid]
    n_valid = int(clean.size)
    required = max(30, int(0.8 * n_boot))
    low: float | None = None
    high: float | None = None
    if n_valid >= required:
        low_q, high_q = np.quantile(clean, [alpha / 2, 1 - alpha / 2])
        low = float(low_q)
        high = float(high_q)
    return {
        "estimate": float(estimate),
        "ci_low": low,
        "ci_high": high,
        "n_boot": int(n_boot),
        "n_boot_valid": n_valid,
    }
