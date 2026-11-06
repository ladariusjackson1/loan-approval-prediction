"""Turn a ranking score into a probability without touching the holdout set.

The map is isotonic regression. Each out-of-fold score is transformed by a map
fit on the other folds, and that cross-fit vector is what the cost cutoff sees.
The saved map is then refit on every out-of-fold score and applied to the
fully refit model. The holdout rows are not in either fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


def constant_brier(y_true: np.ndarray | pd.Series) -> float:
    """Brier score of predicting the sample's own fully-paid rate for every row."""

    actual = np.asarray(y_true, dtype=float)
    if actual.size == 0:
        return float("nan")
    rate = float(np.mean(actual))
    return float(np.mean((actual - rate) ** 2))


def cross_fit_isotonic(
    scores: np.ndarray,
    y_true: np.ndarray | pd.Series,
    cv: StratifiedKFold,
) -> tuple[IsotonicRegression, np.ndarray]:
    """Return the full-sample isotonic map and cross-fit calibrated scores."""

    raw = np.asarray(scores, dtype=float)
    actual = np.asarray(y_true).astype(int)
    if raw.shape[0] != actual.shape[0]:
        raise ValueError("scores and y_true must have the same length.")
    calibrated = np.empty(actual.shape[0], dtype=float)
    placeholder = np.zeros((actual.shape[0], 1))
    for train_index, test_index in cv.split(placeholder, actual):
        fold_map = _fit_isotonic(raw[train_index], actual[train_index])
        calibrated[test_index] = _predict_map(fold_map, raw[test_index], actual[train_index])
    full_map = _fit_isotonic(raw, actual)
    return full_map, calibrated


def apply_scores(pipeline: Any, frame: pd.DataFrame, calibrator: Any | None) -> np.ndarray:
    """P(fully paid) for each row. Applies the isotonic map when one was saved."""

    raw = np.asarray(pipeline.predict_proba(frame)[:, 1], dtype=float)
    if calibrator is None:
        return raw
    mapped = np.asarray(calibrator.predict(raw), dtype=float)
    return np.clip(mapped, 0.0, 1.0)


def _fit_isotonic(scores: np.ndarray, actual: np.ndarray) -> IsotonicRegression | None:
    if np.unique(actual).size < 2:
        return None
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(scores, actual)
    return model


def _predict_map(
    model: IsotonicRegression | None, scores: np.ndarray, fallback_from: np.ndarray
) -> np.ndarray:
    if model is None:
        rate = float(np.mean(fallback_from)) if fallback_from.size else 0.5
        return np.full(scores.shape[0], rate, dtype=float)
    return np.clip(np.asarray(model.predict(scores), dtype=float), 0.0, 1.0)
