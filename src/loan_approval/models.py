"""Estimators and the search spaces compared in cross-validation."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from loan_approval.config import MODEL_ORDER


def make_estimator(name: str, seed: int) -> LogisticRegression | HistGradientBoostingClassifier:
    """Return an unfitted classifier.

    Histogram gradient boosting is scikit-learn's histogram booster. LightGBM
    would add a binary wheel for the same algorithm family, which this project
    keeps off the Community Cloud install. Early stopping is off so ``max_iter``
    is the search's tree budget rather than a hidden validation split.
    """

    if name == "logistic_regression":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"Unknown model '{name}'. Expected one of {MODEL_ORDER}.")


def search_space(name: str) -> dict[str, list[object]]:
    """Discrete grids small enough to document and large enough to move AUC.

    Logistic regression has no ``penalty`` argument here: in scikit-learn 1.8+
    that parameter is deprecated and ``lbfgs`` is L2. The booster's class
    weight is searched because it can change the ranking, not because the raw
    score is treated as a probability. Calibration is a separate step.
    """

    if name == "logistic_regression":
        return {
            "clf__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        }
    if name == "hist_gradient_boosting":
        return {
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_iter": [150, 300],
            "clf__max_depth": [3, 6],
            "clf__min_samples_leaf": [10, 20],
            "clf__l2_regularization": [0.0, 1.0],
            "clf__class_weight": [None, "balanced"],
        }
    raise ValueError(f"Unknown model '{name}'. Expected one of {MODEL_ORDER}.")
