"""Estimators and the search spaces compared in cross-validation."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from loan_approval.config import MODEL_ORDER


def make_estimator(
    name: str, seed: int
) -> LogisticRegression | RandomForestClassifier | GradientBoostingClassifier:
    """Return an unfitted classifier. Tree estimators use one job; the search fans out."""

    if name == "logistic_regression":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=1,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown model '{name}'. Expected one of {MODEL_ORDER}.")


def search_space(name: str) -> dict[str, list[object]]:
    """Discrete grids small enough to document and large enough to move AUC.

    Logistic regression has no ``penalty`` argument here: in scikit-learn 1.8+
    that parameter is deprecated and ``lbfgs`` is L2. Gradient boosting has no
    ``class_weight``; imbalance is handled by the cost-based threshold instead.
    """

    if name == "logistic_regression":
        return {
            "clf__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        }
    if name == "random_forest":
        return {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [4, 8, 16, None],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__max_features": ["sqrt", 0.5],
            "clf__class_weight": [None, "balanced"],
        }
    if name == "gradient_boosting":
        return {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [2, 3],
            "clf__subsample": [0.8, 1.0],
            "clf__min_samples_leaf": [1, 5],
        }
    raise ValueError(f"Unknown model '{name}'. Expected one of {MODEL_ORDER}.")
