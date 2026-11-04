"""Global explanations: permutation importance and logistic coefficients."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance

from loan_approval.config import CATEGORICAL_FEATURES, FEATURE_LABELS, VALUE_LABELS


def permutation_frame(
    pipeline: Any,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    repeats: int,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Mean drop in ROC-AUC when each original column is shuffled on the holdout set."""

    result = permutation_importance(
        pipeline,
        features,
        target,
        scoring="roc_auc",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=n_jobs,
    )
    frame = pd.DataFrame(
        {
            "feature": list(features.columns),
            "label": [FEATURE_LABELS.get(column, column) for column in features.columns],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def coefficient_frame(pipeline: Any) -> pd.DataFrame:
    """Signed coefficients from a fitted logistic pipeline, in original-feature language."""

    steps = getattr(pipeline, "named_steps", None)
    if not steps or "clf" not in steps:
        raise TypeError("Coefficient extraction requires a linear classifier.")
    classifier = steps["clf"]
    if not hasattr(classifier, "coef_"):
        raise TypeError("Coefficient extraction requires a linear classifier.")
    names = [
        pretty_transformed_name(name)
        for name in pipeline.named_steps["preprocess"].get_feature_names_out()
    ]
    frame = pd.DataFrame({"feature": names, "coefficient": classifier.coef_.ravel()})
    frame["abs_coefficient"] = frame["coefficient"].abs()
    return frame.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def pretty_transformed_name(name: str) -> str:
    """Turn ``categorical__checking_status_<0`` into a readable label."""

    if name.startswith("numeric__"):
        raw = name.removeprefix("numeric__")
        return FEATURE_LABELS.get(raw, raw)
    if name.startswith("categorical__"):
        raw = name.removeprefix("categorical__")
        for feature in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            prefix = f"{feature}_"
            if raw.startswith(prefix):
                value = raw[len(prefix) :]
                label = FEATURE_LABELS.get(feature, feature)
                pretty_value = VALUE_LABELS.get(feature, {}).get(value, value)
                return f"{label} = {pretty_value}"
        return raw
    return name
