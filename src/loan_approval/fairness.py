"""Outcome gaps across attributes the deployed model is not supposed to use directly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from loan_approval.config import FAIRNESS_REFERENCE


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def binary_rates(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> dict[str, float | int | None]:
    """Approval rate, base rate, TPR, and FPR for one group."""

    actual = np.asarray(y_true).astype(int)
    predicted = np.asarray(y_pred).astype(int)
    positives = actual == 1
    negatives = ~positives
    true_pos = int(np.sum((predicted == 1) & positives))
    false_neg = int(np.sum((predicted == 0) & positives))
    false_pos = int(np.sum((predicted == 1) & negatives))
    true_neg = int(np.sum((predicted == 0) & negatives))
    n = int(len(actual))
    return {
        "n": n,
        "n_positive": int(positives.sum()),
        "n_negative": int(negatives.sum()),
        "base_rate": _rate(int(positives.sum()), n),
        "approval_rate": _rate(int(np.sum(predicted == 1)), n),
        "tpr": _rate(true_pos, true_pos + false_neg),
        "fpr": _rate(false_pos, false_pos + true_neg),
        "precision": _rate(true_pos, true_pos + false_pos),
    }


def fairness_table(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive: pd.DataFrame,
) -> pd.DataFrame:
    """One row per attribute and group, indexed like ``sensitive``."""

    actual = pd.Series(np.asarray(y_true).astype(int), index=sensitive.index)
    predicted = pd.Series(np.asarray(y_pred).astype(int), index=sensitive.index)
    rows: list[dict[str, object]] = []
    for attribute in sensitive.columns:
        grouped = sensitive.groupby(sensitive[attribute], dropna=False).groups
        for group, indexes in grouped.items():
            rates = binary_rates(actual.loc[indexes], predicted.loc[indexes])
            rows.append({"attribute": attribute, "group": str(group), **rates})
    table = pd.DataFrame(rows)
    return _with_reference_gaps(table)


def _with_reference_gaps(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    frames: list[pd.DataFrame] = []
    for attribute, block in table.groupby("attribute", sort=False):
        reference_name = _reference_group(block, str(attribute))
        reference = block.loc[block["group"] == reference_name].iloc[0]
        enriched = block.copy()
        enriched["reference_group"] = reference_name
        enriched["demographic_parity_gap"] = enriched["approval_rate"].map(
            lambda value, ref=reference["approval_rate"]: _gap(value, ref)
        )
        enriched["equal_opportunity_gap"] = enriched["tpr"].map(
            lambda value, ref=reference["tpr"]: _gap(value, ref)
        )
        enriched["disparate_impact"] = enriched["approval_rate"].map(
            lambda value, ref=reference["approval_rate"]: _ratio(value, ref)
        )
        enriched["outside_four_fifths"] = enriched["disparate_impact"].map(_outside_four_fifths)
        frames.append(enriched)
    return pd.concat(frames, ignore_index=True)


def _reference_group(block: pd.DataFrame, attribute: str) -> str:
    preferred = FAIRNESS_REFERENCE.get(attribute)
    groups = set(block["group"].astype(str))
    if preferred is not None and preferred in groups:
        return preferred
    return str(block.sort_values("n", ascending=False).iloc[0]["group"])


def _gap(value: object, reference: object) -> float | None:
    if value is None or reference is None or pd.isna(value) or pd.isna(reference):
        return None
    return float(value) - float(reference)


def _ratio(value: object, reference: object) -> float | None:
    if value is None or reference is None or pd.isna(value) or pd.isna(reference):
        return None
    denominator = float(reference)
    if denominator == 0:
        return None
    return float(value) / denominator


def _outside_four_fifths(ratio: object) -> bool | None:
    """An 80% screening ratio. Not a legal conclusion, and not a protected-class finding.

    This extract has no sex, age, or race fields. The ratio is reported for loan
    purpose, FICO band, and the old credit-policy flag only.
    """

    if ratio is None or pd.isna(ratio):
        return None
    number = float(ratio)
    return bool(number < 0.8 or number > 1.25)
