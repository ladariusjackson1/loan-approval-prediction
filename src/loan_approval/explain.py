"""Turn one application into a decision and a plain-language reason list."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from loan_approval.config import (
    CATEGORY_OPTIONS,
    FEATURE_LABELS,
    NUMERIC_FEATURES,
    REVIEW_BAND,
    VALUE_LABELS,
)


class ApplicationError(ValueError):
    """The payload cannot be scored."""


def reference_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Median for numbers and the most common category, fit on training rows only."""

    profile: dict[str, Any] = {}
    for column in frame.columns:
        series = frame[column]
        if column in NUMERIC_FEATURES or pd.api.types.is_numeric_dtype(series):
            profile[column] = float(np.nanmedian(series.to_numpy(dtype=float)))
        else:
            mode = series.dropna().astype(str).mode()
            profile[column] = None if mode.empty else str(mode.iloc[0])
    return profile


def training_ranges(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Observed min and max, used to warn when an application leaves the training support."""

    ranges: dict[str, dict[str, float]] = {}
    for column in frame.columns:
        if column in NUMERIC_FEATURES or pd.api.types.is_numeric_dtype(frame[column]):
            values = frame[column].to_numpy(dtype=float)
            ranges[column] = {"min": float(np.nanmin(values)), "max": float(np.nanmax(values))}
    return ranges


def frame_from_application(payload: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """Build the single row the pipeline expects, in training column order."""

    missing = [column for column in feature_columns if column not in payload]
    if missing:
        raise ApplicationError(f"Missing fields: {', '.join(missing)}")
    row: dict[str, Any] = {}
    for column in feature_columns:
        value = payload[column]
        if _is_numeric_field(column, value):
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ApplicationError(
                    f"{FEATURE_LABELS.get(column, column)} must be numeric."
                ) from exc
            if not math.isfinite(number):
                raise ApplicationError(
                    f"{FEATURE_LABELS.get(column, column)} must be a finite number."
                )
            row[column] = number
        else:
            if value is None or str(value).strip() == "":
                raise ApplicationError(f"{FEATURE_LABELS.get(column, column)} is required.")
            row[column] = str(value)
    return pd.DataFrame([row], columns=feature_columns)


def local_contributions(
    pipeline: Any,
    application: pd.DataFrame,
    reference: dict[str, Any],
) -> list[dict[str, Any]]:
    """Change in P(good credit) when each field is replaced by the training reference.

    A positive delta means the applicant's actual value raises the probability
    relative to a typical training file. This is an ablation, not a causal effect
    and not a Shapley value: correlated fields share credit.
    """

    full = float(pipeline.predict_proba(application)[0, 1])
    records: list[dict[str, Any]] = []
    for column in application.columns:
        alternate = application.copy()
        if column in NUMERIC_FEATURES or pd.api.types.is_numeric_dtype(alternate[column]):
            alternate[column] = alternate[column].astype(float)
            alternate.loc[alternate.index[0], column] = float(reference[column])
        else:
            alternate[column] = alternate[column].astype(object)
            alternate.loc[alternate.index[0], column] = (
                None if reference[column] is None else str(reference[column])
            )
        swapped = float(pipeline.predict_proba(alternate)[0, 1])
        actual = application.iloc[0][column]
        records.append(
            {
                "feature": column,
                "label": FEATURE_LABELS.get(column, column),
                "actual": _jsonable(actual),
                "reference": _jsonable(reference[column]),
                "actual_text": describe_value(column, actual),
                "reference_text": describe_value(column, reference[column]),
                "delta": full - swapped,
            }
        )
    records.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return records


def explain_application(
    pipeline: Any,
    payload: dict[str, Any],
    *,
    reference: dict[str, Any],
    feature_columns: list[str],
    threshold: float,
    ranges: dict[str, dict[str, float]] | None = None,
    review_band: float = REVIEW_BAND,
) -> dict[str, Any]:
    """Score one application and attach the decision, referral flag, and reasons."""

    application = frame_from_application(payload, feature_columns)
    probability = float(pipeline.predict_proba(application)[0, 1])
    decision = "Approve" if probability >= threshold else "Decline"
    contributions = local_contributions(pipeline, application, reference)
    out_of_range = _outside_training(application, ranges or {})
    return {
        "decision": decision,
        "probability_good": probability,
        "threshold": float(threshold),
        "margin": probability - float(threshold),
        "refer_to_analyst": abs(probability - float(threshold)) < review_band,
        "out_of_range": out_of_range,
        "contributions": contributions,
        "reasons_for": _reasons(contributions, positive=True),
        "reasons_against": _reasons(contributions, positive=False),
        "summary": decision_summary(decision, probability, threshold),
    }


def decision_summary(decision: str, probability: float, threshold: float) -> str:
    """One sentence a reviewer can read before the reason list."""

    gap = probability - threshold
    direction = "above" if gap >= 0 else "below"
    return (
        f"{decision}. Estimated probability of good credit is {probability:.1%}, "
        f"which is {abs(gap):.1%} {direction} the operating cutoff of {threshold:.1%}."
    )


def _is_numeric_field(column: str, value: object) -> bool:
    """Known credit numerics, plus raw numbers for columns outside the category contract."""

    if column in NUMERIC_FEATURES:
        return True
    if column in CATEGORY_OPTIONS or isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def describe_value(feature: str, value: object) -> str:
    """Human label for a raw category code or a numeric field."""

    if (
        feature in NUMERIC_FEATURES
        or isinstance(value, (int, float, np.floating))
        and feature not in VALUE_LABELS
    ):
        return _format_number(feature, float(value))
    text = str(value)
    return VALUE_LABELS.get(feature, {}).get(text, text)


def reason_sentence(record: dict[str, Any]) -> str:
    """Describe one ablation in business language."""

    delta = float(record["delta"])
    if abs(delta) < 0.005:
        movement = "barely moves the score"
    elif delta > 0:
        movement = f"raises estimated P(good credit) by {delta:.1%}"
    else:
        movement = f"lowers estimated P(good credit) by {abs(delta):.1%}"
    return (
        f"{record['label']}: {record['actual_text']} rather than the typical "
        f"profile ({record['reference_text']}) {movement}."
    )


def _reasons(
    contributions: list[dict[str, Any]], *, positive: bool, limit: int = 3
) -> list[dict[str, Any]]:
    chosen = [
        record
        for record in contributions
        if (float(record["delta"]) > 0.005 if positive else float(record["delta"]) < -0.005)
    ]
    ordered = sorted(chosen, key=lambda record: float(record["delta"]), reverse=positive)
    reasons = []
    for record in ordered[:limit]:
        reasons.append(
            {
                "feature": record["feature"],
                "label": record["label"],
                "delta": float(record["delta"]),
                "text": reason_sentence(record),
            }
        )
    return reasons


def _format_number(feature: str, value: float) -> str:
    if feature == "credit_amount":
        return f"{value:,.0f} DM"
    if feature == "duration":
        return f"{value:.0f} months"
    if feature == "age":
        return f"{value:.0f} years"
    if feature in {
        "installment_commitment",
        "residence_since",
        "existing_credits",
        "num_dependents",
    }:
        rounded = int(round(value))
        return str(rounded)
    return f"{value:.2f}"


def _outside_training(application: pd.DataFrame, ranges: dict[str, dict[str, float]]) -> list[str]:
    flags: list[str] = []
    row = application.iloc[0]
    for column, bounds in ranges.items():
        if column not in application.columns:
            continue
        value = float(row[column])
        if value < bounds["min"] or value > bounds["max"]:
            flags.append(column)
    return flags


def _jsonable(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return str(value) if value is not None else None


def preset_payload(
    name: str, reference: dict[str, Any], presets: dict[str, dict[str, object]]
) -> dict[str, Any]:
    """Resolve a named example, or the training profile when ``name`` is the typical file."""

    if name not in presets:
        return dict(reference)
    payload = dict(reference)
    payload.update(presets[name])
    unknown = [
        column
        for column, value in payload.items()
        if column in CATEGORY_OPTIONS and str(value) not in CATEGORY_OPTIONS[column]
    ]
    if unknown:
        raise ApplicationError(f"Preset '{name}' uses unknown categories: {unknown}")
    return payload
