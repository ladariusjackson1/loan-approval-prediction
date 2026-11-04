"""Turn one application into a decision and a plain-language account of what pushed it."""

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
    """Change in P(fully paid) when each field is replaced by the training reference.

    A positive delta means the applicant's actual value raises the chance the loan
    is fully paid, relative to a typical training file. This is an ablation, not a
    causal effect and not a Shapley value: correlated fields share credit.
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
    """Score one application and say, in plain English, what pushed the decision."""

    application = frame_from_application(payload, feature_columns)
    probability = float(pipeline.predict_proba(application)[0, 1])
    decision = "Approve" if probability >= threshold else "Decline"
    contributions = local_contributions(pipeline, application, reference)
    reasons_for = _reasons(contributions, positive=True)
    reasons_against = _reasons(contributions, positive=False)
    out_of_range = _outside_training(application, ranges or {})
    return {
        "decision": decision,
        "probability_good": probability,
        "probability_fully_paid": probability,
        "threshold": float(threshold),
        "margin": probability - float(threshold),
        "refer_to_analyst": abs(probability - float(threshold)) < review_band,
        "out_of_range": out_of_range,
        "contributions": contributions,
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
        "summary": decision_summary(decision, probability, threshold),
        "narrative": decision_narrative(
            decision,
            probability,
            threshold,
            reasons_for,
            reasons_against,
        ),
    }


def decision_summary(decision: str, probability: float, threshold: float) -> str:
    """One sentence a reviewer can read before the longer note."""

    gap = probability - threshold
    direction = "above" if gap >= 0 else "below"
    return (
        f"{decision}. The model score for a fully paid loan is {probability:.1%}, "
        f"which is {abs(gap):.1%} {direction} the operating cutoff of {threshold:.1%}."
    )


def decision_narrative(
    decision: str,
    probability: float,
    threshold: float,
    reasons_for: list[dict[str, Any]],
    reasons_against: list[dict[str, Any]],
) -> str:
    """Plain-English note: the stamp, then the fields that pushed it.

    Each driver is the applicant's value compared with a typical training file.
    The note does not claim those fields caused repayment.
    """

    gap = probability - threshold
    relation = "above" if gap >= 0 else "below"
    outcome = "approved" if decision == "Approve" else "declined"
    sentences = [
        (
            f"{decision}. The model score for a fully paid loan is {probability:.1%}, "
            f"which is {abs(gap):.1%} {relation} the cutoff of {threshold:.1%}, "
            f"so the application is {outcome}."
        )
    ]
    if reasons_against:
        sentences.append(
            "What pushed the decision toward a decline: " + _driver_clause(reasons_against) + "."
        )
    if reasons_for:
        sentences.append(
            "What pushed the decision toward approval: " + _driver_clause(reasons_for) + "."
        )
    if not reasons_for and not reasons_against:
        sentences.append(
            "No single field pushed the decision. Replacing any one field with a typical "
            "training application moved the model score by less than half a percentage point."
        )
    return " ".join(sentences)


def _driver_clause(reasons: list[dict[str, Any]]) -> str:
    bits = [
        (
            f"{record['label']} is {record['actual_text']} "
            f"(a typical training application is {record['reference_text']})"
        )
        for record in reasons
    ]
    if len(bits) == 1:
        return bits[0]
    if len(bits) == 2:
        return f"{bits[0]} and {bits[1]}"
    return ", ".join(bits[:-1]) + ", and " + bits[-1]


def _is_numeric_field(column: str, value: object) -> bool:
    """Known numeric loan fields, plus raw numbers for columns outside the category contract."""

    if column in NUMERIC_FEATURES:
        return True
    if column in CATEGORY_OPTIONS or isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def describe_value(feature: str, value: object) -> str:
    """Human label for a raw category code or a numeric field."""

    if feature == "credit_policy":
        return VALUE_LABELS["credit_policy"].get(str(int(round(float(value)))), str(value))
    if feature in VALUE_LABELS and not _looks_numeric(feature, value):
        return VALUE_LABELS[feature].get(str(value), str(value))
    if feature in NUMERIC_FEATURES or (
        isinstance(value, (int, float, np.floating)) and feature not in VALUE_LABELS
    ):
        return _format_number(feature, float(value))
    text = str(value)
    return VALUE_LABELS.get(feature, {}).get(text, text)


def _looks_numeric(feature: str, value: object) -> bool:
    if feature in CATEGORY_OPTIONS:
        return False
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def reason_sentence(record: dict[str, Any]) -> str:
    """Describe one field in the words a reviewer would say out loud."""

    delta = float(record["delta"])
    comparison = (
        f"{record['label']} is {record['actual_text']}, against {record['reference_text']} "
        "on a typical training application"
    )
    if abs(delta) < 0.005:
        return f"{comparison}, and that barely moves the score."
    if delta > 0:
        return (
            f"{comparison}. That pushed the decision toward approval and raised "
            f"the model score by {delta:.1%}."
        )
    return (
        f"{comparison}. That pushed the decision toward a decline and lowered "
        f"the model score by {abs(delta):.1%}."
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
                "actual_text": record["actual_text"],
                "reference_text": record["reference_text"],
                "text": reason_sentence(record),
            }
        )
    return reasons


def _format_number(feature: str, value: float) -> str:
    if feature == "log_annual_inc":
        return f"${math.exp(value):,.0f} a year"
    if feature == "installment":
        return f"${value:,.0f} per month"
    if feature == "int_rate":
        return f"{value:.1%}"
    if feature == "dti":
        return f"{value:.1f}%"
    if feature == "fico":
        return f"{int(round(value))}"
    if feature == "days_with_cr_line":
        return f"{value / 365.25:.1f} years"
    if feature == "revol_bal":
        return f"${value:,.0f}"
    if feature == "revol_util":
        return f"{value:.0f}%"
    if feature in {"inq_last_6mths", "delinq_2yrs", "pub_rec", "credit_policy"}:
        return str(int(round(value)))
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
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
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
