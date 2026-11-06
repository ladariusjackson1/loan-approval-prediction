"""Score a table of applications and attach decline reasons.

Probabilities are one batched prediction. Field ablations are one batched
prediction per column. Counterfactual hints for declined rows try a shared
grid of in-range values, one batched prediction per candidate, and report a
value that was actually scored.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from loan_approval.calibration import apply_scores
from loan_approval.config import CATEGORY_OPTIONS, FEATURE_LABELS, NUMERIC_FEATURES
from loan_approval.explain import (
    ApplicationError,
    adverse_reason_text,
    describe_value,
    frame_from_application,
)

SCORE_COLUMNS = (
    "probability_fully_paid",
    "decision",
    "adverse_reason_1",
    "adverse_reason_2",
    "adverse_reason_3",
    "counterfactual",
)
_BATCH_GRID = 13


def prepare_applications(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Validate every row and return the columns the pipeline expects, in order."""

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ApplicationError(f"Missing columns: {', '.join(missing)}")
    rows: list[pd.Series] = []
    for position, record in enumerate(frame.to_dict(orient="records"), start=1):
        payload = {column: record[column] for column in feature_columns}
        try:
            one = frame_from_application(payload, feature_columns)
        except ApplicationError as exc:
            raise ApplicationError(f"Row {position}: {exc}") from exc
        rows.append(one.iloc[0])
    if not rows:
        return pd.DataFrame(columns=feature_columns)
    return pd.DataFrame(rows, columns=feature_columns).reset_index(drop=True)


def score_batch(
    pipeline: Any,
    frame: pd.DataFrame,
    *,
    reference: dict[str, Any],
    feature_columns: list[str],
    threshold: float,
    ranges: dict[str, dict[str, float]] | None = None,
    calibrator: Any | None = None,
) -> pd.DataFrame:
    """Return the input rows plus a decision, three decline reasons, and a hint.

    Extra columns on ``frame`` are kept and written back beside the scores.
    Rows are validated before any model call.
    """

    prepared = prepare_applications(frame, feature_columns)
    scored = frame.reset_index(drop=True).copy()
    if prepared.empty:
        for column in SCORE_COLUMNS:
            scored[column] = pd.Series(dtype=object)
        return scored
    probability = apply_scores(pipeline, prepared, calibrator)
    decision = np.where(probability >= float(threshold), "Approve", "Decline")
    reason_columns = _adverse_columns(pipeline, prepared, reference, probability, calibrator)
    hints = _batch_hints(
        pipeline,
        prepared,
        probability=probability,
        threshold=float(threshold),
        ranges=ranges or {},
        calibrator=calibrator,
    )
    scored["probability_fully_paid"] = probability
    scored["decision"] = decision
    for index, column in enumerate(("adverse_reason_1", "adverse_reason_2", "adverse_reason_3")):
        texts = []
        for row_index, label in enumerate(decision):
            if label != "Decline":
                texts.append("")
                continue
            reasons = reason_columns[row_index]
            texts.append(reasons[index] if index < len(reasons) else "")
        scored[column] = texts
    scored["counterfactual"] = hints
    return scored


def application_template(
    feature_columns: list[str],
    examples: list[dict[str, Any]],
    labels: list[str],
) -> pd.DataFrame:
    """A small CSV of example rows. ``labels`` are not model inputs."""

    rows = []
    for label, payload in zip(labels, examples, strict=True):
        row = {"example": label}
        for column in feature_columns:
            row[column] = payload[column]
        rows.append(row)
    return pd.DataFrame(rows)


def _adverse_columns(
    pipeline: Any,
    prepared: pd.DataFrame,
    reference: dict[str, Any],
    probability: np.ndarray,
    calibrator: Any | None,
) -> list[list[str]]:
    deltas = _ablation_deltas(pipeline, prepared, reference, probability, calibrator)
    reasons: list[list[str]] = []
    for row_index in range(len(prepared)):
        contributions = []
        for column in prepared.columns:
            actual = prepared.iloc[row_index][column]
            contributions.append(
                {
                    "feature": column,
                    "label": FEATURE_LABELS.get(column, column),
                    "actual": actual,
                    "reference": reference[column],
                    "actual_text": describe_value(column, actual),
                    "reference_text": describe_value(column, reference[column]),
                    "delta": float(deltas[column][row_index]),
                }
            )
        harmful = [item for item in contributions if item["delta"] < -0.005]
        harmful.sort(key=lambda item: float(item["delta"]))
        reasons.append([adverse_reason_text(item) for item in harmful[:3]])
    return reasons


def _ablation_deltas(
    pipeline: Any,
    prepared: pd.DataFrame,
    reference: dict[str, Any],
    probability: np.ndarray,
    calibrator: Any | None,
) -> dict[str, np.ndarray]:
    deltas: dict[str, np.ndarray] = {}
    for column in prepared.columns:
        alternate = prepared.copy()
        if column in NUMERIC_FEATURES or pd.api.types.is_numeric_dtype(alternate[column]):
            alternate[column] = alternate[column].astype(float)
            alternate[column] = float(reference[column])
        else:
            alternate[column] = alternate[column].astype(object)
            alternate[column] = None if reference[column] is None else str(reference[column])
        swapped = apply_scores(pipeline, alternate, calibrator)
        deltas[column] = probability - swapped
    return deltas


def _batch_hints(
    pipeline: Any,
    prepared: pd.DataFrame,
    *,
    probability: np.ndarray,
    threshold: float,
    ranges: dict[str, dict[str, float]],
    calibrator: Any | None,
) -> list[str]:
    texts = np.empty(len(prepared), dtype=object)
    approved = probability >= threshold
    texts[approved] = "This file is at or above the cutoff, so there is no decline to reverse."
    declined_index = np.flatnonzero(~approved)
    if declined_index.size == 0:
        return texts.tolist()
    declined = prepared.iloc[declined_index].reset_index(drop=True)
    base = probability[declined_index]
    best_distance = np.full(declined_index.size, np.inf)
    best_text = np.empty(declined_index.size, dtype=object)
    best_text[:] = ""
    for column in prepared.columns:
        _consider_column(
            pipeline,
            declined,
            column=column,
            base=base,
            threshold=threshold,
            ranges=ranges,
            calibrator=calibrator,
            best_distance=best_distance,
            best_text=best_text,
        )
    unresolved = ~np.isfinite(best_distance)
    best_text[unresolved] = "No single field, moved within the training range, clears the cutoff."
    texts[declined_index] = best_text
    return texts.tolist()


def _consider_column(
    pipeline: Any,
    declined: pd.DataFrame,
    *,
    column: str,
    base: np.ndarray,
    threshold: float,
    ranges: dict[str, dict[str, float]],
    calibrator: Any | None,
    best_distance: np.ndarray,
    best_text: np.ndarray,
) -> None:
    numeric = column in NUMERIC_FEATURES or pd.api.types.is_numeric_dtype(declined[column])
    if not numeric:
        options = CATEGORY_OPTIONS.get(column, [])
        current = declined[column].astype(str).to_numpy()
        for option in options:
            distance = np.where(current == option, np.inf, 1.0)
            alternate = declined.copy()
            alternate[column] = alternate[column].astype(object)
            alternate[column] = option
            _keep_better(
                declined,
                column=column,
                value=option,
                scores=apply_scores(pipeline, alternate, calibrator),
                base=base,
                threshold=threshold,
                distance=distance,
                best_distance=best_distance,
                best_text=best_text,
            )
        return
    bounds = ranges.get(column)
    if not bounds:
        return
    low = float(bounds["min"])
    high = float(bounds["max"])
    span = high - low
    if span <= 0:
        return
    current = declined[column].to_numpy(dtype=float)
    for value in _batch_grid(low, high):
        distance = np.abs(current - float(value)) / span
        distance = np.where(distance <= 1e-6, np.inf, distance)
        alternate = declined.copy()
        alternate[column] = alternate[column].astype(float)
        alternate[column] = float(value)
        _keep_better(
            declined,
            column=column,
            value=value,
            scores=apply_scores(pipeline, alternate, calibrator),
            base=base,
            threshold=threshold,
            distance=distance,
            best_distance=best_distance,
            best_text=best_text,
        )


def _keep_better(
    declined: pd.DataFrame,
    *,
    column: str,
    value: object,
    scores: np.ndarray,
    base: np.ndarray,
    threshold: float,
    distance: np.ndarray,
    best_distance: np.ndarray,
    best_text: np.ndarray,
) -> None:
    flips = (scores >= threshold) & (distance < best_distance)
    if not np.any(flips):
        return
    label = FEATURE_LABELS.get(column, column)
    new_text = describe_value(column, value)
    for index in np.flatnonzero(flips):
        old_text = describe_value(column, declined.iloc[int(index)][column])
        best_distance[index] = float(distance[index])
        best_text[index] = (
            f"Holding every other field fixed, {label} of {new_text} instead of {old_text} "
            f"moves the score from {float(base[index]):.1%} to {float(scores[index]):.1%} "
            f"and clears the cutoff of {threshold:.1%}. This is a one-field illustration "
            "on the model, not a change to make on a credit file."
        )


def _batch_grid(low: float, high: float) -> list[float]:
    if math.isclose(low, high):
        return []
    return [float(value) for value in np.linspace(low, high, _BATCH_GRID)]
