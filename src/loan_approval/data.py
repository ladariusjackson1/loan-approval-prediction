"""Load the German Credit table from the local cache or from OpenML."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from loan_approval.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    EXCLUDED_FEATURES,
    FEATURE_COLUMNS,
    NEGATIVE_LABEL,
    NUMERIC_FEATURES,
    OPENML_DATA_ID,
    POSITIVE_CLASS,
    POSITIVE_LABEL,
    REQUIRED_COLUMNS,
    TARGET,
)


class DatasetError(ValueError):
    """The credit table is missing, mis-shaped, or uses an unknown label."""


def load_credit_data(path: Path | None = None, *, download: bool = False) -> pd.DataFrame:
    """Return the credit table, fetching OpenML credit-g only when asked.

    The committed CSV is the reproducible source for training, tests, and CI.
    ``download=True`` refreshes that file from ``sklearn.datasets.fetch_openml``.
    """

    destination = Path(path) if path is not None else DATA_PATH
    if destination.exists() and not download:
        return _read_credit_csv(destination)
    if not download and not destination.exists():
        raise DatasetError(
            f"Dataset not found at {destination}. Run with download=True to fetch "
            f"OpenML data id {OPENML_DATA_ID} (credit-g)."
        )
    frame = download_openml_credit()
    _write_credit_csv(frame, destination)
    return _read_credit_csv(destination)


def download_openml_credit() -> pd.DataFrame:
    """Fetch OpenML data id 31. Network is required; the result is not yet schema-checked."""

    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    frame = bunch.frame
    if frame is None:
        raise DatasetError("OpenML credit-g returned no dataframe.")
    return frame


def encode_target(series: pd.Series) -> pd.Series:
    """Map good → 1 (approve) and bad → 0 (decline)."""

    labels = set(series.dropna().astype(str).str.strip().unique())
    unknown = labels - {POSITIVE_LABEL, NEGATIVE_LABEL}
    if unknown:
        raise DatasetError(f"Unknown class labels: {sorted(unknown)}")
    mapped = series.astype(str).str.strip().map({POSITIVE_LABEL: 1, NEGATIVE_LABEL: 0})
    if mapped.isna().any():
        raise DatasetError("Target contains missing values.")
    encoded = mapped.astype(int)
    encoded.name = "approve_good_credit"
    if POSITIVE_CLASS != 1:
        raise DatasetError("Positive class must stay coded as 1.")
    return encoded


def sex_from_personal_status(value: object) -> str:
    """Collapse the mixed personal-status field into a sex label for the audit.

    The source field has no code for single women, so sex and marital status
    cannot be separated. ``unknown`` is returned for anything else.
    """

    text = str(value).strip().lower()
    if text.startswith("male"):
        return "male"
    if text.startswith("female"):
        return "female"
    return "unknown"


def age_band(age: object) -> str:
    """Screening bands used only in the fairness report, not as model inputs."""

    number = float(age)
    if math.isnan(number):
        return "unknown"
    if number < 25:
        return "Under 25"
    return "25 and over"


def sensitive_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the audit columns. These are not all fed to the deployed model."""

    missing = [
        column for column in ("personal_status", "age", "foreign_worker") if column not in df
    ]
    if missing:
        raise DatasetError(f"Cannot build the fairness frame without {missing}.")
    return pd.DataFrame(
        {
            "sex": df["personal_status"].map(sex_from_personal_status),
            "age_band": df["age"].map(age_band),
            "foreign_worker": df["foreign_worker"].astype(str).str.strip(),
        },
        index=df.index,
    )


def modeling_matrices(
    df: pd.DataFrame,
    *,
    excluded: list[str] | tuple[str, ...] = EXCLUDED_FEATURES,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y, and the sensitive-attribute frame aligned on the same index."""

    missing = [column for column in excluded if column not in df.columns]
    if missing:
        raise DatasetError(f"Excluded columns are not in the table: {missing}")
    features = [column for column in FEATURE_COLUMNS if column not in set(excluded)]
    predictors = df[features].copy()
    target = encode_target(df[TARGET])
    return predictors, target, sensitive_frame(df)


def feature_types(columns: list[str] | tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Split a column list into numeric and categorical features, preserving order."""

    unknown = [column for column in columns if column not in FEATURE_COLUMNS]
    if unknown:
        raise DatasetError(f"Unknown feature columns: {unknown}")
    numeric = [column for column in columns if column in NUMERIC_FEATURES]
    categorical = [column for column in columns if column in CATEGORICAL_FEATURES]
    return numeric, categorical


def _read_credit_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(f"{path} is missing columns: {missing}")
    return _coerce(frame[REQUIRED_COLUMNS].copy())


def _write_credit_csv(frame: pd.DataFrame, path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(f"OpenML frame is missing columns: {missing}")
    cleaned = _coerce(frame[REQUIRED_COLUMNS].copy())
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(path, index=False)


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    coerced = frame.copy()
    for column in NUMERIC_FEATURES:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce").astype("float64")
    for column in [*CATEGORICAL_FEATURES, TARGET]:
        as_text = coerced[column].astype("string").str.strip()
        as_object = as_text.astype(object)
        invalid = as_text.isna() | as_text.isin(["", "nan", "None", "<NA>"])
        as_object = as_object.where(~invalid.to_numpy(), np.nan)
        coerced[column] = as_object
    return coerced
