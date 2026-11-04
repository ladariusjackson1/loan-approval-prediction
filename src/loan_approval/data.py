"""Load the Lending Club sample from the local cache or from OpenML."""

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
    NUMERIC_FEATURES,
    OPENML_DATA_ID,
    OPENML_NAME,
    POSITIVE_CLASS,
    REQUIRED_COLUMNS,
    TARGET,
    VALUE_LABELS,
)

# OpenML keeps the original dotted names. The cache and the model use underscores.
OPENML_RENAMES = {
    "credit.policy": "credit_policy",
    "int.rate": "int_rate",
    "log.annual.inc": "log_annual_inc",
    "days.with.cr.line": "days_with_cr_line",
    "revol.bal": "revol_bal",
    "revol.util": "revol_util",
    "inq.last.6mths": "inq_last_6mths",
    "delinq.2yrs": "delinq_2yrs",
    "pub.rec": "pub_rec",
    "not.fully.paid": "not_fully_paid",
}


class DatasetError(ValueError):
    """The loan table is missing, mis-shaped, or uses an unknown label."""


def load_loan_data(path: Path | None = None, *, download: bool = False) -> pd.DataFrame:
    """Return the cached Lending Club sample, refreshing it from OpenML when asked.

    The committed CSV is the whole OpenML extract (a public sample, not the full
    servicing tape). Training, tests, CI, and the app read that file offline.
    ``download=True`` refetches OpenML data id 43729 and rewrites the cache.
    """

    destination = Path(path) if path is not None else DATA_PATH
    if destination.exists() and not download:
        return _read_loan_csv(destination)
    if not download and not destination.exists():
        raise DatasetError(
            f"Dataset not found at {destination}. Run with download=True to fetch "
            f"OpenML data id {OPENML_DATA_ID} ({OPENML_NAME})."
        )
    frame = download_openml_loans()
    _write_loan_csv(frame, destination)
    return _read_loan_csv(destination)


def download_openml_loans() -> pd.DataFrame:
    """Fetch OpenML data id 43729. Network is required; no account is required."""

    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    frame = bunch.frame
    if frame is None:
        raise DatasetError(f"OpenML {OPENML_NAME} returned no dataframe.")
    renamed = frame.rename(columns=OPENML_RENAMES)
    return renamed


def encode_target(series: pd.Series) -> pd.Series:
    """Map not-fully-paid 0 → 1 (fully paid, approve) and 1 → 0 (decline)."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise DatasetError("Target must be 0 (fully paid) or 1 (not fully paid), with no gaps.")
    encoded = (numeric == 0).astype(int)
    encoded.name = "fully_paid"
    if POSITIVE_CLASS != 1:
        raise DatasetError("Positive class must stay coded as 1.")
    return encoded


def fico_band(fico: object) -> str:
    """Screening bands used only in the segment report, not as model inputs."""

    number = float(fico)
    if math.isnan(number):
        return "unknown"
    if number < 680:
        return "Under 680"
    if number < 720:
        return "680–719"
    if number < 760:
        return "720–759"
    return "760 and over"


def credit_policy_label(value: object) -> str:
    """Label the platform's own underwriting flag. It is not a model input."""

    number = float(value)
    if math.isnan(number):
        return "unknown"
    if number >= 1:
        return "Meets policy"
    return "Outside policy"


def purpose_label(value: object) -> str:
    text = str(value).strip()
    return VALUE_LABELS["purpose"].get(text, text)


def sensitive_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Segment columns for the audit. They are not all fed to the deployed model.

    This extract has no sex, age, or race fields, so the report cannot be a
    protected-class audit. It compares the decision across the old policy flag,
    FICO band, and loan purpose.
    """

    missing = [column for column in ("credit_policy", "fico", "purpose") if column not in df]
    if missing:
        raise DatasetError(f"Cannot build the segment frame without {missing}.")
    return pd.DataFrame(
        {
            "credit_policy": df["credit_policy"].map(credit_policy_label),
            "fico_band": df["fico"].map(fico_band),
            "purpose": df["purpose"].map(purpose_label),
        },
        index=df.index,
    )


def modeling_matrices(
    df: pd.DataFrame,
    *,
    excluded: list[str] | tuple[str, ...] = EXCLUDED_FEATURES,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y, and the segment frame aligned on the same index."""

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


def _read_loan_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(f"{path} is missing columns: {missing}")
    return _coerce(frame[REQUIRED_COLUMNS].copy())


def _write_loan_csv(frame: pd.DataFrame, path: Path) -> None:
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
    for column in CATEGORICAL_FEATURES:
        as_text = coerced[column].astype("string").str.strip()
        as_object = as_text.astype(object)
        invalid = as_text.isna() | as_text.isin(["", "nan", "None", "<NA>"])
        as_object = as_object.where(~invalid.to_numpy(), np.nan)
        coerced[column] = as_object
    target = pd.to_numeric(coerced[TARGET], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise DatasetError("Target must be 0 (fully paid) or 1 (not fully paid), with no gaps.")
    coerced[TARGET] = target.astype("int64")
    return coerced
