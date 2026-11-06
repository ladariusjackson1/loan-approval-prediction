"""Sklearn preprocessing that stays inside the model pipeline."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_approval.data import feature_types


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    """Median-impute and scale numbers; mode-impute and one-hot encode categories.

    Scaling sits in the shared pipeline so logistic regularization sees comparable
    units. Tree splits are invariant to a per-feature affine rescaling, so
    histogram gradient boosting is not helped or hurt by that step.
    """

    numeric, categorical = feature_types(columns)
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(estimator: object, columns: list[str]) -> Pipeline:
    """Chain preprocessing and a classifier. The whole object is what gets cross-validated."""

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(columns)),
            ("clf", estimator),
        ]
    )
