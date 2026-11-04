"""The shared imputation, encoding, and scaling pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from loan_approval.models import make_estimator, search_space
from loan_approval.preprocessing import build_model_pipeline, build_preprocessor


def test_pipeline_imputes_scales_and_ignores_unseen_categories(synthetic_credit):
    columns = ["fico", "installment", "purpose", "dti"]
    frame = synthetic_credit[columns].copy()
    frame.loc[0, "fico"] = np.nan
    frame.loc[1, "purpose"] = np.nan
    pipeline = build_model_pipeline(make_estimator("logistic_regression", seed=0), columns)
    pipeline.fit(frame, (synthetic_credit["not_fully_paid"] == 0).astype(int))

    row = pd.DataFrame(
        [
            {
                "fico": np.nan,
                "installment": 250.0,
                "purpose": "brand new purpose",
                "dti": np.nan,
            }
        ]
    )
    transformed = pipeline.named_steps["preprocess"].transform(row)
    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()
    probabilities = pipeline.predict_proba(row)
    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(), 1)


def test_preprocessor_drops_columns_that_are_not_declared(synthetic_credit):
    columns = ["fico", "purpose"]
    preprocessor = build_preprocessor(columns)
    extra = synthetic_credit[columns].copy()
    extra["leak"] = 1
    transformed = preprocessor.fit_transform(extra)
    names = list(preprocessor.get_feature_names_out())
    assert transformed.shape[1] == len(names)
    assert all("leak" not in name for name in names)


def test_unknown_model_names_are_rejected():
    for factory in (make_estimator, search_space):
        try:
            factory("svm", 0) if factory is make_estimator else factory("svm")
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "Unknown model" in str(exc)
