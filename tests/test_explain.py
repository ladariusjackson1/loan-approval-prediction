"""Single-application decisions and ablation sentences."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from loan_approval.config import PRESETS
from loan_approval.explain import (
    ApplicationError,
    decision_summary,
    describe_value,
    explain_application,
    frame_from_application,
    local_contributions,
    preset_payload,
    reason_sentence,
    reference_profile,
    training_ranges,
)


def _separable_pipeline() -> tuple[Pipeline, pd.DataFrame]:
    frame = pd.DataFrame({"x": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], "z": [0, 1, 0, 1, 0, 1]})
    target = np.array([0, 0, 0, 1, 1, 1])
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ]
    )
    pipeline.fit(frame, target)
    return pipeline, frame


def test_reference_profile_uses_median_and_mode():
    frame = pd.DataFrame({"duration": [10, 20, 30], "housing": ["own", "own", "rent"]})
    profile = reference_profile(frame)
    assert profile["duration"] == 20
    assert profile["housing"] == "own"
    ranges = training_ranges(frame)
    assert ranges["duration"] == {"min": 10.0, "max": 30.0}
    assert "housing" not in ranges


def test_ablation_credits_the_field_that_separates_the_classes():
    pipeline, frame = _separable_pipeline()
    reference = {"x": 0.0, "z": 0}
    application = frame.iloc[[5]].copy()
    application["z"] = 0
    records = local_contributions(pipeline, application, reference)
    by_feature = {record["feature"]: record for record in records}
    assert by_feature["x"]["delta"] > 0
    assert abs(by_feature["z"]["delta"]) < 1e-8


def test_explain_application_flags_review_and_out_of_range_values():
    pipeline, _frame = _separable_pipeline()
    result = explain_application(
        pipeline,
        {"x": 3.0, "z": 1},
        reference={"x": 0.0, "z": 0},
        feature_columns=["x", "z"],
        threshold=0.5,
        ranges={"x": {"min": -2, "max": 1}},
        review_band=0.05,
    )
    assert result["decision"] == "Approve"
    assert result["out_of_range"] == ["x"]
    assert "Approve" in result["summary"]
    near = explain_application(
        pipeline,
        {"x": 0.0, "z": 0},
        reference={"x": 0.0, "z": 0},
        feature_columns=["x", "z"],
        threshold=float(pipeline.predict_proba(pd.DataFrame([{"x": 0.0, "z": 0}]))[0, 1]),
        review_band=0.05,
    )
    assert near["refer_to_analyst"] is True
    assert near["reasons_for"] == []
    assert near["reasons_against"] == []


def test_application_validation_errors():
    with pytest.raises(ApplicationError, match="Missing fields"):
        frame_from_application({"duration": 12}, ["duration", "purpose"])
    with pytest.raises(ApplicationError, match="numeric"):
        frame_from_application({"duration": "soon", "purpose": "radio/tv"}, ["duration", "purpose"])
    with pytest.raises(ApplicationError, match="finite"):
        frame_from_application(
            {"duration": math.nan, "purpose": "radio/tv"}, ["duration", "purpose"]
        )
    with pytest.raises(ApplicationError, match="required"):
        frame_from_application({"duration": 12, "purpose": "  "}, ["duration", "purpose"])


def test_presets_and_sentence_cover_known_and_unknown_fields():
    reference = {"checking_status": "no checking", "duration": 18, "purpose": "radio/tv"}
    typical = preset_payload("Typical training profile", reference, PRESETS)
    assert typical == reference
    stronger = preset_payload("Stronger file", reference, PRESETS)
    assert stronger["checking_status"] == "no checking"
    assert stronger["savings_status"] == ">=1000"
    assert stronger["purpose"] == "used car"
    broken = {"Broken": {"checking_status": "not a real code"}}
    with pytest.raises(ApplicationError, match="unknown categories"):
        preset_payload("Broken", reference, broken)
    assert "DM" in describe_value("credit_amount", 1500)
    assert "months" in describe_value("duration", 12)
    assert describe_value("checking_status", "<0").startswith("Overdrawn")
    assert describe_value("mystery", "raw") == "raw"
    text = reason_sentence(
        {
            "label": "Age (years)",
            "actual_text": "22 years",
            "reference_text": "33 years",
            "delta": -0.08,
        }
    )
    assert "lowers" in text
    flat = reason_sentence(
        {
            "label": "Dependents",
            "actual_text": "1",
            "reference_text": "1",
            "delta": 0.0,
        }
    )
    assert "barely moves" in flat


def test_decision_summary_states_which_side_of_the_cutoff():
    text = decision_summary("Decline", 0.42, 0.60)
    assert "Decline" in text
    assert "below" in text
