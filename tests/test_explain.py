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
    decision_narrative,
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
    assert "fully paid" in result["narrative"]
    assert "pushed the decision toward approval" in result["narrative"]
    assert result["reasons_for"]
    assert result["reasons_for"][0]["actual_text"] in result["narrative"]
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
    assert "No single field pushed the decision" in near["narrative"]


def test_application_validation_errors():
    with pytest.raises(ApplicationError, match="Missing fields"):
        frame_from_application({"fico": 700}, ["fico", "purpose"])
    with pytest.raises(ApplicationError, match="numeric"):
        frame_from_application(
            {"fico": "high", "purpose": "debt_consolidation"}, ["fico", "purpose"]
        )
    with pytest.raises(ApplicationError, match="finite"):
        frame_from_application(
            {"fico": math.nan, "purpose": "debt_consolidation"}, ["fico", "purpose"]
        )
    with pytest.raises(ApplicationError, match="required"):
        frame_from_application({"fico": 700, "purpose": "  "}, ["fico", "purpose"])


def test_presets_and_sentence_cover_known_and_unknown_fields():
    reference = {"purpose": "debt_consolidation", "fico": 707, "dti": 12.6}
    typical = preset_payload("Typical training profile", reference, PRESETS)
    assert typical == reference
    stronger = preset_payload("Stronger file", reference, PRESETS)
    assert stronger["fico"] == 770
    assert stronger["purpose"] == "credit_card"
    assert stronger["dti"] == 6.5
    broken = {"Broken": {"purpose": "not a real purpose"}}
    with pytest.raises(ApplicationError, match="unknown categories"):
        preset_payload("Broken", reference, broken)
    assert "a year" in describe_value("log_annual_inc", math.log(55000))
    assert "per month" in describe_value("installment", 319)
    assert describe_value("purpose", "small_business") == "Small business"
    assert describe_value("fico", 662) == "662"
    assert describe_value("credit_policy", 1) == "Meets policy"
    assert describe_value("credit_policy", 0) == "Outside policy"
    assert describe_value("int_rate", 0.1226) == "12.3%"
    assert describe_value("dti", 12.66) == "12.7%"
    assert "years" in describe_value("days_with_cr_line", 3652.5)
    assert describe_value("revol_bal", 8596) == "$8,596"
    assert describe_value("revol_util", 46.2) == "46%"
    assert describe_value("inq_last_6mths", 1) == "1"
    assert describe_value("mystery", "raw") == "raw"
    text = reason_sentence(
        {
            "label": "FICO score",
            "actual_text": "662",
            "reference_text": "707",
            "delta": -0.08,
        }
    )
    assert "pushed the decision toward a decline" in text
    assert "lowered" in text
    flat = reason_sentence(
        {
            "label": "Derogatory public records",
            "actual_text": "0",
            "reference_text": "0",
            "delta": 0.0,
        }
    )
    assert "barely moves" in flat


def test_decision_summary_and_narrative_name_what_pushed_the_decision():
    text = decision_summary("Decline", 0.42, 0.60)
    assert "Decline" in text
    assert "below" in text
    assert "fully paid" in text
    narrative = decision_narrative(
        "Decline",
        0.42,
        0.60,
        reasons_for=[
            {
                "label": "Annual income",
                "actual_text": "$90,000 a year",
                "reference_text": "$55,000 a year",
            }
        ],
        reasons_against=[
            {
                "label": "FICO score",
                "actual_text": "640",
                "reference_text": "707",
            }
        ],
    )
    assert narrative.startswith("Decline.")
    assert "so the application is declined" in narrative
    assert "What pushed the decision toward a decline: FICO score is 640" in narrative
    assert "What pushed the decision toward approval: Annual income is $90,000 a year" in narrative
    quiet = decision_narrative("Approve", 0.70, 0.50, [], [])
    assert "No single field pushed the decision" in quiet
