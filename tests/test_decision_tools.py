"""Bootstrap intervals, calibration, portfolio math, reasons, and batch scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from loan_approval.bootstrap import bootstrap_roc_auc, paired_auc_difference
from loan_approval.calibration import (
    _predict_map,
    apply_scores,
    constant_brier,
    cross_fit_isotonic,
)
from loan_approval.config import FN_COST, FP_COST
from loan_approval.explain import (
    adverse_reason_text,
    adverse_reasons,
    counterfactual_hint,
    explain_application,
)
from loan_approval.model_card_doc import render_model_card
from loan_approval.models import make_estimator, search_space
from loan_approval.portfolio import portfolio_curve, portfolio_snapshot
from loan_approval.scoring import application_template, score_batch
from loan_approval.training import _honest_winner, _selection_note


def test_unknown_model_names_are_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        make_estimator("random_forest", 0)
    with pytest.raises(ValueError, match="Unknown model"):
        search_space("gradient_boosting")


def test_bootstrap_interval_and_the_paired_difference():
    y = np.array([0, 1])
    too_small = bootstrap_roc_auc(y, np.array([0.2, 0.8]), n_boot=20, seed=0)
    assert too_small["ci_low"] is None
    assert too_small["estimate"] == pytest.approx(1.0)

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 400)
    perfect = labels.astype(float)
    constant = np.full(400, 0.5)
    noise = rng.random(400)
    difference = paired_auc_difference(labels, perfect, constant, n_boot=200, seed=1)
    assert difference["estimate"] == pytest.approx(0.5)
    assert difference["includes_zero"] is False
    assert difference["ci_low"] > 0

    identical = paired_auc_difference(labels, noise, noise, n_boot=80, seed=2)
    assert identical["estimate"] == 0
    assert identical["includes_zero"] is True
    assert identical["ci_low"] == 0
    assert identical["ci_high"] == 0

    point = bootstrap_roc_auc(labels, noise, n_boot=80, seed=3)
    assert point["ci_low"] <= point["estimate"] <= point["ci_high"]


def test_constant_brier_and_cross_fit_isotonic_stay_inside_the_unit_interval():
    labels = np.array([0, 0, 1, 1])
    assert constant_brier(labels) == pytest.approx(0.25)
    assert np.isnan(constant_brier(np.array([])))
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    mapping, calibrated = cross_fit_isotonic(scores, labels, cv)
    assert calibrated.shape == scores.shape
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    assert np.all((mapping.predict(scores) >= 0) & (mapping.predict(scores) <= 1))
    fallback = _predict_map(None, scores, np.array([1, 0, 1, 1]))
    assert fallback.shape == scores.shape
    with pytest.raises(ValueError, match="same length"):
        cross_fit_isotonic(scores, labels[:2], cv)


def test_portfolio_snapshot_matches_a_hand_calculation():
    actual = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2])
    snapshot = portfolio_snapshot(actual, scores, 0.65)
    assert snapshot["approvals"] == 3
    assert snapshot["expected_defaults"] == pytest.approx(0.6)
    hand = (
        0.9 * FN_COST
        - 0.1 * FP_COST
        + 0.8 * FN_COST
        - 0.2 * FP_COST
        + 0.7 * FN_COST
        - 0.3 * FP_COST
    )
    assert snapshot["expected_profit_vs_decline_all"] == pytest.approx(hand)
    assert snapshot["realized_defaults"] == 1
    assert snapshot["realized_fully_paid"] == 2
    empty = portfolio_snapshot(actual, scores, 0.99)
    assert empty["approvals"] == 0
    assert empty["expected_profit_vs_decline_all"] == 0
    assert empty["expected_default_rate"] is None
    curve = portfolio_curve(actual, scores, grid=np.array([0.65, 0.99]))
    assert list(curve["approvals"]) == [3, 0]
    with pytest.raises(ValueError, match="same length"):
        portfolio_snapshot(actual, scores[:2], 0.5)


def test_honest_winner_keeps_logistic_regression_when_the_gap_includes_zero():
    labels = np.array([0, 1] * 80)
    scores = np.linspace(0.05, 0.95, labels.size)
    chosen, details = _honest_winner(
        "hist_gradient_boosting",
        {"hist_gradient_boosting": scores, "logistic_regression": scores},
        pd.Series(labels),
        n_bootstrap=80,
        seed=0,
    )
    assert chosen == "logistic_regression"
    assert details[0]["includes_zero"] is True
    note = _selection_note("hist_gradient_boosting", chosen, details)
    assert "includes zero" in note
    assert "Logistic regression" in note

    perfect = labels.astype(float)
    flat = np.full(labels.size, 0.5)
    boosted, boosted_details = _honest_winner(
        "hist_gradient_boosting",
        {"hist_gradient_boosting": perfect, "logistic_regression": flat},
        pd.Series(labels),
        n_bootstrap=80,
        seed=0,
    )
    assert boosted == "hist_gradient_boosting"
    assert "excludes zero" in _selection_note("hist_gradient_boosting", boosted, boosted_details)

    tiny = np.array([0, 1])
    unavailable, unavailable_details = _honest_winner(
        "hist_gradient_boosting",
        {
            "hist_gradient_boosting": np.array([0.2, 0.8]),
            "logistic_regression": np.array([0.4, 0.6]),
        },
        pd.Series(tiny),
        n_bootstrap=20,
        seed=0,
    )
    assert unavailable == "hist_gradient_boosting"
    assert unavailable_details[0]["includes_zero"] is None
    assert "unavailable" in _selection_note(
        "hist_gradient_boosting", unavailable, unavailable_details
    )
    assert "excludes zero" in _selection_note(
        "logistic_regression",
        "logistic_regression",
        [
            {
                "other_model": "hist_gradient_boosting",
                "includes_zero": False,
                "auc_difference": 0.04,
                "ci_low": 0.01,
                "ci_high": 0.07,
            }
        ],
    )
    assert "No more complex model" in _selection_note(
        "logistic_regression",
        "logistic_regression",
        [
            {
                "other_model": "hist_gradient_boosting",
                "includes_zero": True,
                "auc_difference": 0.0,
                "ci_low": -0.01,
                "ci_high": 0.01,
            }
        ],
    )


def test_adverse_reasons_and_a_scored_counterfactual():
    reason = adverse_reason_text(
        {
            "feature": "fico",
            "label": "FICO score",
            "actual": 640,
            "reference": 710,
            "actual_text": "640",
            "reference_text": "710",
            "delta": -0.08,
        }
    )
    assert reason.startswith("Credit score is below a typical training file")
    assert "lowered the ranking score by 8.0%" in reason
    higher = adverse_reason_text(
        {
            "feature": "inq_last_6mths",
            "label": "Credit inquiries in the last 6 months",
            "actual": 6,
            "reference": 1,
            "actual_text": "6",
            "reference_text": "1",
            "delta": -0.04,
        }
    )
    assert higher.startswith("Recent credit inquiries are above")
    purpose = adverse_reason_text(
        {
            "feature": "purpose",
            "label": "Loan purpose",
            "actual": "small_business",
            "reference": "debt_consolidation",
            "actual_text": "Small business",
            "reference_text": "Debt consolidation",
            "delta": -0.03,
        }
    )
    assert purpose.startswith("Loan purpose is Small business")
    odd = adverse_reason_text(
        {
            "feature": "fico",
            "label": "FICO score",
            "actual": 760,
            "reference": 700,
            "actual_text": "760",
            "reference_text": "700",
            "delta": -0.02,
        }
    )
    assert "differs from a typical training file" in odd
    blank = adverse_reason_text(
        {
            "feature": "mystery",
            "label": "Mystery",
            "actual": "x",
            "reference": "y",
            "actual_text": "x",
            "reference_text": "y",
            "delta": -0.02,
        }
    )
    assert blank.startswith("Mystery differs")
    ranked = adverse_reasons(
        [
            {
                "feature": "fico",
                "label": "FICO score",
                "delta": -0.08,
                "actual_text": "640",
                "reference_text": "710",
                "actual": 640,
                "reference": 710,
            },
            {
                "feature": "dti",
                "label": "Debt-to-income ratio",
                "delta": -0.001,
                "actual_text": "1.0%",
                "reference_text": "1.0%",
                "actual": 1,
                "reference": 1,
            },
        ]
    )
    assert len(ranked) == 1

    frame = pd.DataFrame({"x": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], "z": [0, 1, 0, 1, 0, 1]})
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipeline.fit(frame, np.array([0, 0, 0, 1, 1, 1]))
    application = frame.iloc[[0]].copy()
    hint = counterfactual_hint(
        pipeline,
        application,
        threshold=0.5,
        ranges={"x": {"min": -2.0, "max": 3.0}, "z": {"min": 0.0, "max": 1.0}},
    )
    assert hint["flipped"] is True
    assert "clears the cutoff" in hint["text"]
    approved = counterfactual_hint(
        pipeline,
        frame.iloc[[5]].copy(),
        threshold=0.5,
        ranges={"x": {"min": -2.0, "max": 3.0}},
    )
    assert approved["flipped"] is False
    assert "no decline to reverse" in approved["text"]
    stuck = counterfactual_hint(
        pipeline,
        application,
        threshold=1.01,
        ranges={"x": {"min": -2.0, "max": 3.0}, "z": {"min": 0.0, "max": 1.0}},
    )
    assert stuck["flipped"] is False
    assert "No single field" in stuck["text"]
    explained = explain_application(
        pipeline,
        {"x": -2.0, "z": 0},
        reference={"x": 0.0, "z": 0},
        feature_columns=["x", "z"],
        threshold=0.5,
        ranges={"x": {"min": -2.0, "max": 3.0}, "z": {"min": 0.0, "max": 1.0}},
    )
    assert explained["decision"] == "Decline"
    assert explained["adverse_reasons"]
    assert explained["counterfactual"]["flipped"] is True


def test_batch_scoring_validates_rows_and_writes_reasons():
    frame = pd.DataFrame({"x": [-2.0, 3.0], "z": [0.0, 1.0]})
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipeline.fit(frame, np.array([0, 1]))
    scored = score_batch(
        pipeline,
        frame,
        reference={"x": 0.0, "z": 0.0},
        feature_columns=["x", "z"],
        threshold=0.5,
        ranges={"x": {"min": -2.0, "max": 3.0}, "z": {"min": 0.0, "max": 1.0}},
    )
    assert list(scored["decision"]) == ["Decline", "Approve"]
    assert scored.loc[0, "adverse_reason_1"]
    assert "clears the cutoff" in scored.loc[0, "counterfactual"]
    assert scored.loc[1, "adverse_reason_1"] == ""
    assert "no decline to reverse" in scored.loc[1, "counterfactual"]
    probabilities = apply_scores(pipeline, frame, None)
    assert probabilities.shape == (2,)

    from loan_approval.explain import ApplicationError

    with pytest.raises(ApplicationError, match="Missing columns"):
        score_batch(
            pipeline,
            frame,
            reference={"x": 0.0},
            feature_columns=["x", "purpose"],
            threshold=0.5,
        )
    bad = pd.DataFrame({"fico": ["high"], "purpose": ["debt_consolidation"]})
    with pytest.raises(ApplicationError, match="Row 1"):
        score_batch(
            pipeline,
            bad,
            reference={"fico": 700, "purpose": "debt_consolidation"},
            feature_columns=["fico", "purpose"],
            threshold=0.5,
        )
    empty = score_batch(
        pipeline,
        frame.iloc[[]],
        reference={"x": 0.0, "z": 0.0},
        feature_columns=["x", "z"],
        threshold=0.5,
    )
    assert empty.empty
    assert "decision" in empty.columns
    template = application_template(["x"], [{"x": 1.0}], ["Example"])
    assert template.loc[0, "example"] == "Example"


def test_batch_purpose_change_calibrator_and_empty_card_branches():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    from loan_approval.calibration import _fit_isotonic
    from loan_approval.explain import _looks_numeric
    from loan_approval.preprocessing import build_model_pipeline

    assert _looks_numeric("purpose", 1.0) is False
    assert _looks_numeric("mystery", True) is False
    assert _looks_numeric("mystery", float("nan")) is True
    assert _fit_isotonic(np.array([0.2, 0.8]), np.array([1, 1])) is None

    features = pd.DataFrame(
        {
            "fico": [620.0, 640.0, 760.0, 780.0],
            "purpose": ["small_business", "small_business", "credit_card", "credit_card"],
            "installment": [100.0, 120.0, 400.0, 420.0],
        }
    )
    labels = np.array([0, 0, 1, 1])
    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=400),
        ["fico", "purpose", "installment"],
    )
    pipeline.fit(features, labels)
    raw = pipeline.predict_proba(features)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw, labels)
    mapped = apply_scores(pipeline, features, calibrator)
    assert mapped.shape == (4,)
    scored = score_batch(
        pipeline,
        features,
        reference={"fico": 700.0, "purpose": "debt_consolidation", "installment": 200.0},
        feature_columns=["fico", "purpose", "installment"],
        threshold=0.5,
        ranges={
            "fico": {"min": 600.0, "max": 800.0},
            "installment": {"min": 50.0, "max": 500.0},
        },
        calibrator=calibrator,
    )
    assert set(scored["decision"]) <= {"Approve", "Decline"}
    approved_only = score_batch(
        pipeline,
        features,
        reference={"fico": 700.0, "purpose": "credit_card", "installment": 200.0},
        feature_columns=["fico", "purpose", "installment"],
        threshold=0.0,
        ranges={"fico": {"min": 700.0, "max": 700.0}},
        calibrator=calibrator,
    )
    assert set(approved_only["decision"]) == {"Approve"}
    wide = features.iloc[[0]].copy()
    counterfactual_hint(
        pipeline,
        wide,
        threshold=0.99,
        ranges={
            "fico": {"min": 500.0, "max": 820.0},
            "installment": {"min": 50.0, "max": 900.0},
        },
        calibrator=calibrator,
    )
    no_room = counterfactual_hint(
        pipeline,
        wide,
        threshold=0.99,
        ranges={},
        calibrator=calibrator,
    )
    assert "cutoff" in no_room["text"]
    curve = portfolio_curve(labels, mapped)
    assert len(curve) == 91
    empty_card = render_model_card(
        {
            "winner": {},
            "comparison": [],
            "bootstrap_comparisons": [
                {
                    "sample": "holdout",
                    "reference_model": "logistic_regression",
                    "other_model": "hist_gradient_boosting",
                    "auc_difference": 0.01,
                    "ci_low": None,
                    "includes_zero": False,
                    "used_for_selection": False,
                }
            ],
            "fairness_oof": pd.DataFrame(
                [
                    {
                        "attribute": "purpose",
                        "group": "Education",
                        "n": 2,
                        "base_rate": None,
                        "approval_rate": "n/a",
                        "disparate_impact": None,
                    }
                ]
            ),
            "holdout": {"cost_per_applicant": None},
            "calibration": {},
            "money": {},
            "policy": {},
            "dataset": {},
            "split": {},
            "sensitive_feature_comparison": None,
        }
    )
    assert "_No comparison rows._" in empty_card
    assert "not available" in empty_card
    assert "Education" in empty_card


def test_model_card_markdown_copies_the_card_numbers():
    card = {
        "winner": {
            "display_name": "Logistic regression",
            "provisional_display_name": "Histogram gradient boosting",
            "selection_note": "The simpler model is deployed.",
            "holdout_roc_auc_ci": [0.66, 0.74],
        },
        "selection_rule": "Cross-validated ROC-AUC, then the bootstrap tie break.",
        "author": "Ladarius Jackson",
        "sklearn_version": "1.9.1",
        "positive_class": "fully paid",
        "feature_columns": ["fico", "purpose"],
        "positive_rate": 0.84,
        "policy": {
            "excluded_features": ["credit_policy"],
            "reason": "It is the old flag.",
            "fp_cost": 5000,
            "fn_cost": 600,
            "cost_source": "A planning scenario.",
        },
        "dataset": {
            "name": "Lending Club loan sample",
            "openml_id": 43729,
            "openml_url": "https://www.openml.org/d/43729",
            "license": "CC0: Public Domain",
            "fetched_with": "fetch_openml",
            "target_definition": "fully paid",
            "license_notes": "Public extract.",
        },
        "split": {
            "test_size": 0.2,
            "seed": 42,
            "n_train": 80,
            "n_test": 20,
            "cv_folds": 5,
            "search_iterations_requested": 16,
            "n_bootstrap": 1000,
        },
        "comparison": [
            {
                "display_name": "Logistic regression",
                "stage": "default",
                "cv_roc_auc_mean": 0.66,
                "cv_roc_auc_std": 0.02,
                "holdout_roc_auc": None,
            },
            {
                "display_name": "Logistic regression",
                "stage": "tuned",
                "cv_roc_auc_mean": 0.67,
                "cv_roc_auc_std": 0.01,
                "holdout_roc_auc": 0.70,
                "holdout_roc_auc_ci_low": 0.66,
                "holdout_roc_auc_ci_high": 0.74,
            },
        ],
        "bootstrap_comparisons": [
            {
                "sample": "out_of_fold",
                "reference_model": "hist_gradient_boosting",
                "other_model": "logistic_regression",
                "auc_difference": 0.004,
                "ci_low": -0.01,
                "ci_high": 0.02,
                "includes_zero": True,
                "used_for_selection": True,
            }
        ],
        "holdout": {
            "threshold": 0.64,
            "roc_auc": 0.704,
            "average_precision": 0.92,
            "f1": 0.409,
            "precision": 0.948,
            "recall": 0.261,
            "approval_rate": 0.231,
            "confusion_matrix": [[1, 2], [3, 4]],
            "tn": 1,
            "fp": 2,
            "fn": 3,
            "tp": 4,
            "n": 10,
            "brier": 0.12,
            "cost_per_applicant": 10,
        },
        "calibration": {
            "holdout_roc_auc_uncalibrated": 0.703,
            "holdout_brier_calibrated": 0.12,
            "holdout_brier_uncalibrated": 0.22,
            "holdout_brier_base_rate": 0.135,
        },
        "money": {
            "model_cost_total": 100,
            "approve_all_cost_total": 500,
            "decline_all_cost_total": 200,
            "better_naive_rule": "decline_all",
            "saved_total_vs_better_naive": 100,
            "saved_per_applicant_vs_better_naive": 10,
        },
        "sensitive_feature_comparison": {
            "cv_roc_auc_mean": 0.671,
            "cv_roc_auc_std": 0.01,
            "auc_difference_vs_policy": 0.001,
            "note": "Not used to choose.",
        },
        "ablation_feature_comparison": {},
        "fairness_oof": [
            {
                "attribute": "fico_band",
                "group": "Under 680",
                "n": 10,
                "base_rate": 0.7,
                "approval_rate": 0.1,
                "disparate_impact": 0.2,
            }
        ],
        "data_sha256": "abc",
    }
    text = render_model_card(card)
    assert "# Model card" in text
    assert "0.704" in text
    assert "0.220" in text or "0.22" in text
    assert "Ladarius Jackson" in text
    assert "not a FCRA" in text or "FCRA" in text
    assert "Under 680" in text
    assert "Not used to choose." in text
    assert "FICO removed" in text
