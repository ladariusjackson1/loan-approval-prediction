"""Cost threshold, classification metrics, and fairness arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from loan_approval.fairness import binary_rates, fairness_table
from loan_approval.metrics import (
    choose_threshold,
    classification_metrics,
    dollars,
    expected_cost,
    money_saved,
    policy_baselines,
)


def test_planning_cost_prices_a_false_approval_at_five_thousand_dollars():
    actual = np.array([1, 1, 0, 0])
    predicted = np.array([1, 0, 1, 0])
    assert expected_cost(actual, predicted) == 5600.0
    baselines = policy_baselines(actual)
    assert baselines["approve_all"] == 2500.0
    assert baselines["decline_all"] == 300.0
    assert expected_cost(actual, predicted, fp_cost=5, fn_cost=1) == 6.0


def test_threshold_ties_break_toward_the_stricter_cutoff():
    actual = np.array([0, 1])
    scores = np.array([0.2, 0.9])
    chosen, grid, costs = choose_threshold(actual, scores)
    assert chosen == 0.9
    assert len(grid) == len(costs)
    assert costs.min() == 0


def test_classification_metrics_report_the_confusion_matrix():
    actual = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2])
    metrics = classification_metrics(actual, scores, threshold=0.65)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 0
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert 0 < metrics["roc_auc"] <= 1
    assert metrics["f1"] == 0.8


def test_money_saved_uses_the_cheaper_rule_that_ignores_the_model():
    metrics = {"cost_per_applicant": 400.0, "n": 10}
    saved = money_saved(metrics, {"approve_all": 800.0, "decline_all": 500.0})
    assert saved["better_naive_rule"] == "decline_all"
    assert saved["saved_per_applicant_vs_better_naive"] == 100.0
    assert saved["saved_total_vs_better_naive"] == 1000.0
    assert saved["saved_total_vs_approve_all"] == 4000.0
    approved = money_saved(
        {"cost_per_applicant": 100.0, "n": 4},
        {"approve_all": 200.0, "decline_all": 900.0},
    )
    assert approved["better_naive_rule"] == "approve_all"
    assert approved["saved_total_vs_better_naive"] == 400.0
    worse = money_saved(
        {"cost_per_applicant": 900.0, "n": 2},
        {"approve_all": 700.0, "decline_all": 700.0},
    )
    assert worse["better_naive_rule"] == "tie"
    assert worse["saved_total_vs_better_naive"] == -400.0
    assert dollars(1000) == "1,000.00"


def test_fairness_gaps_use_the_reference_group_and_fall_back_to_the_largest():
    actual = np.array([1, 1, 0, 0, 1, 0])
    predicted = np.array([1, 0, 1, 0, 1, 0])
    sensitive = pd.DataFrame(
        {
            "credit_policy": [
                "Meets policy",
                "Meets policy",
                "Outside policy",
                "Outside policy",
                "Outside policy",
                "Outside policy",
            ],
            "fico_band": ["720–759"] * 6,
        }
    )
    table = fairness_table(actual, predicted, sensitive)
    meets = table.loc[
        (table["attribute"] == "credit_policy") & (table["group"] == "Meets policy")
    ].iloc[0]
    outside = table.loc[
        (table["attribute"] == "credit_policy") & (table["group"] == "Outside policy")
    ].iloc[0]
    assert meets["demographic_parity_gap"] == 0
    assert outside["reference_group"] == "Meets policy"
    assert outside["disparate_impact"] == outside["approval_rate"] / meets["approval_rate"]
    assert outside["outside_four_fifths"] in {True, False}

    only_education = pd.DataFrame({"purpose": ["Education", "Education"]})
    fallback = fairness_table(np.array([1, 0]), np.array([1, 0]), only_education)
    assert fallback.iloc[0]["reference_group"] == "Education"


def test_zero_approval_reference_has_no_disparate_impact_ratio():
    sensitive = pd.DataFrame({"credit_policy": ["Meets policy", "Meets policy", "Outside policy"]})
    table = fairness_table(np.array([1, 0, 1]), np.array([0, 0, 1]), sensitive)
    minority = table.loc[table["group"] == "Outside policy"].iloc[0]
    assert minority["disparate_impact"] is None
    assert minority["outside_four_fifths"] is None


def test_binary_rates_are_none_when_a_denominator_is_zero():
    rates = binary_rates(np.array([1, 1]), np.array([0, 0]))
    assert rates["fpr"] is None
    assert rates["tpr"] == 0
    empty = binary_rates(np.array([]), np.array([]))
    assert empty["approval_rate"] is None


def test_empty_sensitive_frame_returns_an_empty_table():
    sensitive = pd.DataFrame({"sex": pd.Series(dtype=object)})
    table = fairness_table(np.array([]), np.array([]), sensitive)
    assert table.empty
