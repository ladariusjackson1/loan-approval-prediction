"""Cost threshold, classification metrics, and fairness arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from loan_approval.fairness import binary_rates, fairness_table
from loan_approval.metrics import (
    choose_threshold,
    classification_metrics,
    expected_cost,
    policy_baselines,
)


def test_statlog_cost_counts_false_approvals_five_times():
    actual = np.array([1, 1, 0, 0])
    predicted = np.array([1, 0, 1, 0])
    assert expected_cost(actual, predicted) == 6.0
    baselines = policy_baselines(actual)
    assert baselines["approve_all"] == 2.5
    assert baselines["decline_all"] == 0.5


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


def test_fairness_gaps_use_the_reference_group_and_fall_back_to_the_largest():
    actual = np.array([1, 1, 0, 0, 1, 0])
    predicted = np.array([1, 0, 1, 0, 1, 0])
    sensitive = pd.DataFrame(
        {
            "sex": ["male", "male", "female", "female", "female", "female"],
            "age_band": ["25 and over"] * 6,
            "foreign_worker": ["no"] * 6,
        }
    )
    table = fairness_table(actual, predicted, sensitive)
    male = table.loc[(table["attribute"] == "sex") & (table["group"] == "male")].iloc[0]
    female = table.loc[(table["attribute"] == "sex") & (table["group"] == "female")].iloc[0]
    assert male["demographic_parity_gap"] == 0
    assert female["reference_group"] == "male"
    assert female["disparate_impact"] == female["approval_rate"] / male["approval_rate"]
    assert female["outside_four_fifths"] in {True, False}

    only_women = pd.DataFrame({"sex": ["female", "female"]})
    fallback = fairness_table(np.array([1, 0]), np.array([1, 0]), only_women)
    assert fallback.iloc[0]["reference_group"] == "female"


def test_zero_approval_reference_has_no_disparate_impact_ratio():
    sensitive = pd.DataFrame({"foreign_worker": ["no", "no", "yes"]})
    table = fairness_table(np.array([1, 0, 1]), np.array([0, 0, 1]), sensitive)
    minority = table.loc[table["group"] == "yes"].iloc[0]
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
