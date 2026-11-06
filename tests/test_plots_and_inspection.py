"""Report figures and the global explanation tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loan_approval.inspection import coefficient_frame, permutation_frame, pretty_transformed_name
from loan_approval.plots import (
    contribution_figure,
    plot_bootstrap_intervals,
    plot_calibration,
    plot_class_balance,
    plot_coefficients,
    plot_confusion,
    plot_fairness,
    plot_importance,
    plot_model_comparison,
    plot_roc_curves,
    plot_threshold_cost,
)


def _comparison_rows() -> list[dict[str, object]]:
    rows = []
    for name, default, tuned in (
        ("logistic_regression", 0.70, 0.74),
        ("hist_gradient_boosting", 0.72, 0.76),
    ):
        for stage, score in (("default", default), ("tuned", tuned)):
            rows.append(
                {
                    "model": name,
                    "stage": stage,
                    "cv_roc_auc_mean": score,
                    "cv_roc_auc_std": 0.02,
                    "holdout_roc_auc": score,
                    "holdout_roc_auc_ci_low": score - 0.02,
                    "holdout_roc_auc_ci_high": score + 0.02,
                }
            )
    return rows


def test_every_report_figure_writes_a_png(tmp_path):
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    proba = np.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7, 0.9, 0.1])
    plot_class_balance(7, 3, tmp_path / "balance.png")
    plot_model_comparison(
        _comparison_rows(), tmp_path / "compare.png", winner="hist_gradient_boosting"
    )
    plot_bootstrap_intervals(
        _comparison_rows(), tmp_path / "bootstrap.png", winner="hist_gradient_boosting"
    )
    missing_interval = _comparison_rows()
    for row in missing_interval:
        row["holdout_roc_auc_ci_low"] = None
        row["holdout_roc_auc_ci_high"] = None
    plot_bootstrap_intervals(
        missing_interval, tmp_path / "bootstrap_open.png", winner="logistic_regression"
    )
    plot_roc_curves(
        [{"model": "logistic_regression", "y_true": y_true, "proba": proba, "roc_auc": 0.8}],
        tmp_path / "roc.png",
    )
    plot_confusion([[4, 1], [1, 4]], tmp_path / "cm.png")
    plot_threshold_cost(
        np.linspace(0.1, 0.9, 5),
        np.array([0.8, 0.5, 0.4, 0.6, 0.9]),
        0.5,
        {"approve_all": 1.2, "decline_all": 0.7},
        tmp_path / "cost.png",
    )
    importance = pd.DataFrame(
        {
            "label": ["Checking account status", "Age (years)"],
            "importance_mean": [0.08, 0.02],
            "importance_std": [0.01, 0.01],
        }
    )
    plot_importance(importance, tmp_path / "importance.png")
    coefficients = pd.DataFrame(
        {
            "feature": ["Checking account status = No checking account", "Age (years)"],
            "coefficient": [0.4, -0.2],
        }
    )
    plot_coefficients(coefficients, tmp_path / "coef.png")
    fairness = pd.DataFrame(
        {
            "attribute": ["sex", "sex", "age_band", "age_band", "foreign_worker", "foreign_worker"],
            "group": ["female", "male", "Under 25", "25 and over", "yes", "no"],
            "n": [40, 60, 20, 80, 10, 90],
            "base_rate": [0.6, 0.7, 0.5, 0.72, 0.4, 0.7],
            "approval_rate": [0.5, 0.75, 0.4, 0.7, 0.3, 0.72],
        }
    )
    plot_fairness(fairness, tmp_path / "fair.png")
    plot_fairness(pd.DataFrame(), tmp_path / "fair_empty.png")
    plot_fairness(
        pd.DataFrame({"attribute": ["not_tracked"], "group": ["a"]}), tmp_path / "fair_unknown.png"
    )
    plot_calibration(y_true, proba, tmp_path / "calibration.png", uncalibrated=proba)
    plot_calibration(
        np.array([0, 1] * 10),
        np.full(20, 0.4),
        tmp_path / "calibration_flat.png",
        uncalibrated=np.linspace(0.1, 0.9, 20),
    )
    plot_calibration(
        np.array([1, 1, 1, 0]), np.array([0.2, 0.3, 0.4, 0.5]), tmp_path / "calibration_small.png"
    )
    for path in tmp_path.glob("*.png"):
        assert path.stat().st_size > 0


def test_contribution_figure_handles_an_empty_reason_list():
    figure = contribution_figure([])
    assert figure.axes
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_transformed_names_and_coefficient_guard():
    assert pretty_transformed_name("numeric__fico") == "FICO score"
    assert "Small business" in pretty_transformed_name("categorical__purpose_small_business")
    assert pretty_transformed_name("other") == "other"
    pipeline = type("Bare", (), {})()
    with pytest.raises(TypeError, match="linear classifier"):
        coefficient_frame(pipeline)


def test_permutation_importance_ranks_a_useful_column():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=80)
    frame = pd.DataFrame({"signal": signal, "noise": rng.normal(size=80)})
    target = pd.Series((signal > 0).astype(int))
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression().fit(frame, target)
    ranked = permutation_frame(model, frame, target, repeats=5, seed=1, n_jobs=1)
    assert ranked.iloc[0]["feature"] == "signal"
    assert ranked.iloc[0]["importance_mean"] > ranked.iloc[1]["importance_mean"]
