"""End-to-end training on a toy table, artifact IO, and the shipped model card."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from loan_approval.artifacts import (
    file_sha256,
    json_safe,
    load_bundle,
    load_model_card,
    save_bundle,
    save_model_card,
)
from loan_approval.config import DATA_PATH, MODEL_CARD_PATH, MODEL_PATH
from loan_approval.data import DatasetError, load_loan_data
from loan_approval.explain import explain_application
from loan_approval.metrics import dollars
from loan_approval.train import main
from loan_approval.training import ExperimentConfig, _select_winner, run_experiment


def test_winner_tie_breaks_toward_the_simpler_model():
    rows = [
        {
            "model": "hist_gradient_boosting",
            "stage": "tuned",
            "cv_roc_auc_mean": 0.8,
            "cv_roc_auc_std": 0.01,
        },
        {
            "model": "logistic_regression",
            "stage": "tuned",
            "cv_roc_auc_mean": 0.8,
            "cv_roc_auc_std": 0.01,
        },
        {
            "model": "logistic_regression",
            "stage": "default",
            "cv_roc_auc_mean": 0.99,
            "cv_roc_auc_std": 0.0,
        },
    ]
    assert _select_winner(rows) == "logistic_regression"


def test_run_experiment_rejects_a_single_class(synthetic_credit):
    frame = synthetic_credit.copy()
    frame["not_fully_paid"] = 0
    with pytest.raises(DatasetError, match="Both fully paid and not fully paid"):
        run_experiment(frame, ExperimentConfig(cv_folds=2, n_iter=1, n_jobs=1))


def test_cli_trains_a_toy_file_and_writes_figures(tmp_path, synthetic_credit):
    data_path = tmp_path / "lending_club_sample.csv"
    synthetic_credit.to_csv(data_path, index=False)
    figure_dir = tmp_path / "figures"
    main(
        [
            "--data",
            str(data_path),
            "--model-path",
            str(tmp_path / "model.joblib"),
            "--card-path",
            str(tmp_path / "card.json"),
            "--figure-dir",
            str(figure_dir),
            "--n-iter",
            "1",
            "--cv",
            "2",
            "--jobs",
            "1",
            "--seed",
            "0",
            "--bootstrap",
            "200",
            "--card-md",
            str(tmp_path / "MODEL_CARD.md"),
        ]
    )
    bundle = load_bundle(tmp_path / "model.joblib")
    card = load_model_card(tmp_path / "card.json")
    assert bundle["model_name"] == card["winner"]["model"]
    assert card["data_sha256"] == file_sha256(data_path)
    assert {path.name for path in figure_dir.glob("*.png")} == {
        "class_balance.png",
        "model_comparison.png",
        "roc_curves.png",
        "confusion_matrix.png",
        "threshold_cost.png",
        "feature_importance.png",
        "logistic_coefficients.png",
        "fairness.png",
        "calibration.png",
        "bootstrap_intervals.png",
    }
    assert (tmp_path / "MODEL_CARD.md").is_file()
    payload = {column: synthetic_credit.iloc[0][column] for column in bundle["feature_columns"]}
    explained = explain_application(
        bundle["pipeline"],
        payload,
        reference=bundle["reference"],
        feature_columns=bundle["feature_columns"],
        threshold=bundle["threshold"],
        ranges=bundle["ranges"],
    )
    assert 0 <= explained["probability_good"] <= 1
    assert explained["decision"] in {"Approve", "Decline"}


def test_artifact_helpers_reject_bad_bundles_and_sanitize_json(tmp_path):
    with pytest.raises(KeyError, match="missing keys"):
        save_bundle(tmp_path / "partial.joblib", {"pipeline": object()})
        load_bundle(tmp_path / "partial.joblib")
    joblib.dump(["not", "a", "dict"], tmp_path / "list.joblib")
    with pytest.raises(TypeError, match="dictionary"):
        load_bundle(tmp_path / "list.joblib")

    frame = pd.DataFrame({"a": [np.float64(1.5)], "b": [np.int64(2)]})
    payload = {
        "nan": np.float64("nan"),
        "inf": float("inf"),
        "flag": np.True_,
        "frame": frame,
        "series": frame["a"],
        "missing": pd.NA,
    }
    safe = json_safe(payload)
    assert safe["nan"] is None
    assert safe["inf"] is None
    assert safe["flag"] is True
    assert safe["frame"][0]["b"] == 2
    assert safe["missing"] is None
    save_model_card(tmp_path / "card.json", payload)
    reloaded = json.loads((tmp_path / "card.json").read_text(encoding="utf-8"))
    assert (
        reloaded["series"]["0"] == 1.5
        or reloaded["series"][0] == 1.5
        or "1.5" in json.dumps(reloaded)
    )


def test_repo_is_ready_for_streamlit_community_cloud():
    root = DATA_PATH.parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    pinned = [line.strip() for line in requirements.splitlines() if line.strip()]
    assert pinned
    assert all("==" in line and not line.startswith("-r") for line in pinned)
    assert "pytest" not in requirements
    dev = (root / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "-r requirements.txt" in dev
    assert "pytest==" in dev
    assert (root / "runtime.txt").read_text(encoding="utf-8").strip() == "python-3.12"
    config = (root / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "[theme]" in config
    entry = (root / "streamlit_app.py").read_text(encoding="utf-8")
    assert "app.streamlit_app" in entry
    assert "streamlit run streamlit_app.py" in entry
    readme = (root / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    assert "live demo" not in lowered
    assert "live%20demo" not in lowered
    assert "https://loan-approval-prediction.streamlit.app" in readme
    assert "sign-in" in lowered
    assert "access is pending" in lowered
    assert "not publicly open" in lowered
    assert "streamlit run streamlit_app.py" in readme
    assert "docs/demo.gif" in readme
    assert (root / "docs" / "demo.gif").is_file()
    assert (root / "MODEL_CARD.md").is_file()


def test_shipped_model_matches_the_readme_and_explains_a_real_row():
    card = load_model_card(MODEL_CARD_PATH)
    bundle = load_bundle(MODEL_PATH)
    root = DATA_PATH.parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    holdout = card["holdout"]
    money = card["money"]
    assert bundle["model_name"] == card["winner"]["model"]
    assert holdout["roc_auc"] > 0.6
    assert holdout["n"] > 1000
    assert sum(sum(row) for row in holdout["confusion_matrix"]) == holdout["n"]
    assert f"{holdout['roc_auc']:.3f}" in readme
    assert f"{holdout['f1']:.3f}" in readme
    assert f"{holdout['threshold']:.2f}" in readme
    assert dollars(float(holdout["cost_per_applicant"])) in readme
    assert dollars(float(money["saved_total_vs_better_naive"])) in readme
    assert dollars(float(money["model_cost_total"])) in readme
    assert f"{holdout['brier']:.3f}" in readme
    low, high = card["winner"]["holdout_roc_auc_ci"]
    assert f"{float(low):.3f}" in readme
    assert f"{float(high):.3f}" in readme
    assert f"{float(card['calibration']['holdout_brier_uncalibrated']):.3f}" in readme
    assert f"{float(card['calibration']['holdout_brier_base_rate']):.3f}" in readme
    card_md = (root / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert f"{holdout['roc_auc']:.3f}" in card_md
    assert card["winner"]["display_name"] in readme
    assert card["winner"]["display_name"] in card_md
    assert "43729" in readme
    assert "CC0" in readme
    assert card["data_sha256"] == file_sha256(DATA_PATH)
    assert "credit_policy" not in bundle["feature_columns"]
    assert "int_rate" not in bundle["feature_columns"]
    assert bundle["feature_columns"]

    row = load_loan_data().iloc[0]
    payload = {column: row[column] for column in bundle["feature_columns"]}
    explained = explain_application(
        bundle["pipeline"],
        payload,
        reference=bundle["reference"],
        feature_columns=list(bundle["feature_columns"]),
        threshold=float(bundle["threshold"]),
        ranges=bundle["ranges"],
        calibrator=bundle.get("calibrator"),
    )
    assert "counterfactual" in explained
    assert explained["counterfactual"]["text"]
    assert explained["decision"] in {"Approve", "Decline"}
    assert "fully paid" in explained["narrative"]
    assert "pushed" in explained["narrative"]
    assert len(explained["contributions"]) == len(bundle["feature_columns"])
