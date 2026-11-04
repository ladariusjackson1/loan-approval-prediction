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
from loan_approval.data import DatasetError, load_credit_data
from loan_approval.explain import explain_application
from loan_approval.train import main
from loan_approval.training import ExperimentConfig, _select_winner, run_experiment


def test_winner_tie_breaks_toward_the_simpler_model():
    rows = [
        {
            "model": "gradient_boosting",
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
            "model": "random_forest",
            "stage": "default",
            "cv_roc_auc_mean": 0.99,
            "cv_roc_auc_std": 0.0,
        },
    ]
    assert _select_winner(rows) == "logistic_regression"


def test_run_experiment_rejects_a_single_class(synthetic_credit):
    frame = synthetic_credit.copy()
    frame["class"] = "good"
    with pytest.raises(DatasetError, match="Both good and bad"):
        run_experiment(frame, ExperimentConfig(cv_folds=2, n_iter=1, n_jobs=1))


def test_cli_trains_a_toy_file_and_writes_figures(tmp_path, synthetic_credit):
    data_path = tmp_path / "credit-g.csv"
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
    }
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


def test_shipped_model_matches_the_readme_and_scores_a_real_row():
    card = load_model_card(MODEL_CARD_PATH)
    bundle = load_bundle(MODEL_PATH)
    readme = (DATA_PATH.parents[1] / "README.md").read_text(encoding="utf-8")
    holdout = card["holdout"]
    assert bundle["model_name"] == card["winner"]["model"]
    assert holdout["roc_auc"] > 0.65
    assert holdout["n"] == 200
    assert sum(sum(row) for row in holdout["confusion_matrix"]) == 200
    assert f"{holdout['roc_auc']:.3f}" in readme
    assert f"{holdout['f1']:.3f}" in readme
    assert card["winner"]["display_name"] in readme
    assert card["data_sha256"] == file_sha256(DATA_PATH)
    assert "personal_status" not in bundle["feature_columns"]
    assert "foreign_worker" not in bundle["feature_columns"]

    row = load_credit_data().iloc[0]
    payload = {column: row[column] for column in bundle["feature_columns"]}
    explained = explain_application(
        bundle["pipeline"],
        payload,
        reference=bundle["reference"],
        feature_columns=list(bundle["feature_columns"]),
        threshold=float(bundle["threshold"]),
        ranges=bundle["ranges"],
    )
    assert explained["decision"] in {"Approve", "Decline"}
    assert len(explained["contributions"]) == len(bundle["feature_columns"])
