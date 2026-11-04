"""Dataset loading, target coding, and the sensitive-attribute frame."""

from __future__ import annotations

import pandas as pd
import pytest

from loan_approval.config import CATEGORY_OPTIONS, DATA_PATH, REQUIRED_COLUMNS, VALUE_LABELS
from loan_approval.data import (
    DatasetError,
    age_band,
    download_openml_credit,
    encode_target,
    feature_types,
    load_credit_data,
    modeling_matrices,
    sensitive_frame,
    sex_from_personal_status,
)


def test_cached_credit_table_matches_the_openml_contract():
    frame = load_credit_data(DATA_PATH, download=False)
    assert frame.shape == (1000, 21)
    assert list(frame.columns) == REQUIRED_COLUMNS
    assert set(frame["class"]) == {"good", "bad"}
    assert frame["class"].value_counts()["good"] == 700


def test_category_options_match_the_cached_file():
    frame = load_credit_data()
    for column, options in CATEGORY_OPTIONS.items():
        assert set(frame[column].dropna().astype(str)) == set(options)
        assert set(VALUE_LABELS[column]) == set(options)


def test_missing_cache_raises(tmp_path):
    with pytest.raises(DatasetError, match="download=True"):
        load_credit_data(tmp_path / "missing.csv", download=False)


def test_download_writes_a_schema_checked_csv(tmp_path, monkeypatch):
    raw = pd.DataFrame(
        {
            "checking_status": ["no checking"],
            "duration": [12],
            "credit_history": ["existing paid"],
            "purpose": ["radio/tv"],
            "credit_amount": [1000],
            "savings_status": ["<100"],
            "employment": [">=7"],
            "installment_commitment": [2],
            "personal_status": ["male single"],
            "other_parties": ["none"],
            "residence_since": [2],
            "property_magnitude": ["car"],
            "age": [30],
            "other_payment_plans": ["none"],
            "housing": ["own"],
            "existing_credits": [1],
            "job": ["skilled"],
            "num_dependents": [1],
            "own_telephone": ["yes"],
            "foreign_worker": ["yes"],
            "class": ["good"],
            "extra": ["ignore me"],
        }
    )

    def fake_download():
        return raw

    monkeypatch.setattr("loan_approval.data.download_openml_credit", fake_download)
    destination = tmp_path / "credit-g.csv"
    loaded = load_credit_data(destination, download=True)
    assert destination.exists()
    assert "extra" not in loaded.columns
    assert loaded.loc[0, "class"] == "good"
    assert loaded.loc[0, "age"] == 30


def test_bad_cache_and_bad_download_are_rejected(tmp_path, monkeypatch):
    broken = tmp_path / "broken.csv"
    broken.write_text("checking_status,class\nno checking,good\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing columns"):
        load_credit_data(broken, download=False)

    monkeypatch.setattr(
        "loan_approval.data.download_openml_credit",
        lambda: pd.DataFrame({"class": ["good"]}),
    )
    with pytest.raises(DatasetError, match="missing columns"):
        load_credit_data(tmp_path / "fresh.csv", download=True)


def test_openml_without_a_frame_raises(monkeypatch):
    class Empty:
        frame = None

    monkeypatch.setattr("sklearn.datasets.fetch_openml", lambda **kwargs: Empty())
    with pytest.raises(DatasetError, match="no dataframe"):
        download_openml_credit()


def test_target_and_feature_guards(synthetic_credit):
    encoded = encode_target(synthetic_credit["class"])
    assert set(encoded.unique()) <= {0, 1}
    assert encoded.loc[0] == 1
    with pytest.raises(DatasetError, match="Unknown class"):
        encode_target(pd.Series(["good", "maybe"]))
    with pytest.raises(DatasetError, match="Unknown feature"):
        feature_types(["duration", "not_a_field"])
    broken = synthetic_credit.drop(columns=["foreign_worker"])
    with pytest.raises(DatasetError, match="Excluded columns"):
        modeling_matrices(broken, excluded=["foreign_worker"])


def test_sensitive_attributes_cover_the_awkward_coding():
    assert sex_from_personal_status("male single") == "male"
    assert sex_from_personal_status("female div/dep/mar") == "female"
    assert sex_from_personal_status("other") == "unknown"
    assert age_band(24) == "Under 25"
    assert age_band(25) == "25 and over"
    assert age_band(float("nan")) == "unknown"
    frame = pd.DataFrame(
        {
            "personal_status": ["male mar/wid", "female div/dep/mar"],
            "age": [22, 40],
            "foreign_worker": ["yes", "no"],
        }
    )
    sensitive = sensitive_frame(frame)
    assert list(sensitive["sex"]) == ["male", "female"]
    with pytest.raises(DatasetError, match="fairness frame"):
        sensitive_frame(frame.drop(columns=["age"]))
