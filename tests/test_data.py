"""Dataset loading, target coding, and the segment frame."""

from __future__ import annotations

import pandas as pd
import pytest

from loan_approval.config import CATEGORY_OPTIONS, DATA_PATH, REQUIRED_COLUMNS, VALUE_LABELS
from loan_approval.data import (
    DatasetError,
    credit_policy_label,
    download_openml_loans,
    encode_target,
    feature_types,
    fico_band,
    load_loan_data,
    modeling_matrices,
    sensitive_frame,
)


def test_cached_sample_is_the_full_openml_extract():
    frame = load_loan_data(DATA_PATH, download=False)
    assert frame.shape[0] > 1000
    assert frame.shape == (9578, 14)
    assert list(frame.columns) == REQUIRED_COLUMNS
    assert set(frame["not_fully_paid"]) == {0, 1}
    assert int((frame["not_fully_paid"] == 0).sum()) == 8045
    assert int((frame["not_fully_paid"] == 1).sum()) == 1533
    assert "personal_status" not in frame.columns
    assert "class" not in frame.columns


def test_category_options_match_the_cached_file():
    frame = load_loan_data()
    for column, options in CATEGORY_OPTIONS.items():
        assert set(frame[column].dropna().astype(str)) == set(options)
        assert set(VALUE_LABELS[column]) == set(options)


def test_missing_cache_raises(tmp_path):
    with pytest.raises(DatasetError, match="download=True"):
        load_loan_data(tmp_path / "missing.csv", download=False)


def test_download_writes_a_schema_checked_csv(tmp_path, monkeypatch):
    def fake_download():
        dotted = pd.DataFrame(
            [
                {
                    "credit.policy": 1,
                    "purpose": "credit_card",
                    "int.rate": 0.11,
                    "installment": 220.0,
                    "log.annual.inc": 11.0,
                    "dti": 8.0,
                    "fico": 740,
                    "days.with.cr.line": 5000.0,
                    "revol.bal": 2000,
                    "revol.util": 15.0,
                    "inq.last.6mths": 0,
                    "delinq.2yrs": 0,
                    "pub.rec": 0,
                    "not.fully.paid": 0,
                    "extra": "ignore me",
                }
            ]
        )
        from loan_approval.data import OPENML_RENAMES

        return dotted.rename(columns=OPENML_RENAMES)

    monkeypatch.setattr("loan_approval.data.download_openml_loans", fake_download)
    destination = tmp_path / "lending_club_sample.csv"
    loaded = load_loan_data(destination, download=True)
    assert destination.exists()
    assert "extra" not in loaded.columns
    assert loaded.loc[0, "not_fully_paid"] == 0
    assert loaded.loc[0, "fico"] == 740
    assert loaded.loc[0, "purpose"] == "credit_card"


def test_bad_cache_and_bad_download_are_rejected(tmp_path, monkeypatch):
    broken = tmp_path / "broken.csv"
    broken.write_text("purpose,not_fully_paid\ndebt_consolidation,0\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing columns"):
        load_loan_data(broken, download=False)

    monkeypatch.setattr(
        "loan_approval.data.download_openml_loans",
        lambda: pd.DataFrame({"not_fully_paid": [0]}),
    )
    with pytest.raises(DatasetError, match="missing columns"):
        load_loan_data(tmp_path / "fresh.csv", download=True)


def test_openml_without_a_frame_raises(monkeypatch):
    class Empty:
        frame = None

    monkeypatch.setattr("sklearn.datasets.fetch_openml", lambda **kwargs: Empty())
    with pytest.raises(DatasetError, match="no dataframe"):
        download_openml_loans()


def test_target_and_feature_guards(synthetic_credit):
    encoded = encode_target(synthetic_credit["not_fully_paid"])
    assert set(encoded.unique()) <= {0, 1}
    assert encoded.loc[0] == 1
    assert encoded.loc[1] == 0
    with pytest.raises(DatasetError, match="Target must be"):
        encode_target(pd.Series([0, 2]))
    with pytest.raises(DatasetError, match="Unknown feature"):
        feature_types(["fico", "not_a_field"])
    broken = synthetic_credit.drop(columns=["int_rate"])
    with pytest.raises(DatasetError, match="Excluded columns"):
        modeling_matrices(broken, excluded=["int_rate"])


def test_segments_cover_fico_policy_and_purpose():
    assert fico_band(679) == "Under 680"
    assert fico_band(680) == "680–719"
    assert fico_band(720) == "720–759"
    assert fico_band(760) == "760 and over"
    assert fico_band(float("nan")) == "unknown"
    assert credit_policy_label(1) == "Meets policy"
    assert credit_policy_label(0) == "Outside policy"
    assert credit_policy_label(float("nan")) == "unknown"
    frame = pd.DataFrame(
        {
            "credit_policy": [1, 0],
            "fico": [650, 780],
            "purpose": ["small_business", "debt_consolidation"],
        }
    )
    sensitive = sensitive_frame(frame)
    assert list(sensitive["credit_policy"]) == ["Meets policy", "Outside policy"]
    assert list(sensitive["purpose"]) == ["Small business", "Debt consolidation"]
    with pytest.raises(DatasetError, match="segment frame"):
        sensitive_frame(frame.drop(columns=["fico"]))
