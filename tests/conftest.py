"""Shared fixtures. Matplotlib stays headless for the whole suite."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_credit() -> pd.DataFrame:
    """Small table with the real schema and both outcomes."""

    rng = np.random.default_rng(0)
    n = 80
    frame = pd.DataFrame(
        {
            "checking_status": rng.choice(["<0", "0<=X<200", ">=200", "no checking"], n),
            "duration": rng.integers(6, 48, n),
            "credit_history": rng.choice(
                ["existing paid", "critical/other existing credit", "delayed previously"],
                n,
            ),
            "purpose": rng.choice(["radio/tv", "new car", "education", "business"], n),
            "credit_amount": rng.integers(500, 8000, n),
            "savings_status": rng.choice(["<100", ">=1000", "no known savings"], n),
            "employment": rng.choice(["<1", "1<=X<4", ">=7", "unemployed"], n),
            "installment_commitment": rng.integers(1, 5, n),
            "personal_status": rng.choice(
                ["male single", "female div/dep/mar", "male mar/wid", "male div/sep"],
                n,
            ),
            "other_parties": rng.choice(["none", "guarantor", "co applicant"], n),
            "residence_since": rng.integers(1, 5, n),
            "property_magnitude": rng.choice(
                ["real estate", "car", "no known property", "life insurance"],
                n,
            ),
            "age": rng.integers(19, 70, n),
            "other_payment_plans": rng.choice(["none", "bank", "stores"], n),
            "housing": rng.choice(["own", "rent", "for free"], n),
            "existing_credits": rng.integers(1, 4, n),
            "job": rng.choice(
                ["skilled", "unskilled resident", "high qualif/self emp/mgmt"],
                n,
            ),
            "num_dependents": rng.integers(1, 3, n),
            "own_telephone": rng.choice(["none", "yes"], n),
            "foreign_worker": rng.choice(["yes", "no"], n),
        }
    )
    signal = (
        frame["checking_status"].eq("no checking").astype(int)
        + (frame["age"] > 30).astype(int)
        - (frame["credit_amount"] > 5000).astype(int)
    )
    frame["class"] = np.where(signal.to_numpy() > 0, "good", "bad")
    frame.loc[0, "class"] = "good"
    frame.loc[1, "class"] = "bad"
    return frame
