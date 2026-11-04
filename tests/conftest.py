"""Shared fixtures. Matplotlib stays headless for the whole suite."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_credit() -> pd.DataFrame:
    """Small table with the Lending Club schema and both repayment outcomes."""

    rng = np.random.default_rng(0)
    n = 80
    purposes = np.array(
        ["debt_consolidation", "credit_card", "all_other", "small_business", "educational"]
    )
    frame = pd.DataFrame(
        {
            "credit_policy": rng.integers(0, 2, n).astype(float),
            "purpose": rng.choice(purposes, n),
            "int_rate": rng.uniform(0.07, 0.2, n),
            "installment": rng.uniform(80, 700, n),
            "log_annual_inc": np.log(rng.uniform(25000, 120000, n)),
            "dti": rng.uniform(2, 28, n),
            "fico": rng.integers(630, 790, n).astype(float),
            "days_with_cr_line": rng.uniform(400, 9000, n),
            "revol_bal": rng.integers(0, 40000, n).astype(float),
            "revol_util": rng.uniform(0, 100, n),
            "inq_last_6mths": rng.integers(0, 8, n).astype(float),
            "delinq_2yrs": rng.integers(0, 3, n).astype(float),
            "pub_rec": rng.integers(0, 2, n).astype(float),
        }
    )
    frame["not_fully_paid"] = 0
    frame.loc[frame.index % 5 == 0, "not_fully_paid"] = 1
    frame.loc[frame["not_fully_paid"] == 1, "fico"] = 640
    frame.loc[frame["not_fully_paid"] == 0, "fico"] = 740
    frame.loc[0, "not_fully_paid"] = 0
    frame.loc[1, "not_fully_paid"] = 1
    return frame
