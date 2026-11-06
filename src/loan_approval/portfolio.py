"""Portfolio arithmetic for one cutoff on a scored book.

Two views are kept apart on purpose.

Expected figures use only the calibrated probability that a loan is fully paid.
They answer a planning question for a book that has not resolved yet.

Realized figures use the historical labels. They answer what this cutoff would
have done on the book that was scored. They are not a forecast.

Expected profit versus declining every loan, for each approved loan, is

    p * decline_cost - (1 - p) * approval_loss

where ``p`` is the calibrated probability the loan is fully paid, ``decline_cost``
is the planning weight on turning away a loan that would have been fully paid,
and ``approval_loss`` is the planning weight on funding a loan that is not fully
paid. Declined loans contribute zero under that baseline. The sum is the
expected reduction in planning cost versus declining the whole book. It is not
interest income and it is not a recovered charge-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loan_approval.config import FN_COST, FP_COST
from loan_approval.metrics import expected_cost


def portfolio_snapshot(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    threshold: float,
    *,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> dict[str, float | int | None]:
    """Approvals, expected defaults, and expected profit at one cutoff."""

    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    if actual.shape[0] != scores.shape[0]:
        raise ValueError("y_true and proba must have the same length.")
    approved = scores >= float(threshold)
    n = int(actual.shape[0])
    n_approved = int(np.sum(approved))
    approved_scores = scores[approved]
    approved_actual = actual[approved]
    realized_defaults = int(np.sum(approved_actual == 0))
    realized_fully_paid = int(np.sum(approved_actual == 1))
    if n_approved:
        expected_defaults = float(np.sum(1.0 - approved_scores))
        expected_profit = float(
            np.sum(approved_scores * fn_cost - (1.0 - approved_scores) * fp_cost)
        )
    else:
        expected_defaults = 0.0
        expected_profit = 0.0
    decisions = approved.astype(int)
    realized_cost = expected_cost(actual, decisions, fp_cost=fp_cost, fn_cost=fn_cost)
    decline_all = expected_cost(actual, np.zeros(n, dtype=int), fp_cost=fp_cost, fn_cost=fn_cost)
    approve_all = expected_cost(actual, np.ones(n, dtype=int), fp_cost=fp_cost, fn_cost=fn_cost)
    return {
        "n": n,
        "threshold": float(threshold),
        "approvals": n_approved,
        "approval_rate": (n_approved / n) if n else None,
        "expected_defaults": expected_defaults,
        "expected_default_rate": (expected_defaults / n_approved) if n_approved else None,
        "expected_profit_vs_decline_all": expected_profit,
        "realized_defaults": realized_defaults,
        "realized_fully_paid": realized_fully_paid,
        "realized_default_rate": (realized_defaults / n_approved) if n_approved else None,
        "realized_planning_cost": realized_cost,
        "realized_savings_vs_decline_all": decline_all - realized_cost,
        "decline_all_cost": decline_all,
        "approve_all_cost": approve_all,
    }


def portfolio_curve(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    grid: np.ndarray | None = None,
    *,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> pd.DataFrame:
    """One snapshot per cutoff, for the simulator chart."""

    if grid is None:
        grid = np.round(np.linspace(0.05, 0.95, 91), 4)
    rows = [
        portfolio_snapshot(y_true, proba, float(threshold), fp_cost=fp_cost, fn_cost=fn_cost)
        for threshold in np.asarray(grid, dtype=float)
    ]
    return pd.DataFrame(rows)
