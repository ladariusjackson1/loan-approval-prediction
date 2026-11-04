"""Threshold-free and cost-sensitive scores for an approve / decline decision."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from loan_approval.config import FN_COST, FP_COST


def expected_cost(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> float:
    """Total Statlog cost. False approvals (bad credits marked good) use ``fp_cost``."""

    actual = np.asarray(y_true).astype(int)
    predicted = np.asarray(y_pred).astype(int)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    false_approvals = int(np.sum((predicted == 1) & (actual == 0)))
    false_declines = int(np.sum((predicted == 0) & (actual == 1)))
    return float(fp_cost * false_approvals + fn_cost * false_declines)


def choose_threshold(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    *,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
    grid: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Pick the cutoff that minimizes average cost on out-of-fold scores.

    Ties break toward the higher cutoff. Under a 5:1 loss, a tie should not
    become a looser lending rule.
    """

    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    if actual.shape[0] != scores.shape[0]:
        raise ValueError("y_true and proba must have the same length.")
    if grid is None:
        grid = np.round(np.linspace(0.05, 0.95, 91), 4)
    else:
        grid = np.asarray(grid, dtype=float)
    costs = np.empty(len(grid), dtype=float)
    n = max(len(actual), 1)
    for index, threshold in enumerate(grid):
        predicted = (scores >= threshold).astype(int)
        costs[index] = expected_cost(actual, predicted, fp_cost=fp_cost, fn_cost=fn_cost) / n
    minimum = float(costs.min())
    candidates = grid[np.isclose(costs, minimum)]
    return float(np.max(candidates)), grid, costs


def policy_baselines(
    y_true: np.ndarray | pd.Series,
    *,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> dict[str, float]:
    """Cost per applicant of the two rules that ignore the model."""

    actual = np.asarray(y_true).astype(int)
    n = max(len(actual), 1)
    approve_all = np.ones(len(actual), dtype=int)
    decline_all = np.zeros(len(actual), dtype=int)
    return {
        "approve_all": expected_cost(actual, approve_all, fp_cost=fp_cost, fn_cost=fn_cost) / n,
        "decline_all": expected_cost(actual, decline_all, fp_cost=fp_cost, fn_cost=fn_cost) / n,
    }


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    threshold: float,
    *,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> dict[str, object]:
    """ROC-AUC and F1 plus the confusion matrix at a frozen threshold.

    F1 treats "good credit / approve" as the positive class, which is what the
    product decision is. Because that class is 70% of the file, F1 can look
    strong while the cost matrix still prefers a strict cutoff. Both numbers
    are reported for that reason.
    """

    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    true_neg, false_pos, false_neg, true_pos = (int(value) for value in matrix.ravel())
    negatives = true_neg + false_pos
    total_cost = expected_cost(actual, predicted, fp_cost=fp_cost, fn_cost=fn_cost)
    n = int(len(actual))
    return {
        "roc_auc": float(roc_auc_score(actual, scores)),
        "average_precision": float(average_precision_score(actual, scores)),
        "brier": float(brier_score_loss(actual, scores)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "specificity": float(true_neg / negatives) if negatives else None,
        "bad_credit_capture": float(true_neg / negatives) if negatives else None,
        "approval_rate": float(predicted.mean()) if n else None,
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "tn": true_neg,
        "fp": false_pos,
        "fn": false_neg,
        "tp": true_pos,
        "cost_total": total_cost,
        "cost_per_applicant": total_cost / n if n else None,
        "n": n,
    }
