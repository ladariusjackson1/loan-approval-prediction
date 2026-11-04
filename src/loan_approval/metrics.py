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


def dollars(value: float) -> str:
    """Format a planning-cost figure the README and the tests both look for."""

    return f"{float(value):,.2f}"


def expected_cost(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> float:
    """Total planning cost. False approvals (not-fully-paid loans marked paid) use ``fp_cost``."""

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

    Ties break toward the higher cutoff, so a tie does not become a looser lending rule.
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


def money_saved(
    holdout_metrics: dict[str, object],
    baselines: dict[str, float],
) -> dict[str, float | str]:
    """Dollars versus the two rules that ignore the model.

    A negative save means the scored cutoff costs more than that rule on this book.
    The better naive rule is the cheaper of approve-everyone and decline-everyone.
    """

    per_applicant = float(holdout_metrics["cost_per_applicant"])
    n = int(holdout_metrics["n"])
    approve_all = float(baselines["approve_all"])
    decline_all = float(baselines["decline_all"])
    if decline_all < approve_all:
        better_name = "decline_all"
        better = decline_all
    elif approve_all < decline_all:
        better_name = "approve_all"
        better = approve_all
    else:
        better_name = "tie"
        better = approve_all
    saved_per = better - per_applicant
    return {
        "better_naive_rule": better_name,
        "better_naive_cost_per_applicant": better,
        "model_cost_per_applicant": per_applicant,
        "saved_per_applicant_vs_better_naive": saved_per,
        "saved_total_vs_better_naive": saved_per * n,
        "saved_total_vs_approve_all": (approve_all - per_applicant) * n,
        "saved_total_vs_decline_all": (decline_all - per_applicant) * n,
        "approve_all_cost_total": approve_all * n,
        "decline_all_cost_total": decline_all * n,
        "model_cost_total": per_applicant * n,
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

    F1 treats "fully paid / approve" as the positive class, which is the product
    decision. That class is the majority of this file, so F1 can look strong
    while the dollar weights still prefer a strict cutoff. Both numbers are
    reported for that reason.
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
