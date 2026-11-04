"""Figures written to reports/figures and the explanation chart used by the app."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from loan_approval.config import ATTRIBUTE_LABELS, DISPLAY_NAMES, GROUP_ORDER, MODEL_ORDER

NAVY = "#1F4E79"
TEAL = "#0F6E6E"
GOLD = "#C47B17"
BRICK = "#8C2F39"
SLATE = "#5C6B7A"
LIGHT = "#D9E2EC"


def plot_class_balance(n_good: int, n_bad: int, path: Path) -> None:
    total = max(n_good + n_bad, 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = ["Fully paid", "Not fully paid"]
    counts = [n_good, n_bad]
    bars = ax.bar(labels, counts, color=[TEAL, BRICK], width=0.62)
    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({count / total:.0%})",
            ha="center",
            va="bottom",
            color=NAVY,
        )
    ax.set_ylabel("Loans")
    ax.set_ylim(0, max(counts) * 1.28)
    ax.set_title(f"{n_good / total:.0%} of loans in this sample were fully paid")
    _save(fig, path)


def plot_model_comparison(rows: list[dict[str, object]], path: Path, *, winner: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    positions = np.arange(len(MODEL_ORDER))
    width = 0.36
    defaults = [_row(rows, name, "default") for name in MODEL_ORDER]
    tuned = [_row(rows, name, "tuned") for name in MODEL_ORDER]
    default_means = [float(row["cv_roc_auc_mean"]) for row in defaults]
    tuned_means = [float(row["cv_roc_auc_mean"]) for row in tuned]
    default_std = [float(row["cv_roc_auc_std"]) for row in defaults]
    tuned_std = [float(row["cv_roc_auc_std"]) for row in tuned]
    ax.bar(
        positions - width / 2,
        default_means,
        width,
        yerr=default_std,
        color=SLATE,
        label="Default hyperparameters",
        capsize=3,
    )
    tuned_colors = [TEAL if name == winner else NAVY for name in MODEL_ORDER]
    ax.bar(
        positions + width / 2,
        tuned_means,
        width,
        yerr=tuned_std,
        color=tuned_colors,
        label="After search",
        capsize=3,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY_NAMES[name] for name in MODEL_ORDER])
    floor = min([*default_means, *tuned_means, 0.5]) - 0.08
    ax.set_ylim(max(0.0, floor), 1.02)
    ax.set_ylabel("Cross-validated ROC-AUC")
    ax.set_title("Model choice uses cross-validated ROC-AUC, not the holdout set")
    for position, mean, spread in zip(positions + width / 2, tuned_means, tuned_std, strict=True):
        ax.text(
            position,
            min(mean + spread + 0.012, 0.99),
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=NAVY,
        )
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def plot_roc_curves(curves: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    colors = {
        "logistic_regression": NAVY,
        "hist_gradient_boosting": GOLD,
        "random_forest": TEAL,
        "gradient_boosting": GOLD,
    }
    for curve in curves:
        name = str(curve["model"])
        y_true = np.asarray(curve["y_true"]).astype(int)
        scores = np.asarray(curve["proba"], dtype=float)
        fpr, tpr, _thresholds = roc_curve(y_true, scores)
        auc = float(curve["roc_auc"])
        ax.plot(
            fpr,
            tpr,
            color=colors.get(name, SLATE),
            linewidth=2.2,
            label=f"{DISPLAY_NAMES.get(name, name)} ({auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color=SLATE, linewidth=1, label="Chance")
    ax.set_xlabel("False approval rate")
    ax.set_ylabel("True approval rate")
    ax.set_title("Holdout ROC curves")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _save(fig, path)


def plot_confusion(matrix: np.ndarray | list[list[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    grid = np.asarray(matrix, dtype=int)
    sns.heatmap(
        grid,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Decline", "Approve"],
        yticklabels=["Not fully paid", "Fully paid"],
        ax=ax,
        linewidths=1,
        linecolor="white",
    )
    ax.set_xlabel("Decision")
    ax.set_ylabel("Actual outcome")
    ax.set_title("Holdout decisions at the cost-minimizing cutoff")
    ax.grid(False)
    _save(fig, path)


def plot_threshold_cost(
    grid: np.ndarray,
    costs: np.ndarray,
    chosen: float,
    baselines: dict[str, float],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(grid, costs, color=NAVY, linewidth=2.2, label="Model cost")
    ax.axvline(chosen, color=GOLD, linewidth=1.6, label=f"Chosen cutoff {chosen:.2f}")
    ax.axhline(
        baselines["approve_all"],
        color=BRICK,
        linestyle="--",
        linewidth=1.2,
        label=f"Approve everyone (${baselines['approve_all']:,.0f})",
    )
    ax.axhline(
        baselines["decline_all"],
        color=TEAL,
        linestyle="--",
        linewidth=1.2,
        label=f"Decline everyone (${baselines['decline_all']:,.0f})",
    )
    ax.set_xlabel("Approve if the model score is at least")
    ax.set_ylabel("Planning cost per application ($)")
    ax.set_title("Training out-of-fold planning cost")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def plot_importance(frame: pd.DataFrame, path: Path, *, top_n: int = 12) -> None:
    subset = frame.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.barh(
        subset["label"],
        subset["importance_mean"],
        xerr=subset["importance_std"],
        color=NAVY,
        capsize=3,
    )
    ax.set_xlabel("Drop in holdout ROC-AUC when the field is shuffled")
    ax.set_title("Fields the holdout ranking actually uses")
    _save(fig, path)


def plot_coefficients(frame: pd.DataFrame, path: Path, *, top_n: int = 12) -> None:
    subset = frame.head(top_n).iloc[::-1]
    colors = [TEAL if value >= 0 else BRICK for value in subset["coefficient"]]
    labels = ["\n".join(textwrap.wrap(str(value), width=34)) for value in subset["feature"]]
    fig, ax = plt.subplots(figsize=(11.0, 8.2))
    ax.barh(labels, subset["coefficient"], color=colors)
    ax.axvline(0, color=SLATE, linewidth=1)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Log-odds of a fully paid loan (standardized inputs)")
    ax.set_title("Direction of effect in the regularized logistic model")
    _save(fig, path)


def plot_fairness(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty or "attribute" not in frame.columns:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        ax.text(0.5, 0.5, "No fairness rows", ha="center")
        ax.set_axis_off()
        _save(fig, path)
        return
    attributes = [column for column in ATTRIBUTE_LABELS if column in set(frame["attribute"])]
    if not attributes:
        fig, axis = plt.subplots(figsize=(6.2, 3.6))
        axis.text(0.5, 0.5, "No fairness rows", ha="center")
        axis.set_axis_off()
        _save(fig, path)
        return
    fig, axes = plt.subplots(
        1,
        len(attributes),
        figsize=(4.6 * len(attributes), 5.2),
        sharey=True,
        squeeze=False,
    )
    axis_list = list(axes[0])
    for axis, attribute in zip(axis_list, attributes, strict=True):
        block = frame.loc[frame["attribute"] == attribute].copy()
        order = GROUP_ORDER.get(attribute, list(block["group"]))
        block["group"] = pd.Categorical(block["group"], categories=order, ordered=True)
        block = block.sort_values("group")
        x = np.arange(len(block))
        width = 0.36
        axis.bar(
            x - width / 2,
            block["base_rate"].astype(float),
            width,
            color=LIGHT,
            label="Fully paid rate",
        )
        axis.bar(
            x + width / 2,
            block["approval_rate"].astype(float),
            width,
            color=NAVY,
            label="Model approval rate",
        )
        labels = [
            f"{group}\n(n={int(n)})" for group, n in zip(block["group"], block["n"], strict=True)
        ]
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_title(ATTRIBUTE_LABELS[attribute], fontsize=11)
        if axis is axis_list[0]:
            axis.set_ylabel("Rate")
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Out-of-fold approval versus the share of loans fully paid", fontsize=13)
    _save(fig, path)


def plot_bootstrap_intervals(rows: list[dict[str, object]], path: Path, *, winner: str) -> None:
    """Holdout ROC-AUC with the percentile bootstrap interval used in the report."""

    tuned = [_row(rows, name, "tuned") for name in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    positions = np.arange(len(tuned))
    estimates = [float(row["holdout_roc_auc"]) for row in tuned]
    low = []
    high = []
    for row, estimate in zip(tuned, estimates, strict=True):
        ci_low = row.get("holdout_roc_auc_ci_low")
        ci_high = row.get("holdout_roc_auc_ci_high")
        if ci_low is None or ci_high is None:
            low.append(0.0)
            high.append(0.0)
        else:
            low.append(estimate - float(ci_low))
            high.append(float(ci_high) - estimate)
    colors = [TEAL if str(row["model"]) == winner else NAVY for row in tuned]
    ax.bar(
        positions,
        estimates,
        color=colors,
        yerr=np.vstack([low, high]),
        capsize=4,
        width=0.62,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY_NAMES[str(row["model"])] for row in tuned])
    floor = min(estimates) - 0.08
    ax.set_ylim(max(0.0, floor), 1.02)
    ax.set_ylabel("Holdout ROC-AUC")
    ax.set_title("Holdout ROC-AUC with a 95% bootstrap interval")
    for position, estimate in zip(positions, estimates, strict=True):
        ax.text(position, min(estimate + 0.02, 0.98), f"{estimate:.3f}", ha="center", color=NAVY)
    _save(fig, path)


def plot_calibration(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    path: Path,
    *,
    uncalibrated: np.ndarray | pd.Series | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    ax.plot([0, 1], [0, 1], linestyle="--", color=SLATE, label="Perfect calibration")
    if len(np.unique(actual)) < 2 or len(actual) < 8:
        ax.text(
            0.5, 0.55, "Not enough holdout variation\nto draw a reliability curve.", ha="center"
        )
    else:
        _draw_reliability(
            ax,
            actual,
            scores,
            color=NAVY,
            label="After isotonic calibration" if uncalibrated is not None else "Deployed model",
        )
        if uncalibrated is not None:
            _draw_reliability(
                ax,
                actual,
                np.asarray(uncalibrated, dtype=float),
                color=GOLD,
                label="Before calibration",
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Predicted score for a fully paid loan")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Holdout calibration of the deployed model")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def _draw_reliability(
    ax: plt.Axes, actual: np.ndarray, scores: np.ndarray, *, color: str, label: str
) -> None:
    bins = max(3, min(8, len(actual) // 5))
    try:
        fraction, mean_predicted = calibration_curve(
            actual, scores, n_bins=bins, strategy="quantile"
        )
    except ValueError:
        fraction, mean_predicted = calibration_curve(
            actual, scores, n_bins=min(5, bins), strategy="uniform"
        )
    ax.plot(mean_predicted, fraction, marker="o", color=color, label=label)


def contribution_figure(records: list[dict[str, object]], *, top_n: int = 8) -> plt.Figure:
    """Horizontal ablation chart for one application. The caller displays it."""

    ranked = sorted(records, key=lambda record: abs(float(record["delta"])), reverse=True)[:top_n]
    ranked = list(reversed(ranked))
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    if not ranked:
        ax.text(0.5, 0.5, "No field-level shifts", ha="center")
        return fig
    labels = [str(record["label"]) for record in ranked]
    deltas = [float(record["delta"]) for record in ranked]
    colors = [TEAL if delta >= 0 else BRICK for delta in deltas]
    ax.barh(labels, deltas, color=colors)
    ax.axvline(0, color=SLATE, linewidth=1)
    ax.set_xlabel("Change in the ranking score versus a typical file")
    ax.set_title("What pushed this application")
    fig.tight_layout()
    return fig


def _row(rows: list[dict[str, object]], model: str, stage: str) -> dict[str, object]:
    for row in rows:
        if row["model"] == model and row["stage"] == stage:
            return row
    raise KeyError(f"No comparison row for {model} / {stage}.")


def _save(fig: plt.Figure, path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6EDF5",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
        }
    )


_apply_style()
