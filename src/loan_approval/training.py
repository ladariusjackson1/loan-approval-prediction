"""Compare three classifiers and freeze the cost-minimizing decision rule."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from loan_approval.bootstrap import bootstrap_roc_auc, paired_auc_difference
from loan_approval.calibration import constant_brier, cross_fit_isotonic
from loan_approval.config import (
    ABLATION_FEATURE,
    COST_SOURCE,
    CV_FOLDS,
    DATASET,
    DISPLAY_NAMES,
    EXCLUDED_FEATURES,
    FEATURE_COLUMNS,
    FN_COST,
    FP_COST,
    MODEL_ORDER,
    N_BOOTSTRAP,
    N_ITER,
    POLICY_REASON,
    RANDOM_STATE,
    REVIEW_BAND,
    TEST_SIZE,
)
from loan_approval.data import DatasetError, modeling_matrices
from loan_approval.explain import reference_profile, training_ranges
from loan_approval.fairness import fairness_table
from loan_approval.inspection import coefficient_frame, permutation_frame
from loan_approval.metrics import (
    choose_threshold,
    classification_metrics,
    money_saved,
    policy_baselines,
)
from loan_approval.models import make_estimator, search_space
from loan_approval.preprocessing import build_model_pipeline


@dataclass
class ExperimentConfig:
    """Knobs for the train/holdout experiment. Tests use a much smaller search."""

    test_size: float = TEST_SIZE
    seed: int = RANDOM_STATE
    cv_folds: int = CV_FOLDS
    n_iter: int = N_ITER
    n_bootstrap: int = N_BOOTSTRAP
    fp_cost: float = FP_COST
    fn_cost: float = FN_COST
    excluded_features: tuple[str, ...] = field(default_factory=lambda: tuple(EXCLUDED_FEATURES))
    n_jobs: int = -1
    permutation_repeats: int = 15


@dataclass
class ExperimentResult:
    """Fitted policy model plus everything the report figures need."""

    card: dict[str, object]
    pipeline: object
    threshold: float
    reference: dict[str, object]
    ranges: dict[str, dict[str, float]]
    feature_columns: list[str]
    model_name: str
    comparison_rows: list[dict[str, object]]
    roc_curves: list[dict[str, object]]
    confusion: list[list[int]]
    threshold_grid: np.ndarray
    threshold_costs: np.ndarray
    oof_baselines: dict[str, float]
    importance: pd.DataFrame
    coefficients: pd.DataFrame
    fairness_oof: pd.DataFrame
    y_holdout: np.ndarray
    proba_holdout: np.ndarray
    proba_holdout_raw: np.ndarray
    calibrator: object
    n_good: int
    n_bad: int


def run_experiment(df: pd.DataFrame, config: ExperimentConfig | None = None) -> ExperimentResult:
    """Fit the comparison on a stratified training split and score the holdout once."""

    settings = config or ExperimentConfig()
    features, target, sensitive = modeling_matrices(df, excluded=list(settings.excluded_features))
    if target.nunique() < 2:
        raise DatasetError("Both fully paid and not fully paid outcomes are required to train.")

    (
        x_train,
        x_test,
        y_train,
        y_test,
        s_train,
        s_test,
    ) = train_test_split(
        features,
        target,
        sensitive,
        test_size=settings.test_size,
        random_state=settings.seed,
        stratify=target,
    )
    cv = StratifiedKFold(
        n_splits=settings.cv_folds,
        shuffle=True,
        random_state=settings.seed,
    )
    columns = list(features.columns)
    rows: list[dict[str, object]] = []
    fitted: dict[str, object] = {}
    curves: list[dict[str, object]] = []
    oof_by_model: dict[str, np.ndarray] = {}

    for name in MODEL_ORDER:
        estimator = make_estimator(name, settings.seed)
        pipeline = build_model_pipeline(estimator, columns)
        default_mean, default_std = _cv_auc(pipeline, x_train, y_train, cv, settings.n_jobs)
        rows.append(
            _comparison_row(
                name,
                stage="default",
                mean=default_mean,
                std=default_std,
            )
        )

        space = search_space(name)
        n_iter = min(settings.n_iter, len(ParameterGrid(space)))
        search = RandomizedSearchCV(
            estimator=build_model_pipeline(make_estimator(name, settings.seed), columns),
            param_distributions=space,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            refit=True,
            random_state=settings.seed,
            n_jobs=settings.n_jobs,
            error_score="raise",
        )
        started = time.perf_counter()
        search.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        best = search.best_estimator_
        fitted[name] = best
        oof = cross_val_predict(
            clone(best),
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=settings.n_jobs,
        )[:, 1]
        oof_by_model[name] = oof
        threshold, _grid, _costs = choose_threshold(
            y_train,
            oof,
            fp_cost=settings.fp_cost,
            fn_cost=settings.fn_cost,
        )
        holdout_proba = best.predict_proba(x_test)[:, 1]
        holdout = classification_metrics(
            y_test,
            holdout_proba,
            threshold,
            fp_cost=settings.fp_cost,
            fn_cost=settings.fn_cost,
        )
        oof_interval = bootstrap_roc_auc(
            y_train, oof, n_boot=settings.n_bootstrap, seed=settings.seed
        )
        holdout_interval = bootstrap_roc_auc(
            y_test, holdout_proba, n_boot=settings.n_bootstrap, seed=settings.seed
        )
        curves.append(
            {
                "model": name,
                "y_true": np.asarray(y_test).astype(int),
                "proba": holdout_proba,
                "roc_auc": holdout["roc_auc"],
            }
        )
        tuned_row = _comparison_row(
            name,
            stage="tuned",
            mean=float(search.best_score_),
            std=_search_std(search),
            holdout=holdout,
            params=dict(search.best_params_),
            search_seconds=elapsed,
            n_iter=n_iter,
        )
        tuned_row["oof_roc_auc"] = oof_interval["estimate"]
        tuned_row["oof_roc_auc_ci_low"] = oof_interval["ci_low"]
        tuned_row["oof_roc_auc_ci_high"] = oof_interval["ci_high"]
        tuned_row["holdout_roc_auc_ci_low"] = holdout_interval["ci_low"]
        tuned_row["holdout_roc_auc_ci_high"] = holdout_interval["ci_high"]
        rows.append(tuned_row)

    provisional = _select_winner(rows)
    winner_name, oof_gaps = _honest_winner(
        provisional,
        oof_by_model,
        y_train,
        n_bootstrap=settings.n_bootstrap,
        seed=settings.seed,
    )
    holdout_by_model = {str(curve["model"]): np.asarray(curve["proba"]) for curve in curves}
    holdout_gaps = _holdout_gaps(
        provisional,
        holdout_by_model,
        np.asarray(y_test).astype(int),
        n_bootstrap=settings.n_bootstrap,
        seed=settings.seed,
    )
    winner = fitted[winner_name]
    winner_row = _tuned_row(rows, winner_name)
    calibrator, oof_calibrated = cross_fit_isotonic(oof_by_model[winner_name], y_train, cv)
    threshold, grid, costs = choose_threshold(
        y_train,
        oof_calibrated,
        fp_cost=settings.fp_cost,
        fn_cost=settings.fn_cost,
    )
    oof_pred = (oof_calibrated >= threshold).astype(int)
    holdout_raw = np.asarray(winner.predict_proba(x_test)[:, 1], dtype=float)
    if calibrator is None:
        holdout_proba = holdout_raw
    else:
        holdout_proba = np.clip(np.asarray(calibrator.predict(holdout_raw), dtype=float), 0.0, 1.0)
    holdout_pred = (holdout_proba >= threshold).astype(int)
    holdout_metrics = classification_metrics(
        y_test,
        holdout_proba,
        threshold,
        fp_cost=settings.fp_cost,
        fn_cost=settings.fn_cost,
    )
    oof_baselines = policy_baselines(y_train, fp_cost=settings.fp_cost, fn_cost=settings.fn_cost)
    holdout_baselines = policy_baselines(y_test, fp_cost=settings.fp_cost, fn_cost=settings.fn_cost)
    fairness_oof = fairness_table(y_train, oof_pred, s_train)
    fairness_holdout = fairness_table(y_test, holdout_pred, s_test)
    importance = permutation_frame(
        winner,
        x_test,
        y_test,
        repeats=settings.permutation_repeats,
        seed=settings.seed,
        n_jobs=settings.n_jobs,
    )
    coefficients = coefficient_frame(fitted["logistic_regression"])
    if ABLATION_FEATURE not in x_train.columns:
        raise DatasetError(f"Cannot ablate missing feature '{ABLATION_FEATURE}'.")
    ablation_cv = _refit_cv_auc(
        x_train.drop(columns=[ABLATION_FEATURE]),
        y_train,
        winner_name=winner_name,
        winner_params=dict(winner_row["best_params"]),
        cv=cv,
        seed=settings.seed,
        n_jobs=settings.n_jobs,
        baseline_auc=float(winner_row["cv_roc_auc_mean"]),
        note=(
            f"Same model family and hyperparameters, with {ABLATION_FEATURE} removed from an "
            "otherwise unchanged policy feature set. Not used to choose the deployed model."
        ),
    )
    sensitive_cv = _sensitive_feature_lift(
        df.loc[x_train.index],
        y_train,
        winner_name=winner_name,
        winner_params=dict(winner_row["best_params"]),
        cv=cv,
        seed=settings.seed,
        n_jobs=settings.n_jobs,
        policy_auc=float(winner_row["cv_roc_auc_mean"]),
    )
    reference = reference_profile(x_train)
    ranges = training_ranges(x_train)
    calibrated_interval = bootstrap_roc_auc(
        y_test, holdout_proba, n_boot=settings.n_bootstrap, seed=settings.seed
    )
    calibration = {
        "method": (
            "Isotonic regression. Each out-of-fold score is mapped by a fit that did not "
            "see that row. The saved map is refit on all out-of-fold scores and applied "
            "to the model refit on the full training split. Holdout rows are in neither fit."
        ),
        "holdout_brier_uncalibrated": float(brier_score_loss(y_test, holdout_raw)),
        "holdout_brier_calibrated": float(holdout_metrics["brier"]),
        "holdout_brier_base_rate": constant_brier(y_test),
        "holdout_roc_auc_uncalibrated": float(roc_auc_score(y_test, holdout_raw)),
    }
    selection_note = _selection_note(provisional, winner_name, oof_gaps)
    card = _model_card(
        settings=settings,
        rows=rows,
        winner_name=winner_name,
        holdout_metrics=holdout_metrics,
        oof_baselines=oof_baselines,
        holdout_baselines=holdout_baselines,
        threshold_grid=grid,
        threshold_costs=costs,
        fairness_oof=fairness_oof,
        fairness_holdout=fairness_holdout,
        importance=importance,
        coefficients=coefficients,
        sensitive_cv=sensitive_cv,
        ablation_cv=ablation_cv,
        n_train=int(len(x_train)),
        n_test=int(len(x_test)),
        positive_rate=float(target.mean()),
        reference=reference,
        ranges=ranges,
        feature_columns=columns,
        provisional_name=provisional,
        selection_note=selection_note,
        calibration=calibration,
        calibrated_interval=calibrated_interval,
        bootstrap_comparisons=[*oof_gaps, *holdout_gaps],
    )
    return ExperimentResult(
        card=card,
        pipeline=winner,
        threshold=threshold,
        reference=reference,
        ranges=ranges,
        feature_columns=columns,
        model_name=winner_name,
        comparison_rows=rows,
        roc_curves=curves,
        confusion=holdout_metrics["confusion_matrix"],  # type: ignore[arg-type]
        threshold_grid=grid,
        threshold_costs=costs,
        oof_baselines=oof_baselines,
        importance=importance,
        coefficients=coefficients,
        fairness_oof=fairness_oof,
        y_holdout=np.asarray(y_test).astype(int),
        proba_holdout=np.asarray(holdout_proba, dtype=float),
        proba_holdout_raw=holdout_raw,
        calibrator=calibrator,
        n_good=int((target == 1).sum()),
        n_bad=int((target == 0).sum()),
    )


def _cv_auc(
    estimator: object, features: pd.DataFrame, target: pd.Series, cv: StratifiedKFold, n_jobs: int
) -> tuple[float, float]:
    scores = cross_val_score(
        estimator,
        features,
        target,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
    )
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return float(np.mean(scores)), std


def _search_std(search: RandomizedSearchCV) -> float:
    """Fold-level spread of the winning parameter set, not the spread across the search."""

    results = search.cv_results_
    index = int(search.best_index_)
    fold_scores = [
        float(results[f"split{fold}_test_score"][index]) for fold in range(search.n_splits_)
    ]
    if len(fold_scores) < 2:
        return 0.0
    return float(np.std(fold_scores, ddof=1))


def _comparison_row(
    name: str,
    *,
    stage: str,
    mean: float,
    std: float,
    holdout: dict[str, object] | None = None,
    params: dict[str, object] | None = None,
    search_seconds: float | None = None,
    n_iter: int | None = None,
) -> dict[str, object]:
    return {
        "model": name,
        "display_name": DISPLAY_NAMES[name],
        "stage": stage,
        "cv_roc_auc_mean": mean,
        "cv_roc_auc_std": std,
        "holdout_roc_auc": None if holdout is None else holdout["roc_auc"],
        "holdout_f1": None if holdout is None else holdout["f1"],
        "holdout_cost_per_applicant": None if holdout is None else holdout["cost_per_applicant"],
        "threshold": None if holdout is None else holdout["threshold"],
        "holdout_roc_auc_ci_low": None,
        "holdout_roc_auc_ci_high": None,
        "oof_roc_auc": None,
        "oof_roc_auc_ci_low": None,
        "oof_roc_auc_ci_high": None,
        "best_params": params,
        "search_seconds": search_seconds,
        "n_iter": n_iter,
    }


def _select_winner(rows: list[dict[str, object]]) -> str:
    preference = {name: index for index, name in enumerate(MODEL_ORDER)}
    tuned = [row for row in rows if row["stage"] == "tuned"]
    ranked = sorted(
        tuned,
        key=lambda row: (
            -float(row["cv_roc_auc_mean"]),
            float(row["cv_roc_auc_std"]),
            preference[str(row["model"])],
        ),
    )
    return str(ranked[0]["model"])


def _honest_winner(
    provisional: str,
    oof_by_model: dict[str, np.ndarray],
    y_train: pd.Series,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[str, list[dict[str, object]]]:
    """Keep a more complex leader only when the out-of-fold gap excludes zero.

    The paired interval is the selection check. A simpler model whose gap versus
    the cross-validated leader includes zero joins the tie, and the tie breaks
    toward the earlier name in ``MODEL_ORDER``. A missing interval does not
    create a tie.
    """

    details: list[dict[str, object]] = []
    tied = [provisional]
    actual = np.asarray(y_train).astype(int)
    for name in MODEL_ORDER:
        if name == provisional:
            continue
        diff = paired_auc_difference(
            actual,
            oof_by_model[provisional],
            oof_by_model[name],
            n_boot=n_bootstrap,
            seed=seed,
        )
        details.append(
            {
                "reference_model": provisional,
                "other_model": name,
                "sample": "out_of_fold",
                "auc_difference": diff["estimate"],
                "ci_low": diff["ci_low"],
                "ci_high": diff["ci_high"],
                "includes_zero": diff["includes_zero"],
                "n_boot_valid": diff["n_boot_valid"],
                "used_for_selection": True,
            }
        )
        if diff["includes_zero"] is True:
            tied.append(name)
    preference = {name: index for index, name in enumerate(MODEL_ORDER)}
    chosen = sorted(tied, key=lambda name: preference[name])[0]
    return chosen, details


def _holdout_gaps(
    provisional: str,
    holdout_by_model: dict[str, np.ndarray],
    y_test: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    """Paired holdout gaps. Reported, and not used to choose the model."""

    gaps: list[dict[str, object]] = []
    for name in MODEL_ORDER:
        if name == provisional:
            continue
        diff = paired_auc_difference(
            y_test,
            holdout_by_model[provisional],
            holdout_by_model[name],
            n_boot=n_bootstrap,
            seed=seed,
        )
        gaps.append(
            {
                "reference_model": provisional,
                "other_model": name,
                "sample": "holdout",
                "auc_difference": diff["estimate"],
                "ci_low": diff["ci_low"],
                "ci_high": diff["ci_high"],
                "includes_zero": diff["includes_zero"],
                "n_boot_valid": diff["n_boot_valid"],
                "used_for_selection": False,
            }
        )
    return gaps


def _selection_note(provisional: str, chosen: str, details: list[dict[str, object]]) -> str:
    """One sentence that states which rule fired. The numbers are the intervals."""

    if chosen != provisional:
        gap = next(item for item in details if item["other_model"] == chosen)
        return (
            f"{DISPLAY_NAMES[provisional]} led on cross-validated ROC-AUC, but the paired "
            f"out-of-fold bootstrap 95% interval for its ROC-AUC minus "
            f"{DISPLAY_NAMES[chosen]} is {float(gap['ci_low']):.3f} to "
            f"{float(gap['ci_high']):.3f}, which includes zero. "
            f"The simpler model, {DISPLAY_NAMES[chosen]}, is deployed."
        )
    if any(item["includes_zero"] is None for item in details):
        return (
            f"{DISPLAY_NAMES[chosen]} led on cross-validated ROC-AUC. A paired bootstrap "
            "interval was unavailable, so that ranking was not overridden."
        )
    separated = [
        item
        for item in details
        if item["includes_zero"] is False and float(item["auc_difference"]) > 0
    ]
    if details and len(separated) == len(details):
        others = " and ".join(DISPLAY_NAMES[str(item["other_model"])] for item in details)
        return (
            f"{DISPLAY_NAMES[chosen]} is deployed. It led on cross-validated ROC-AUC, and "
            f"the paired out-of-fold bootstrap 95% interval for its ROC-AUC advantage "
            f"over {others} excludes zero."
        )
    if chosen != MODEL_ORDER[0]:
        return (
            f"{DISPLAY_NAMES[chosen]} is deployed. The paired out-of-fold bootstrap 95% "
            "interval for its ROC-AUC advantage over each simpler model excludes zero."
        )
    return (
        f"{DISPLAY_NAMES[chosen]} is deployed. No more complex model separated itself "
        "from it on the paired out-of-fold bootstrap 95% interval."
    )


def _tuned_row(rows: list[dict[str, object]], name: str) -> dict[str, object]:
    for row in rows:
        if row["model"] == name and row["stage"] == "tuned":
            return row
    raise KeyError(name)


def _sensitive_feature_lift(
    train_frame: pd.DataFrame,
    y_train: pd.Series,
    *,
    winner_name: str,
    winner_params: dict[str, object],
    cv: StratifiedKFold,
    seed: int,
    n_jobs: int,
    policy_auc: float,
) -> dict[str, object]:
    """Same winning hyperparameters, but with the withheld fields put back."""

    restored = ", ".join(EXCLUDED_FEATURES)
    comparison = _refit_cv_auc(
        train_frame[FEATURE_COLUMNS],
        y_train,
        winner_name=winner_name,
        winner_params=winner_params,
        cv=cv,
        seed=seed,
        n_jobs=n_jobs,
        baseline_auc=policy_auc,
        note=(
            "Same model family and hyperparameters as the deployed scorer, refit in "
            f"cross-validation with {restored} added back. "
            "This is not a second search, and it is not used to pick the deployed model."
        ),
    )
    comparison["auc_lift_from_sensitive_fields"] = comparison["auc_difference_vs_policy"]
    return comparison


def _refit_cv_auc(
    features: pd.DataFrame,
    y_train: pd.Series,
    *,
    winner_name: str,
    winner_params: dict[str, object],
    cv: StratifiedKFold,
    seed: int,
    n_jobs: int,
    baseline_auc: float,
    note: str,
) -> dict[str, object]:
    """Cross-validated ROC-AUC for one feature set at the already chosen hyperparameters."""

    pipeline = build_model_pipeline(make_estimator(winner_name, seed), list(features.columns))
    pipeline.set_params(**winner_params)
    scores = cross_val_score(
        pipeline,
        features,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
    )
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return {
        "note": note,
        "columns": list(features.columns),
        "cv_roc_auc_mean": mean,
        "cv_roc_auc_std": std,
        "policy_cv_roc_auc_mean": baseline_auc,
        "auc_difference_vs_policy": mean - baseline_auc,
    }


def _model_card(
    *,
    settings: ExperimentConfig,
    rows: list[dict[str, object]],
    winner_name: str,
    holdout_metrics: dict[str, object],
    oof_baselines: dict[str, float],
    holdout_baselines: dict[str, float],
    threshold_grid: np.ndarray,
    threshold_costs: np.ndarray,
    fairness_oof: pd.DataFrame,
    fairness_holdout: pd.DataFrame,
    importance: pd.DataFrame,
    coefficients: pd.DataFrame,
    sensitive_cv: dict[str, object],
    ablation_cv: dict[str, object],
    n_train: int,
    n_test: int,
    positive_rate: float,
    reference: dict[str, object],
    ranges: dict[str, dict[str, float]],
    feature_columns: list[str],
    provisional_name: str,
    selection_note: str,
    calibration: dict[str, object],
    calibrated_interval: dict[str, object],
    bootstrap_comparisons: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "project": "Loan approval prediction",
        "author": "Ladarius Jackson",
        "dataset": DATASET,
        "positive_class": "fully paid, coded as 1, decision = approve",
        "positive_rate": positive_rate,
        "policy": {
            "excluded_features": list(settings.excluded_features),
            "reason": POLICY_REASON,
            "ablation_feature": ABLATION_FEATURE,
            "fp_cost": settings.fp_cost,
            "fn_cost": settings.fn_cost,
            "cost_source": COST_SOURCE,
            "review_band": REVIEW_BAND,
        },
        "split": {
            "test_size": settings.test_size,
            "seed": settings.seed,
            "n_train": n_train,
            "n_test": n_test,
            "cv_folds": settings.cv_folds,
            "search_iterations_requested": settings.n_iter,
            "n_bootstrap": settings.n_bootstrap,
            "scoring": "roc_auc",
        },
        "selection_rule": (
            "Highest mean stratified cross-validated ROC-AUC on uncalibrated training "
            "scores. Ties in that mean break toward the smaller fold standard deviation, "
            "then toward the simpler model (logistic regression, then histogram gradient "
            "boosting). A more complex leader is kept only when the paired bootstrap 95% "
            "interval for its out-of-fold ROC-AUC advantage excludes zero. If the interval "
            "includes zero, the simpler model is deployed. The holdout set is not used to "
            "choose the model or the threshold. After the model is chosen, an isotonic map "
            "is fit on out-of-fold scores and the cost cutoff is chosen on the calibrated "
            "out-of-fold probabilities."
        ),
        "comparison": rows,
        "winner": {
            "model": winner_name,
            "display_name": DISPLAY_NAMES[winner_name],
            "provisional_model": provisional_name,
            "provisional_display_name": DISPLAY_NAMES[provisional_name],
            "selection_note": selection_note,
            "holdout_roc_auc_ci": [
                calibrated_interval["ci_low"],
                calibrated_interval["ci_high"],
            ],
        },
        "calibration": calibration,
        "bootstrap_comparisons": bootstrap_comparisons,
        "holdout": holdout_metrics,
        "money": money_saved(holdout_metrics, holdout_baselines),
        "oof_baselines": oof_baselines,
        "holdout_baselines": holdout_baselines,
        "threshold_curve": {
            "threshold": threshold_grid.tolist(),
            "cost_per_applicant": threshold_costs.tolist(),
        },
        "fairness_oof": fairness_oof,
        "fairness_holdout": fairness_holdout,
        "sensitive_feature_comparison": sensitive_cv,
        "ablation_feature_comparison": ablation_cv,
        "permutation_importance": importance,
        "logistic_coefficients": coefficients,
        "reference_profile": reference,
        "training_ranges": ranges,
        "feature_columns": feature_columns,
        "sklearn_version": sklearn.__version__,
    }
