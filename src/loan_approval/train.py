"""Command line entry: python -m loan_approval.train"""

from __future__ import annotations

import argparse
from pathlib import Path

from loan_approval.artifacts import file_sha256, save_bundle, save_model_card
from loan_approval.config import (
    CV_FOLDS,
    DATA_PATH,
    DISPLAY_NAMES,
    FIGURE_DIR,
    MODEL_CARD_MD_PATH,
    MODEL_CARD_PATH,
    MODEL_PATH,
    N_BOOTSTRAP,
    N_ITER,
    RANDOM_STATE,
    REVIEW_BAND,
)
from loan_approval.data import load_loan_data
from loan_approval.model_card_doc import write_model_card
from loan_approval.plots import (
    plot_bootstrap_intervals,
    plot_calibration,
    plot_class_balance,
    plot_coefficients,
    plot_confusion,
    plot_fairness,
    plot_importance,
    plot_model_comparison,
    plot_roc_curves,
    plot_threshold_cost,
)
from loan_approval.training import ExperimentConfig, ExperimentResult, run_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Lending Club loan decision model and write the report artifacts."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--card-path", type=Path, default=MODEL_CARD_PATH)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--n-iter", type=int, default=N_ITER)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--cv", type=int, default=CV_FOLDS)
    parser.add_argument("--card-md", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Refresh data/lending_club_sample.csv from OpenML data id 43729 before training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> ExperimentResult:
    args = parse_args(argv)
    frame = load_loan_data(args.data, download=args.download)
    result = run_experiment(
        frame,
        ExperimentConfig(
            seed=args.seed,
            cv_folds=args.cv,
            n_iter=args.n_iter,
            n_bootstrap=args.bootstrap,
            n_jobs=args.jobs,
        ),
    )
    card = dict(result.card)
    card["data_sha256"] = file_sha256(args.data)
    save_model_card(args.card_path, card)
    save_bundle(
        args.model_path,
        {
            "pipeline": result.pipeline,
            "threshold": result.threshold,
            "reference": result.reference,
            "ranges": result.ranges,
            "feature_columns": result.feature_columns,
            "model_name": result.model_name,
            "review_band": REVIEW_BAND,
            "calibrator": result.calibrator,
            "holdout_y": result.y_holdout,
            "holdout_proba": result.proba_holdout,
        },
    )
    card_md = args.card_md
    if card_md is None:
        card_md = (
            MODEL_CARD_MD_PATH
            if args.card_path == MODEL_CARD_PATH
            else args.card_path.parent / "MODEL_CARD.md"
        )
    write_model_card(card, card_md)
    _save_figures(result, args.figure_dir)
    holdout = card["holdout"]
    money = card["money"]
    print(
        f"Deployed {DISPLAY_NAMES[result.model_name]} | "
        f"holdout ROC-AUC {holdout['roc_auc']:.3f} | "
        f"F1 {holdout['f1']:.3f} | "
        f"cost/applicant ${holdout['cost_per_applicant']:.2f} | "
        f"saved vs {money['better_naive_rule']} "
        f"${money['saved_total_vs_better_naive']:.2f} | "
        f"threshold {result.threshold:.2f}"
    )
    print(f"Model card: {args.card_path}")
    print(f"Model card markdown: {card_md}")
    print(f"Pipeline: {args.model_path}")
    print(f"Figures: {args.figure_dir}")
    return result


def _save_figures(result: ExperimentResult, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    plot_class_balance(result.n_good, result.n_bad, directory / "class_balance.png")
    plot_model_comparison(
        result.comparison_rows,
        directory / "model_comparison.png",
        winner=result.model_name,
    )
    plot_roc_curves(result.roc_curves, directory / "roc_curves.png")
    plot_confusion(result.confusion, directory / "confusion_matrix.png")
    plot_threshold_cost(
        result.threshold_grid,
        result.threshold_costs,
        result.threshold,
        result.oof_baselines,
        directory / "threshold_cost.png",
    )
    plot_importance(result.importance, directory / "feature_importance.png")
    plot_coefficients(result.coefficients, directory / "logistic_coefficients.png")
    plot_fairness(result.fairness_oof, directory / "fairness.png")
    plot_calibration(
        result.y_holdout,
        result.proba_holdout,
        directory / "calibration.png",
        uncalibrated=result.proba_holdout_raw,
    )
    plot_bootstrap_intervals(
        result.comparison_rows,
        directory / "bootstrap_intervals.png",
        winner=result.model_name,
    )


if __name__ == "__main__":
    main()
