"""Render MODEL_CARD.md from the JSON model card so the prose cannot drift.

Every figure in the markdown is taken from the card produced by training.
Nothing in this module recomputes a metric from a remembered constant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from loan_approval.metrics import dollars

LIMITATIONS = (
    "This is a teaching model on a public 9,578-loan extract. It does not issue a credit offer, a denial, or a compliance determination.",
    "The extract has no sex, age, or race fields, so it cannot support a protected-class audit. FICO band, loan purpose, and the old credit-policy flag are not protected classes.",
    "The four-fifths ratio is a screening arithmetic on those segments. It is not a legal conclusion under the Equal Credit Opportunity Act or any other statute.",
    "There is no reject inference. The file is loans that were made, not applications that were turned away.",
    "The label is fully paid versus not fully paid, which is broader than a charge-off.",
    "The file has no term, no principal, and no loss amount. The $5,000 and $600 figures are planning weights tied to a 36-month sketch at the sample means, not observed recoveries.",
    "The monthly payment still carries some of the offer, because installment depends on amount, rate, and term, even though the interest rate itself is withheld.",
    "Decline reasons are one-field ablations against a typical training file. A counterfactual hint is the smallest one-field change, inside the training range, that the model scores above the cutoff. Correlated fields share credit. Neither is a FCRA adverse-action notice.",
    "The isotonic map makes the deployed score a probability for planning. It was fit on out-of-fold training scores only. It is not a guarantee that a future book will match these frequencies.",
    "The saved pipeline is tied to the scikit-learn version recorded in the model card.",
)


def render_model_card(card: dict[str, Any]) -> str:
    """Markdown model card. Headings follow the Model Cards structure."""

    winner = card.get("winner", {})
    holdout = card.get("holdout", {})
    dataset = card.get("dataset", {})
    policy = card.get("policy", {})
    split = card.get("split", {})
    calibration = card.get("calibration", {})
    money = card.get("money", {})
    lines = [
        "# Model card: loan approval score",
        "",
        "Educational score on a public Lending Club sample. Not a credit offer, a denial, or a compliance determination.",
        "",
        "## Model details",
        "",
        f"- **Deployed model:** {winner.get('display_name', '—')}",
        f"- **Provisional leader on cross-validated ROC-AUC:** {winner.get('provisional_display_name', winner.get('display_name', '—'))}",
        f"- **Selection:** {card.get('selection_rule', '')}",
        f"- **What the run actually did:** {winner.get('selection_note', '')}",
        f"- **Person responsible:** {card.get('author', '—')}",
        f"- **Library:** scikit-learn {card.get('sklearn_version', '—')}",
        f"- **Positive class:** {card.get('positive_class', '—')}",
        f"- **Inputs:** {', '.join(card.get('feature_columns', []))}",
        f"- **Withheld:** {', '.join(policy.get('excluded_features', []))}. {policy.get('reason', '')}",
        "",
        "## Intended use",
        "",
        "Compare a cost-weighted approve / decline cutoff with two rules that ignore the model, on a public sample a reviewer can rerun. The Streamlit desk scores one file, simulates a book at a chosen cutoff, and scores a CSV.",
        "",
        "Out of scope: production underwriting, pricing, adverse-action letters, fair-lending certification, and any decision about a real applicant.",
        "",
        "## Factors",
        "",
        "The score sees the application fields listed above. Segment rates are reported for the old credit-policy flag, FICO band, and loan purpose. Those segments are evaluation slices. FICO and purpose are also inputs. The old policy flag is not an input.",
        "",
        "## Metrics",
        "",
        "Model choice uses stratified cross-validated ROC-AUC on uncalibrated scores, with a paired percentile bootstrap of the out-of-fold gap. The holdout bootstrap interval is reported and is not used to pick the model or the cutoff.",
        "",
        "After the model is chosen, isotonic regression maps the score to a probability. The cutoff minimizes average planning cost on cross-fit calibrated out-of-fold scores. Ties break toward the higher cutoff.",
        "",
        f"Planning weights: ${dollars(float(policy.get('fp_cost', 0)))} for funding a loan that is not fully paid, ${dollars(float(policy.get('fn_cost', 0)))} for declining a loan that is fully paid. {policy.get('cost_source', '')}",
        "",
        "Brier score is reported for the uncalibrated score, the calibrated probability, and a constant prediction at the holdout fully-paid rate.",
        "",
        "## Training and evaluation data",
        "",
        f"- **Dataset:** {dataset.get('name', '—')}",
        f"- **OpenML:** [{dataset.get('openml_id', '—')}]({dataset.get('openml_url', '')})",
        f"- **Licence:** {dataset.get('license', '—')}",
        f"- **How it was fetched:** {dataset.get('fetched_with', '—')}",
        f"- **Target:** {dataset.get('target_definition', '—')}",
        f"- **Fully-paid rate in the file:** {_pct(card.get('positive_rate'))}",
        f"- **Split:** stratified holdout of {_pct(split.get('test_size'))}, seed {split.get('seed', '—')}. Train {split.get('n_train', '—')} / holdout {split.get('n_test', '—')}.",
        f"- **Cross-validation:** {split.get('cv_folds', '—')} stratified folds, seed {split.get('seed', '—')}. Search iterations requested: {split.get('search_iterations_requested', '—')}.",
        f"- **Bootstrap draws:** {split.get('n_bootstrap', '—')}.",
        f"- **Cache hash:** `{card.get('data_sha256', 'computed at train time')}`",
        "",
        dataset.get("license_notes", ""),
        "",
        "## Quantitative analysis",
        "",
        "### Ranking comparison",
        "",
        "Cross-validated ROC-AUC is the selection metric. Holdout ROC-AUC below is the uncalibrated ranking score. The interval is a percentile bootstrap on the holdout rows.",
        "",
        _comparison_table(card.get("comparison", [])),
        "",
        "### Bootstrap gaps",
        "",
        _bootstrap_table(card.get("bootstrap_comparisons", [])),
        "",
        "### Deployed policy, after calibration",
        "",
        f"- **Cutoff:** {_num(holdout.get('threshold'), 2)}",
        f"- **Holdout ROC-AUC (calibrated scores):** {_num(holdout.get('roc_auc'), 3)} "
        f"(95% bootstrap interval {_num(_ci(winner, 0), 3)} to {_num(_ci(winner, 1), 3)})",
        f"- **Uncalibrated holdout ROC-AUC:** {_num(calibration.get('holdout_roc_auc_uncalibrated'), 3)}",
        f"- **Holdout average precision:** {_num(holdout.get('average_precision'), 3)}",
        f"- **Holdout F1 at the cutoff:** {_num(holdout.get('f1'), 3)}",
        f"- **Precision / recall:** {_num(holdout.get('precision'), 3)} / {_num(holdout.get('recall'), 3)}",
        f"- **Approval rate:** {_pct(holdout.get('approval_rate'))}",
        f"- **Confusion matrix** (rows are not fully paid, fully paid; columns are decline, approve): "
        f"{holdout.get('confusion_matrix', '—')}",
        f"- **Counts:** true decline {holdout.get('tn', '—')}, false approval {holdout.get('fp', '—')}, "
        f"false decline {holdout.get('fn', '—')}, true approval {holdout.get('tp', '—')}, "
        f"n = {holdout.get('n', '—')}",
        f"- **Brier, calibrated:** {_num(calibration.get('holdout_brier_calibrated', holdout.get('brier')), 3)}",
        f"- **Brier, uncalibrated:** {_num(calibration.get('holdout_brier_uncalibrated'), 3)}",
        f"- **Brier, constant holdout fully-paid rate:** {_num(calibration.get('holdout_brier_base_rate'), 3)}",
        f"- **Planning cost per holdout loan:** ${dollars(float(holdout.get('cost_per_applicant') or 0))}",
        f"- **Planning cost on the holdout book:** ${dollars(float(money.get('model_cost_total') or 0))}",
        f"- **Approve-everyone cost:** ${dollars(float(money.get('approve_all_cost_total') or 0))}",
        f"- **Decline-everyone cost:** ${dollars(float(money.get('decline_all_cost_total') or 0))}",
        f"- **Saved versus the cheaper naive rule ({money.get('better_naive_rule', '—')}):** "
        f"${dollars(float(money.get('saved_total_vs_better_naive') or 0))} "
        f"(${dollars(float(money.get('saved_per_applicant_vs_better_naive') or 0))} per loan)",
        "",
        "### Side refits, not used to choose the model",
        "",
        _side_refit("Withheld fields put back", card.get("sensitive_feature_comparison")),
        "",
        _side_refit("FICO removed", card.get("ablation_feature_comparison")),
        "",
        "### Where approvals land",
        "",
        "Rates use out-of-fold scores from the deployed model, at the deployed cutoff, so each training loan was scored by a model that did not train on it. The reference groups are meets-policy, FICO 760 and over, and debt consolidation.",
        "",
        _fairness_table(card.get("fairness_oof", [])),
        "",
        "## Ethical considerations",
        "",
        "A strict cutoff concentrates approvals in higher FICO bands and in purposes the training book repaid more often. That is the score and the planning weights. It is not evidence about a protected class, because this extract does not contain one.",
        "",
        "The plain-English decline reasons and the one-field counterfactual are review aids. They are not reasons that satisfy the Fair Credit Reporting Act, and they are not instructions to alter a credit file.",
        "",
        "## Caveats and recommendations",
        "",
        *[f"- {item}" for item in LIMITATIONS],
        "",
        "Quote the cross-validated ROC-AUC, with its bootstrap interval, when comparing models. Quote the calibrated probability against the cutoff when talking about a decision. Quote realized holdout counts separately from expected defaults, which use only the probabilities.",
        "",
    ]
    return "\n".join(lines)


def write_model_card(card: dict[str, Any], path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_model_card(card), encoding="utf-8")


def _comparison_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Model | Stage | CV ROC-AUC | Holdout ROC-AUC | Holdout 95% interval |\n"
        "| --- | --- | ---: | ---: | ---: |"
    )
    body = []
    for row in rows:
        interval = "—"
        if row.get("stage") == "tuned" and row.get("holdout_roc_auc_ci_low") is not None:
            interval = (
                f"{_num(row.get('holdout_roc_auc_ci_low'), 3)} to "
                f"{_num(row.get('holdout_roc_auc_ci_high'), 3)}"
            )
        cv = f"{_num(row.get('cv_roc_auc_mean'), 3)} ± {_num(row.get('cv_roc_auc_std'), 3)}"
        holdout = "—" if row.get("holdout_roc_auc") is None else _num(row.get("holdout_roc_auc"), 3)
        body.append(
            f"| {row.get('display_name', row.get('model'))} | {row.get('stage')} | {cv} | "
            f"{holdout} | {interval} |"
        )
    if not body:
        return "_No comparison rows._"
    return "\n".join([header, *body])


def _bootstrap_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No bootstrap comparisons._"
    header = (
        "| Sample | Reference minus other | Difference | 95% interval | Includes zero | Used to choose |\n"
        "| --- | --- | ---: | ---: | --- | --- |"
    )
    body = []
    for row in rows:
        pair = f"{row.get('reference_model')} − {row.get('other_model')}"
        interval = "not available"
        if row.get("ci_low") is not None:
            interval = f"{_num(row.get('ci_low'), 3)} to {_num(row.get('ci_high'), 3)}"
        includes = row.get("includes_zero")
        if includes is None:
            includes_text = "not available"
        else:
            includes_text = "yes" if includes else "no"
        used = "yes" if row.get("used_for_selection") else "no"
        body.append(
            f"| {row.get('sample')} | {pair} | {_num(row.get('auc_difference'), 3)} | "
            f"{interval} | {includes_text} | {used} |"
        )
    return "\n".join([header, *body])


def _fairness_table(rows: Any) -> str:
    records = _records(rows)
    if not records:
        return "_No segment rows._"
    header = (
        "| Segment | Group | n | Fully paid | Approval rate | Ratio vs reference |\n"
        "| --- | --- | ---: | ---: | ---: | ---: |"
    )
    body = []
    for row in records:
        body.append(
            f"| {row.get('attribute')} | {row.get('group')} | {row.get('n')} | "
            f"{_pct(row.get('base_rate'))} | {_pct(row.get('approval_rate'))} | "
            f"{_num(row.get('disparate_impact'), 2)} |"
        )
    return "\n".join([header, *body])


def _side_refit(title: str, block: Any) -> str:
    if not isinstance(block, dict) or not block:
        return f"**{title}.** Not run."
    return (
        f"**{title}.** Cross-validated ROC-AUC {_num(block.get('cv_roc_auc_mean'), 3)} "
        f"± {_num(block.get('cv_roc_auc_std'), 3)}, a difference of "
        f"{_num(block.get('auc_difference_vs_policy'), 3)} versus the policy feature set. "
        f"{block.get('note', '')}"
    )


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, list):
        return value
    return []


def _ci(winner: dict[str, Any], index: int) -> Any:
    interval = winner.get("holdout_roc_auc_ci") or [None, None]
    if len(interval) <= index:
        return None
    return interval[index]


def _num(value: Any, digits: int) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.1%}"
