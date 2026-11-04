"""Educational loan desk. Run from the repo root: streamlit run streamlit_app.py"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from loan_approval.artifacts import load_bundle, load_model_card
from loan_approval.config import (
    ATTRIBUTE_LABELS,
    CATEGORY_OPTIONS,
    DATASET,
    FEATURE_LABELS,
    FN_COST,
    FP_COST,
    MODEL_CARD_MD_PATH,
    MODEL_CARD_PATH,
    MODEL_PATH,
    PRESETS,
    VALUE_LABELS,
)
from loan_approval.explain import explain_application, preset_payload
from loan_approval.metrics import dollars
from loan_approval.model_card_doc import LIMITATIONS
from loan_approval.plots import contribution_figure
from loan_approval.portfolio import portfolio_curve, portfolio_snapshot
from loan_approval.scoring import application_template, score_batch

TYPICAL = "Typical training profile"
INTEGER_FIELDS = {"fico", "inq_last_6mths", "delinq_2yrs", "pub_rec", "revol_bal"}
# Fallback widget bounds in display units (dollars of income, years of credit history).
FALLBACK_BOUNDS: dict[str, tuple[float, float, float]] = {
    "installment": (16.0, 940.0, 1.0),
    "log_annual_inc": (2000.0, 500000.0, 500.0),
    "dti": (0.0, 30.0, 0.1),
    "fico": (612.0, 827.0, 1.0),
    "days_with_cr_line": (0.5, 48.0, 0.1),
    "revol_bal": (0.0, 200000.0, 100.0),
    "revol_util": (0.0, 119.0, 1.0),
    "inq_last_6mths": (0.0, 33.0, 1.0),
    "delinq_2yrs": (0.0, 13.0, 1.0),
    "pub_rec": (0.0, 5.0, 1.0),
}
REQUEST_FIELDS = ["purpose", "installment", "log_annual_inc", "dti"]
CREDIT_FIELDS = [
    "fico",
    "days_with_cr_line",
    "revol_bal",
    "revol_util",
    "inq_last_6mths",
    "delinq_2yrs",
    "pub_rec",
]


def main() -> None:
    st.set_page_config(page_title="Loan Decision Desk", page_icon="🏦", layout="wide")
    _theme()
    st.title("Loan decision desk")
    st.caption(
        "Educational scorer on a public Lending Club sample. "
        "It is not a credit offer, a denial notice, or a compliance determination."
    )
    if not MODEL_PATH.exists() or not MODEL_CARD_PATH.exists():
        st.error(
            "Trained artifacts are missing. From the repository root run "
            "`python -m loan_approval.train` and reload this page."
        )
        st.stop()

    bundle, card = _load_artifacts()
    _sidebar(card, bundle)
    score_tab, book_tab, batch_tab, limits_tab = st.tabs(
        [
            "Score one file",
            "Portfolio simulator",
            "Batch scoring",
            "Fairness and limits",
        ]
    )
    with score_tab:
        _score_tab(bundle, card)
    with book_tab:
        _portfolio_tab(bundle)
    with batch_tab:
        _batch_tab(bundle)
    with limits_tab:
        _limits_tab(card)


@st.cache_resource
def _load_artifacts() -> tuple[dict, dict]:
    return load_bundle(MODEL_PATH), load_model_card(MODEL_CARD_PATH)


def _theme() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            background: #E7F2EC;
            border: 1px solid #C9DDD2;
            border-radius: 12px;
            padding: 0.7rem 0.9rem;
        }
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _score_tab(bundle: dict, card: dict) -> None:
    _load_example(bundle)
    payload = _application_fields(bundle["feature_columns"], bundle.get("ranges") or {})
    threshold = st.slider(
        "Operating cutoff: approve when the calibrated score is at least",
        min_value=0.05,
        max_value=0.95,
        value=float(bundle["threshold"]),
        step=0.01,
        help=(
            f"The saved cutoff minimizes a planning cost of USD {FP_COST:,.0f} for funding "
            f"a loan that is not fully paid and USD {FN_COST:,.0f} for declining a loan that "
            "would have been fully paid. Those are scenario weights, not observed losses. "
            "Moving the slider changes the decision, not the score. The score is an "
            "isotonic calibration of the model, fit on out-of-fold training scores."
        ),
        key="single_threshold",
    )
    result = explain_application(
        bundle["pipeline"],
        payload,
        reference=bundle["reference"],
        feature_columns=list(bundle["feature_columns"]),
        threshold=float(threshold),
        ranges=bundle.get("ranges"),
        review_band=float(bundle.get("review_band", 0.05)),
        calibrator=bundle.get("calibrator"),
    )
    _render_decision(result, card)


def _portfolio_tab(bundle: dict) -> None:
    st.subheader("Holdout book at a cutoff you choose")
    st.caption(
        "These loans are the stratified holdout. Expected defaults and expected profit "
        "use only the calibrated probability the loan is fully paid. Realized counts use "
        "the historical label. The two are not interchangeable."
    )
    if "holdout_y" not in bundle or "holdout_proba" not in bundle:
        st.error("This model file has no holdout scores. Retrain to use the simulator.")
        return
    threshold = st.slider(
        "Approve when the calibrated probability is at least",
        min_value=0.05,
        max_value=0.95,
        value=float(bundle["threshold"]),
        step=0.01,
        key="book_threshold",
        help=(
            "Expected profit versus declining every loan, for each approved loan, is "
            f"p times USD {FN_COST:,.0f} minus (1 − p) times USD {FP_COST:,.0f}. "
            "Declined loans contribute nothing under that baseline. "
            "It is a planning weight, not interest income."
        ),
    )
    snapshot = portfolio_snapshot(bundle["holdout_y"], bundle["holdout_proba"], float(threshold))
    _metric_row(
        [
            (
                "Approvals",
                f"{int(snapshot['approvals']):,} of {int(snapshot['n']):,}",
                "Holdout loans at or above the cutoff.",
            ),
            (
                "Expected defaults",
                f"{float(snapshot['expected_defaults']):,.1f}",
                "Sum of (1 − p) on approved loans. A probability, not a headcount.",
            ),
            (
                "Expected profit vs decline-all",
                _signed_dollars(float(snapshot["expected_profit_vs_decline_all"])),
                "Planning-weight profit of approving these loans instead of declining them all.",
            ),
            (
                "Realized defaults",
                f"{int(snapshot['realized_defaults']):,}",
                "Approved holdout loans that were not fully paid.",
            ),
        ]
    )
    realized_rate = snapshot["realized_default_rate"]
    expected_rate = snapshot["expected_default_rate"]
    rate_note = "no approvals at this cutoff"
    if realized_rate is not None and expected_rate is not None:
        rate_note = (
            f"realized default rate {realized_rate:.1%} among approvals, "
            f"expected default rate {expected_rate:.1%}"
        )
    st.caption(
        "Realized planning cost "
        f"USD {dollars(float(snapshot['realized_planning_cost']))} "
        "(USD "
        f"{dollars(float(snapshot['realized_savings_vs_decline_all']))} versus declining "
        f"everyone). {rate_note}."
    )
    curve = portfolio_curve(bundle["holdout_y"], bundle["holdout_proba"])
    chart = curve.rename(
        columns={
            "expected_profit_vs_decline_all": "Expected profit vs decline-all",
            "realized_savings_vs_decline_all": "Realized savings vs decline-all",
        }
    )
    st.line_chart(
        chart,
        x="threshold",
        y=["Expected profit vs decline-all", "Realized savings vs decline-all"],
    )
    st.caption(
        "The saved cutoff is the one chosen on the training book. "
        f"This slider starts at {float(bundle['threshold']):.0%} and can move. "
        f"Weights: USD {FP_COST:,.0f} for a funded loan that is not fully paid, "
        f"USD {FN_COST:,.0f} for a declined loan that is fully paid."
    )


def _batch_tab(bundle: dict) -> None:
    st.subheader("Score a CSV")
    st.caption(
        "The file needs the model columns. Extra columns are kept. "
        "Interest rate and the old credit-policy flag are not inputs. "
        "Declined rows get up to three plain-English reasons and one counterfactual hint."
    )
    labels = [TYPICAL, *list(PRESETS)]
    examples = [preset_payload(name, bundle["reference"], PRESETS) for name in labels]
    template = application_template(list(bundle["feature_columns"]), examples, labels)
    st.download_button(
        "Download a template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="application_template.csv",
        mime="text/csv",
    )
    threshold = st.slider(
        "Cutoff for this batch",
        min_value=0.05,
        max_value=0.95,
        value=float(bundle["threshold"]),
        step=0.01,
        key="batch_threshold",
    )
    upload = st.file_uploader("Applications CSV", type="csv")
    if upload is None:
        st.info("Upload a CSV, or start from the template. Nothing is stored.")
        return
    frame = pd.read_csv(upload)
    try:
        scored = score_batch(
            bundle["pipeline"],
            frame,
            reference=bundle["reference"],
            feature_columns=list(bundle["feature_columns"]),
            threshold=float(threshold),
            ranges=bundle.get("ranges"),
            calibrator=bundle.get("calibrator"),
        )
    except Exception as exc:
        st.error(str(exc))
        return
    approved = int((scored["decision"] == "Approve").sum()) if len(scored) else 0
    declined = int((scored["decision"] == "Decline").sum()) if len(scored) else 0
    _metric_row(
        [
            ("Rows", f"{len(scored):,}", "Rows in the uploaded file."),
            ("Approvals", f"{approved:,}", "Rows at or above the cutoff."),
            ("Declines", f"{declined:,}", "Rows below the cutoff."),
        ]
    )
    st.dataframe(scored, width="stretch")
    st.download_button(
        "Download scored CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_applications.csv",
        mime="text/csv",
    )


def _limits_tab(card: dict) -> None:
    st.subheader("Where the cutoff concentrates approvals")
    st.caption(
        "Out-of-fold training scores at the saved cutoff. Each loan was scored by a model "
        "that did not train on it. The ratio versus the reference group is a screening "
        "arithmetic. It is not a legal conclusion, and this extract has no sex, age, or race."
    )
    table = _fairness_view(card.get("fairness_oof") or [])
    if table.empty:
        st.write("No segment table in the model card.")
    else:
        st.dataframe(table, width="stretch", hide_index=True)
    outside = _outside_four_fifths(card.get("fairness_oof") or [])
    if outside:
        st.markdown("**Segments outside a 0.80–1.25 ratio versus their reference**")
        for item in outside:
            st.write(f"- {item}")
    st.subheader("Limits")
    for item in LIMITATIONS:
        st.write(f"- {item}")
    if MODEL_CARD_MD_PATH.exists():
        with st.expander("Full model card"):
            st.markdown(MODEL_CARD_MD_PATH.read_text(encoding="utf-8"))


def _metric_row(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items, strict=True):
        column.metric(label, value, help=help_text)


def _signed_dollars(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${dollars(abs(value))}"


def _fairness_view(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    needed = ["attribute", "group", "n", "base_rate", "approval_rate", "disparate_impact"]
    if any(column not in frame.columns for column in needed):
        return pd.DataFrame()
    view = frame[needed].copy()
    view["attribute"] = view["attribute"].map(lambda value: ATTRIBUTE_LABELS.get(value, value))
    view["base_rate"] = view["base_rate"].map(_percent_cell)
    view["approval_rate"] = view["approval_rate"].map(_percent_cell)
    view["disparate_impact"] = view["disparate_impact"].map(
        lambda value: (
            "—"
            if value is None or (isinstance(value, float) and math.isnan(value))
            else f"{float(value):.2f}"
        )
    )
    return view.rename(
        columns={
            "attribute": "Segment",
            "group": "Group",
            "n": "n",
            "base_rate": "Fully paid",
            "approval_rate": "Approval rate",
            "disparate_impact": "Ratio vs reference",
        }
    )


def _outside_four_fifths(rows: list[dict]) -> list[str]:
    lines = []
    for row in rows:
        if row.get("outside_four_fifths") is not True:
            continue
        attribute = ATTRIBUTE_LABELS.get(row.get("attribute"), row.get("attribute"))
        ratio = row.get("disparate_impact")
        ratio_text = "—" if ratio is None else f"{float(ratio):.2f}"
        lines.append(f"{attribute}, {row.get('group')}: ratio {ratio_text} (n={row.get('n')})")
    return lines


def _percent_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.1%}"


def _load_example(bundle: dict) -> None:
    options = [TYPICAL, *PRESETS]
    if "loaded_preset" not in st.session_state:
        st.session_state.loaded_preset = TYPICAL
        _write_fields(
            preset_payload(TYPICAL, bundle["reference"], PRESETS),
            bundle["feature_columns"],
            bundle.get("ranges") or {},
        )
    choice = st.selectbox("Start from an example", options, key="preset_choice")
    if choice != st.session_state.loaded_preset:
        st.session_state.loaded_preset = choice
        _write_fields(
            preset_payload(choice, bundle["reference"], PRESETS),
            bundle["feature_columns"],
            bundle.get("ranges") or {},
        )
        st.rerun()
    st.caption(
        "Edits score immediately. Interest rate and the old Lending Club credit-policy flag "
        "are not on this form: both already contain a lender's judgment, and the deployed "
        "model does not use them."
    )


def _write_fields(payload: dict, columns: list[str], ranges: dict) -> None:
    for column in columns:
        if column not in payload:
            continue
        value = payload[column]
        if column in CATEGORY_OPTIONS:
            st.session_state[f"field_{column}"] = str(value)
            continue
        low, high, _step = _bounds(column, ranges)
        shown = _to_display(column, value)
        if column in INTEGER_FIELDS:
            shown = int(round(float(shown)))
            st.session_state[f"field_{column}"] = min(max(shown, int(low)), int(high))
        else:
            st.session_state[f"field_{column}"] = min(max(float(shown), float(low)), float(high))


def _application_fields(columns: list[str], ranges: dict) -> dict[str, object]:
    left, right = st.columns(2)
    left.subheader("The request")
    right.subheader("Credit file")
    payload: dict[str, object] = {}
    present = set(columns)
    with left:
        for field in REQUEST_FIELDS:
            if field in present:
                payload[field] = _widget(field, ranges)
    with right:
        for field in CREDIT_FIELDS:
            if field in present:
                payload[field] = _widget(field, ranges)
    return payload


def _widget(field: str, ranges: dict) -> object:
    label = FEATURE_LABELS[field]
    if field in CATEGORY_OPTIONS:
        return st.selectbox(
            label,
            CATEGORY_OPTIONS[field],
            format_func=lambda value, name=field: VALUE_LABELS[name].get(value, value),
            key=f"field_{field}",
        )
    low, high, step = _bounds(field, ranges)
    kwargs: dict[str, object] = {
        "label": label,
        "min_value": int(low) if field in INTEGER_FIELDS else float(low),
        "max_value": int(high) if field in INTEGER_FIELDS else float(high),
        "step": int(step) if field in INTEGER_FIELDS else None,
        "key": f"field_{field}",
    }
    if field == "log_annual_inc":
        kwargs["help"] = "Dollars a year. The model stores the natural log of income."
    elif field == "days_with_cr_line":
        kwargs["help"] = "Years since the oldest credit line. The model stores days."
    elif field == "installment":
        kwargs["help"] = "Monthly payment on the loan being requested."
    displayed = st.number_input(**kwargs)
    return _to_model(field, displayed)


def _bounds(column: str, ranges: dict) -> tuple[float, float, float]:
    fallback = FALLBACK_BOUNDS[column]
    observed = ranges.get(column)
    if not observed:
        return fallback
    low = float(observed["min"])
    high = float(observed["max"])
    if column == "log_annual_inc":
        return (
            max(1000.0, float(math.floor(math.exp(low)))),
            float(math.ceil(math.exp(high))),
            500.0,
        )
    if column == "days_with_cr_line":
        return (round(max(low, 1.0) / 365.25, 1), round(high / 365.25, 1), 0.1)
    step = float(fallback[2])
    if column in INTEGER_FIELDS:
        return (float(math.floor(low)), float(math.ceil(high)), step)
    return (low, high, step)


def _to_display(column: str, model_value: object) -> float:
    number = float(model_value)
    if column == "log_annual_inc":
        return float(round(math.exp(number)))
    if column == "days_with_cr_line":
        return round(number / 365.25, 1)
    if column in INTEGER_FIELDS:
        return float(int(round(number)))
    return number


def _to_model(column: str, displayed: object) -> float:
    number = float(displayed)
    if column == "log_annual_inc":
        return math.log(max(number, 1.0))
    if column == "days_with_cr_line":
        return number * 365.25
    return number


def _sidebar(card: dict, bundle: dict) -> None:
    holdout = card.get("holdout", {})
    winner = card.get("winner", {})
    policy = card.get("policy", {})
    money = card.get("money", {})
    calibration = card.get("calibration", {})
    interval = winner.get("holdout_roc_auc_ci") or [None, None]
    with st.sidebar:
        st.header("Model card")
        st.write(winner.get("display_name", bundle["model_name"]))
        st.caption(winner.get("selection_note", ""))
        auc = float(holdout.get("roc_auc", 0))
        st.metric("Holdout ROC-AUC", f"{auc:.3f}")
        if interval[0] is not None and interval[1] is not None:
            st.caption(
                f"95% bootstrap interval {float(interval[0]):.3f} to {float(interval[1]):.3f}."
            )
        st.metric(
            "Brier, calibrated",
            f"{float(calibration.get('holdout_brier_calibrated', holdout.get('brier', 0))):.3f}",
            help=(
                "Lower is better. Compare with the constant-rate Brier "
                f"{float(calibration.get('holdout_brier_base_rate', 0)):.3f}."
            ),
        )
        st.metric("Holdout F1", f"{float(holdout.get('f1', 0)):.3f}")
        st.metric(
            "Holdout planning cost / loan",
            f"${float(holdout.get('cost_per_applicant', 0)):,.2f}",
        )
        if money:
            saved = float(money.get("saved_total_vs_better_naive", 0))
            rule = str(money.get("better_naive_rule", "the cheaper naive rule")).replace("_", " ")
            st.metric(
                f"Holdout $ vs {rule}",
                f"${saved:,.0f}",
                help="Positive means the scored cutoff costs less than that rule on the holdout.",
            )
        split = card.get("split", {})
        st.caption(
            f"Saved cutoff {float(bundle['threshold']):.0%}. "
            f"Train {split.get('n_train', '—')} / holdout {split.get('n_test', '—')}."
        )
        excluded = ", ".join(policy.get("excluded_features", []))
        st.markdown("**Withheld from the score**")
        st.write(excluded or "None")
        st.caption(policy.get("reason", ""))
        with st.expander("Planning cost"):
            st.markdown(
                f"Funding a loan that is not fully paid is priced at **USD {FP_COST:,.0f}**. "
                "Declining a loan that would have been fully paid is priced at "
                f"**USD {FN_COST:,.0f}**. "
                "These are planning weights. The file has no charge-off balance."
            )
            st.caption(policy.get("cost_source", ""))
        st.markdown("**Dataset**")
        st.caption(DATASET["citation"])
        st.caption(DATASET["license"])


def _render_decision(result: dict, card: dict) -> None:
    st.divider()
    narrative = result.get("narrative") or result["summary"]
    if result["decision"] == "Approve":
        st.success(narrative)
    else:
        st.error(narrative)
    if result["refer_to_analyst"]:
        st.warning(
            "This file sits close to the cutoff. Route it to an analyst instead of "
            "treating the stamp as final."
        )
    if result["out_of_range"]:
        labels = ", ".join(FEATURE_LABELS.get(field, field) for field in result["out_of_range"])
        st.warning(f"Outside the training range: {labels}.")

    probability = result.get("probability_fully_paid", result["probability_good"])
    margin = probability - float(result["threshold"])
    _metric_row(
        [
            ("Decision", result["decision"], "Approve when the score clears the cutoff."),
            ("Calibrated score", f"{probability:.1%}", "Probability the loan is fully paid."),
            ("Cutoff", f"{result['threshold']:.1%}", "The slider, not a refit."),
            ("Margin", f"{margin:+.1%}", "Score minus the cutoff."),
        ]
    )

    if result["decision"] == "Decline":
        st.subheader("Principal reasons for the decline")
        st.caption(
            "Educational analogue of an adverse-action note. Each reason is a field that "
            "lowers the score relative to a typical training file. It is not a FCRA notice "
            "and it is not a cause."
        )
        _reason_list(
            result.get("adverse_reasons") or [],
            "No single field lowered the score by more than half a point.",
        )
        hint = (result.get("counterfactual") or {}).get("text")
        if hint:
            st.info(hint)

    support, risk = st.columns(2)
    with support:
        st.subheader("Pushed toward approval")
        _reason_list(
            result["reasons_for"],
            "No field clearly supports approval versus a typical file.",
        )
    with risk:
        st.subheader("Pushed toward a decline")
        _reason_list(result["reasons_against"], "No field clearly adds risk versus a typical file.")

    figure = contribution_figure(result["contributions"])
    st.pyplot(figure)
    plt.close(figure)
    baselines = card.get("holdout_baselines", {})
    st.caption(
        "Each bar is the change in the ranking score, before the probability map, when "
        "that field is replaced with the training median or the most common purpose. "
        "The map is flat in places, so a field can matter here and not move the "
        "calibrated probability. Correlated fields share the credit. "
        "Holdout decline-all planning cost is "
        f"USD {dollars(float(baselines.get('decline_all', 0)))} "
        f"per loan. Source: {DATASET['name']}, {DATASET['license']}."
    )


def _reason_list(reasons: list[dict], empty: str) -> None:
    if not reasons:
        st.write(empty)
        return
    for reason in reasons:
        st.write(f"- {reason['text']}")


if __name__ == "__main__":
    main()
