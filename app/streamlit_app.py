"""Educational loan desk. Run from the repo root: streamlit run streamlit_app.py"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import streamlit as st

from loan_approval.artifacts import load_bundle, load_model_card
from loan_approval.config import (
    CATEGORY_OPTIONS,
    DATASET,
    FEATURE_LABELS,
    FN_COST,
    FP_COST,
    MODEL_CARD_PATH,
    MODEL_PATH,
    PRESETS,
    VALUE_LABELS,
)
from loan_approval.explain import explain_application, preset_payload
from loan_approval.metrics import dollars
from loan_approval.plots import contribution_figure

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
    st.title("Loan Decision Desk")
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

    bundle = load_bundle(MODEL_PATH)
    card = load_model_card(MODEL_CARD_PATH)
    _sidebar(card, bundle)
    _load_example(bundle)
    payload = _application_fields(bundle["feature_columns"], bundle.get("ranges") or {})
    threshold = st.slider(
        "Operating cutoff: approve when the model score is at least",
        min_value=0.05,
        max_value=0.95,
        value=float(bundle["threshold"]),
        step=0.01,
        help=(
            f"The default cutoff minimizes a planning cost of ${FP_COST:,.0f} for funding "
            f"a loan that is not fully paid and ${FN_COST:,.0f} for declining a loan that "
            "would have been fully paid. Those are scenario weights, not observed losses. "
            "Moving the slider changes the decision, not the score. The score is not a "
            "calibrated probability: this model uses balanced class weights."
        ),
    )
    result = explain_application(
        bundle["pipeline"],
        payload,
        reference=bundle["reference"],
        feature_columns=list(bundle["feature_columns"]),
        threshold=float(threshold),
        ranges=bundle.get("ranges"),
        review_band=float(bundle.get("review_band", 0.05)),
    )
    _render_decision(result, card)


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
    with st.sidebar:
        st.header("Model card")
        st.write(winner.get("display_name", bundle["model_name"]))
        st.metric("Holdout ROC-AUC", f"{float(holdout.get('roc_auc', 0)):.3f}")
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
                f"Funding a loan that is not fully paid is priced at **${FP_COST:,.0f}**. "
                "Declining a loan that would have been fully paid is priced at "
                f"**${FN_COST:,.0f}**. "
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

    probability_col, cutoff_col, baseline_col = st.columns(3)
    probability = result.get("probability_fully_paid", result["probability_good"])
    probability_col.metric("Model score, fully paid", f"{probability:.1%}")
    cutoff_col.metric("Cutoff in use", f"{result['threshold']:.1%}")
    baselines = card.get("holdout_baselines", {})
    baseline_col.metric(
        "Holdout cost, decline-all",
        f"${dollars(float(baselines.get('decline_all', 0)))}",
        help="Planning cost per holdout loan of declining every application.",
    )

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
    st.caption(
        "Each bar is the change in the model score when that field is replaced with the "
        "training median or the most common purpose. The score is not a calibrated "
        "probability. Correlated fields share the credit, so read this as a review note, "
        "not as a cause and not as an adverse-action letter. "
        f"Source: {DATASET['name']}, {DATASET['license']}."
    )


def _reason_list(reasons: list[dict], empty: str) -> None:
    if not reasons:
        st.write(empty)
        return
    for reason in reasons:
        st.write(f"- {reason['text']}")


if __name__ == "__main__":
    main()
