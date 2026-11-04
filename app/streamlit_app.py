"""Educational credit desk. Run from the repo root: streamlit run app/streamlit_app.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from loan_approval.artifacts import load_bundle, load_model_card
from loan_approval.config import (
    CATEGORY_OPTIONS,
    DATASET,
    FEATURE_LABELS,
    MODEL_CARD_PATH,
    MODEL_PATH,
    PRESETS,
    VALUE_LABELS,
)
from loan_approval.explain import explain_application, preset_payload
from loan_approval.plots import contribution_figure

TYPICAL = "Typical training profile"
NUMERIC_BOUNDS = {
    "duration": (4, 72, 1),
    "credit_amount": (250, 20000, 1),
    "installment_commitment": (1, 4, 1),
    "residence_since": (1, 4, 1),
    "age": (18, 80, 1),
    "existing_credits": (1, 4, 1),
    "num_dependents": (1, 2, 1),
}


def main() -> None:
    st.set_page_config(page_title="Credit Decision Desk", page_icon="🏦", layout="wide")
    st.title("Credit Decision Desk")
    st.caption(
        "Educational scorer for a business analytics portfolio. "
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
    payload = _application_fields(bundle["feature_columns"])
    threshold = st.slider(
        "Operating cutoff: approve when P(good credit) is at least",
        min_value=0.05,
        max_value=0.95,
        value=float(bundle["threshold"]),
        step=0.01,
        help=(
            "The default cutoff minimizes the dataset's published cost matrix "
            "(5 for approving a bad credit, 1 for declining a good credit) "
            "on training out-of-fold scores. Moving it changes the decision, not the probability."
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
            preset_payload(TYPICAL, bundle["reference"], PRESETS), bundle["feature_columns"]
        )
    choice = st.selectbox("Start from an example", options, key="preset_choice")
    if choice != st.session_state.loaded_preset:
        st.session_state.loaded_preset = choice
        _write_fields(
            preset_payload(choice, bundle["reference"], PRESETS),
            bundle["feature_columns"],
        )
        st.rerun()
    st.caption(
        "Edits score immediately. Personal status and foreign-worker status are not collected: "
        "they are withheld from the deployed model and used only in the fairness audit."
    )


def _write_fields(payload: dict, columns: list[str]) -> None:
    for column in columns:
        value = payload[column]
        if column in CATEGORY_OPTIONS:
            st.session_state[f"field_{column}"] = str(value)
            continue
        low, high, _step = NUMERIC_BOUNDS[column]
        number = int(round(float(value)))
        st.session_state[f"field_{column}"] = min(max(number, low), high)


def _application_fields(columns: list[str]) -> dict[str, object]:
    left, right = st.columns(2)
    groups = {
        left: [
            "credit_amount",
            "duration",
            "purpose",
            "installment_commitment",
            "checking_status",
            "savings_status",
            "credit_history",
            "other_payment_plans",
            "existing_credits",
        ],
        right: [
            "employment",
            "job",
            "age",
            "housing",
            "residence_since",
            "property_magnitude",
            "other_parties",
            "num_dependents",
            "own_telephone",
        ],
    }
    left.subheader("The request")
    right.subheader("Work and household")
    payload: dict[str, object] = {}
    for host, fields in groups.items():
        with host:
            for field in fields:
                if field not in columns:
                    continue
                payload[field] = _widget(field)
    return payload


def _widget(field: str) -> object:
    label = FEATURE_LABELS[field]
    if field in CATEGORY_OPTIONS:
        return st.selectbox(
            label,
            CATEGORY_OPTIONS[field],
            format_func=lambda value, name=field: VALUE_LABELS[name].get(value, value),
            key=f"field_{field}",
        )
    low, high, step = NUMERIC_BOUNDS[field]
    return st.number_input(
        label,
        min_value=low,
        max_value=high,
        step=step,
        key=f"field_{field}",
    )


def _sidebar(card: dict, bundle: dict) -> None:
    holdout = card.get("holdout", {})
    winner = card.get("winner", {})
    policy = card.get("policy", {})
    with st.sidebar:
        st.header("Model card")
        st.write(winner.get("display_name", bundle["model_name"]))
        st.metric("Holdout ROC-AUC", f"{float(holdout.get('roc_auc', 0)):.3f}")
        st.metric("Holdout F1", f"{float(holdout.get('f1', 0)):.3f}")
        st.metric("Holdout cost / applicant", f"{float(holdout.get('cost_per_applicant', 0)):.3f}")
        split = card.get("split", {})
        st.caption(
            f"Saved cutoff {float(bundle['threshold']):.0%}. "
            f"Train {split.get('n_train', '—')} / holdout {split.get('n_test', '—')}."
        )
        excluded = ", ".join(policy.get("excluded_features", []))
        st.markdown("**Withheld from the score**")
        st.write(excluded or "None")
        st.caption(policy.get("reason", ""))
        with st.expander("Cost matrix"):
            st.markdown(
                "Approving a bad credit costs **5**. Declining a good credit costs **1**. "
                "Those weights are the ones published with the UCI file, not a currency forecast."
            )
        st.markdown("**Dataset**")
        st.caption(DATASET["citation"])
        st.caption(DATASET["license"])


def _render_decision(result: dict, card: dict) -> None:
    st.divider()
    if result["decision"] == "Approve":
        st.success(result["summary"])
    else:
        st.error(result["summary"])
    if result["refer_to_analyst"]:
        st.warning(
            "This file sits close to the cutoff. Route it to an analyst instead of "
            "treating the stamp as final."
        )
    if result["out_of_range"]:
        labels = ", ".join(FEATURE_LABELS.get(field, field) for field in result["out_of_range"])
        st.warning(f"Outside the training range: {labels}.")

    probability_col, cutoff_col, baseline_col = st.columns(3)
    probability_col.metric("P(good credit)", f"{result['probability_good']:.1%}")
    cutoff_col.metric("Cutoff in use", f"{result['threshold']:.1%}")
    baselines = card.get("holdout_baselines", {})
    baseline_col.metric(
        "Holdout cost, decline-all",
        f"{float(baselines.get('decline_all', 0)):.3f}",
        help=(
            "Average Statlog cost of declining every holdout applicant. "
            "A useful model has to beat this."
        ),
    )

    support, risk = st.columns(2)
    with support:
        st.subheader("Supports approval")
        _reason_list(result["reasons_for"], "No field is clearly above a typical file.")
    with risk:
        st.subheader("Raises risk")
        _reason_list(result["reasons_against"], "No field is clearly riskier than a typical file.")

    figure = contribution_figure(result["contributions"])
    st.pyplot(figure)
    import matplotlib.pyplot as plt

    plt.close(figure)
    st.caption(
        "Each bar is the change in probability when that field is replaced with the "
        "training median or the most common category. Correlated fields share credit, "
        "so read the list as a review aid, not as a causal explanation. "
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
