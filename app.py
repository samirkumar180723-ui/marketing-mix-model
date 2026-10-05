"""
Marketing Mix Model & Budget Optimizer -- Streamlit Dashboard
------------------------------------------------------------------------
Run locally with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_generator import TRUE_PARAMS
from mmm_fit import fit_full_mmm, CHANNELS, _transform
from budget_optimizer import optimize_scipy, steady_state_response

st.set_page_config(page_title="Marketing Mix Model & Budget Optimizer", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "mmm_data.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def get_fit(df):
    return fit_full_mmm(df)


df = load_data()
fit_results = get_fit(df)

st.title("📊 Marketing Mix Model & Budget Optimizer")
st.caption(
    "All data is SYNTHETIC, generated with known ground-truth adstock/saturation "
    "parameters so the fitted model can be validated against them directly. "
    "Methodology: grid search + coordinate descent for adstock/saturation params, "
    "with a bounded-reallocation safeguard on the budget optimizer (explained in Tab 3)."
)

tab1, tab2, tab3 = st.tabs(["📈 Model Fit & Validation", "🎯 Channel Contribution", "💰 Budget Optimizer"])

# ------------------------------------------------------------------
# TAB 1: MODEL FIT & VALIDATION
# ------------------------------------------------------------------
with tab1:
    st.subheader("Fitted vs. True Parameters")
    st.markdown(f"**Joint model R-squared: {fit_results['joint_r2']:.4f}**")

    rows = []
    for c in CHANNELS:
        p = fit_results["per_channel_search"][c]
        true_p = TRUE_PARAMS[c]
        rows.append({
            "Channel": c, "True Decay": true_p["adstock_decay"], "Fit Decay": p["decay"],
            "True Alpha": true_p["alpha"], "Fit Alpha": p["alpha"],
            "Fit Beta": fit_results["channel_betas"][c],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.warning(
        "⚠️ **Honest limitation**: Digital and Social's fitted alpha hits the grid "
        "search boundary (0.90), indicating these channels' saturation curvature is "
        "weakly identified from this data -- a known, real MMM challenge, not a bug. "
        "This directly affects the Budget Optimizer tab, where it's handled explicitly."
    )

    st.subheader("Actual vs. Predicted Sales")
    week_num = df["week"].values
    baseline_features = np.column_stack([
        week_num, np.sin(2 * np.pi * week_num / 52), np.cos(2 * np.pi * week_num / 52)
    ])
    transformed = [
        _transform(df[f"spend_{c}"].values, fit_results["per_channel_search"][c]["decay"],
                   fit_results["per_channel_search"][c]["alpha"])
        for c in CHANNELS
    ]
    X_full = np.column_stack([baseline_features] + transformed)
    predicted = fit_results["joint_model"].predict(X_full)

    chart_df = pd.DataFrame({"Actual Sales": df["sales"], "Predicted Sales": predicted})
    st.line_chart(chart_df)

# ------------------------------------------------------------------
# TAB 2: CHANNEL CONTRIBUTION
# ------------------------------------------------------------------
with tab2:
    st.subheader("Estimated Contribution to Sales by Channel")

    contributions = {}
    for i, c in enumerate(CHANNELS):
        contributions[c] = fit_results["channel_betas"][c] * np.mean(transformed[i])
    baseline_contribution = np.mean(X_full[:, :3] @ fit_results["baseline_coef"]) + fit_results["intercept"]

    contrib_df = pd.DataFrame({
        "Source": list(contributions.keys()) + ["Baseline (trend/seasonality)"],
        "Avg Weekly Contribution": list(contributions.values()) + [baseline_contribution],
    })
    st.bar_chart(contrib_df.set_index("Source"))
    st.dataframe(contrib_df.style.format({"Avg Weekly Contribution": "₹{:,.0f}"}), use_container_width=True)

# ------------------------------------------------------------------
# TAB 3: BUDGET OPTIMIZER
# ------------------------------------------------------------------
with tab3:
    st.subheader("Interactive Budget Allocation")

    channel_params = {
        c: {
            "decay": fit_results["per_channel_search"][c]["decay"],
            "alpha": fit_results["per_channel_search"][c]["alpha"],
            "beta": fit_results["channel_betas"][c],
        }
        for c in CHANNELS
    }

    st.markdown("**Current weekly spend per channel:**")
    cols = st.columns(4)
    current_spends = {}
    defaults = {"TV": 15000, "Digital": 8000, "Social": 5000, "Print": 3000}
    for i, c in enumerate(CHANNELS):
        with cols[i]:
            current_spends[c] = st.number_input(c, min_value=0, value=defaults[c], step=500)

    total_budget = sum(current_spends.values())
    max_change_pct = st.slider(
        "Max reallocation bound (% of current spend)", 0.1, 1.0, 0.4, 0.05,
        help="Limits how far the optimizer can shift budget away from current spend, "
             "avoiding unreliable extrapolation of weakly-identified response curves."
    )

    if st.button("Optimize Budget"):
        unconstrained_alloc, unconstrained_response = optimize_scipy(channel_params, total_budget)
        bounded_alloc, bounded_response = optimize_scipy(
            channel_params, total_budget, current_spends=current_spends, max_change_pct=max_change_pct
        )
        current_response = sum(
            steady_state_response(current_spends[c], p["decay"], p["alpha"], p["beta"])
            for c, p in channel_params.items()
        )

        st.error(
            f"⚠️ **Unconstrained optimum** would shift budget to: "
            + ", ".join(f"{c}: ₹{unconstrained_alloc[c]:,.0f}" for c in CHANNELS)
            + " -- flagged as unreliable (extreme concentration signals weak parameter identification, not real opportunity)."
        )

        st.success(f"✅ **Realistic, bounded recommendation** (±{max_change_pct:.0%} of current spend):")
        result_df = pd.DataFrame({
            "Channel": CHANNELS,
            "Current": [current_spends[c] for c in CHANNELS],
            "Recommended": [bounded_alloc[c] for c in CHANNELS],
            "Change": [bounded_alloc[c] - current_spends[c] for c in CHANNELS],
        })
        st.dataframe(result_df.style.format({
            "Current": "₹{:,.0f}", "Recommended": "₹{:,.0f}", "Change": "{:+,.0f}",
        }), use_container_width=True)

        uplift = bounded_response - current_response
        uplift_pct = uplift / current_response * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Response", f"₹{current_response:,.0f}")
        col2.metric("Optimized Response", f"₹{bounded_response:,.0f}")
        col3.metric("Uplift", f"{uplift_pct:+.1f}%")

st.divider()
st.caption(
    "Built by Samir Kumar | Methodology: adstock + saturation transforms fitted via "
    "coordinate-descent grid search, validated against known synthetic ground truth. "
    "Budget optimization cross-validated via SLSQP + greedy marginal allocation, with "
    "a bounded-reallocation safeguard against unreliable curve extrapolation."
)
