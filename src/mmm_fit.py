"""
Marketing Mix Model Fitting
------------------------------------------------------------------------
For each channel, grid-searches over candidate (adstock_decay, alpha)
pairs, transforms spend accordingly, and fits a linear regression of
sales on [baseline trend/seasonality features + transformed channel
spends]. Selects the parameter combination with the best R-squared.

This mirrors real MMM practice: adstock/saturation parameters are not
directly estimable by OLS (they're inside a nonlinear transform), so
they're searched over a grid while the LINEAR coefficients (beta) on
the transformed features are fit by OLS at each grid point.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

CHANNELS = ["TV", "Digital", "Social", "Print"]

DECAY_GRID = np.arange(0.05, 0.85, 0.05)
ALPHA_GRID = np.arange(0.3, 0.95, 0.05)


def apply_adstock(spend, decay):
    adstocked = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked


def apply_saturation(adstocked_spend, alpha):
    return np.power(np.maximum(adstocked_spend, 0), alpha)


def _transform(spend, decay, alpha):
    return apply_saturation(apply_adstock(spend, decay), alpha)


def grid_search_channel_params(df, channel, baseline_features, other_channel_transforms):
    """Find the (decay, alpha) pair for THIS channel that gives the best fit
    of sales ~ baseline_features + OTHER channels' current transforms +
    this channel's candidate transform. Controlling for other channels
    avoids the confounding bug where a single-channel search absorbs other
    channels' true contributions into its own (distorted) parameter estimate."""
    spend = df[f"spend_{channel}"].values
    y = df["sales"].values
    best_r2 = -np.inf
    best_params = None

    other_cols = list(other_channel_transforms.values())
    control_X = np.column_stack([baseline_features] + other_cols) if other_cols else baseline_features

    for decay in DECAY_GRID:
        adstocked = apply_adstock(spend, decay)
        for alpha in ALPHA_GRID:
            transformed = apply_saturation(adstocked, alpha)
            X = np.column_stack([control_X, transformed])
            model = LinearRegression().fit(X, y)
            r2 = r2_score(y, model.predict(X))
            if r2 > best_r2:
                best_r2 = r2
                best_params = {"decay": decay, "alpha": alpha, "r2": r2}

    return best_params


def fit_full_mmm(df, n_passes=3):
    week_num = df["week"].values
    seasonal_sin = np.sin(2 * np.pi * week_num / 52)
    seasonal_cos = np.cos(2 * np.pi * week_num / 52)
    baseline_features = np.column_stack([week_num, seasonal_sin, seasonal_cos])

    # Initialize all channels with a neutral starting transform
    current_params = {c: {"decay": 0.3, "alpha": 0.6} for c in CHANNELS}

    # Coordinate descent: repeatedly re-fit each channel's params while
    # holding the OTHERS fixed at their current best estimate, for several
    # passes, so estimates converge rather than being computed in one
    # confounded shot.
    for pass_num in range(n_passes):
        for channel in CHANNELS:
            other_transforms = {
                c: _transform(df[f"spend_{c}"].values, p["decay"], p["alpha"])
                for c, p in current_params.items() if c != channel
            }
            best = grid_search_channel_params(df, channel, baseline_features, other_transforms)
            current_params[channel] = {"decay": best["decay"], "alpha": best["alpha"], "r2": best["r2"]}

    # Final joint model: baseline + ALL channels' converged transforms together
    transformed_cols = [
        _transform(df[f"spend_{c}"].values, current_params[c]["decay"], current_params[c]["alpha"])
        for c in CHANNELS
    ]
    X_full = np.column_stack([baseline_features] + transformed_cols)
    y = df["sales"].values
    joint_model = LinearRegression().fit(X_full, y)
    joint_r2 = r2_score(y, joint_model.predict(X_full))

    channel_betas = {channel: joint_model.coef_[3 + i] for i, channel in enumerate(CHANNELS)}

    return {
        "per_channel_search": current_params,
        "joint_model": joint_model,
        "joint_r2": joint_r2,
        "channel_betas": channel_betas,
        "baseline_coef": joint_model.coef_[:3],
        "intercept": joint_model.intercept_,
    }


def report(df, fit_results, true_params):
    print("=" * 72)
    print("MARKETING MIX MODEL: FITTED VS. TRUE PARAMETERS")
    print("=" * 72)
    print(f"\nJoint model R-squared: {fit_results['joint_r2']:.4f}\n")

    print(f"{'Channel':<10}{'True Decay':>12}{'Fit Decay':>12}{'True Alpha':>12}{'Fit Alpha':>12}{'Fit Beta':>12}")
    for channel in CHANNELS:
        p = fit_results["per_channel_search"][channel]
        true_p = true_params[channel]
        fit_beta = fit_results["channel_betas"][channel]
        print(f"{channel:<10}{true_p['adstock_decay']:>12.2f}{p['decay']:>12.2f}"
              f"{true_p['alpha']:>12.2f}{p['alpha']:>12.2f}{fit_beta:>12.4f}")

    print("\n" + "-" * 72)
    print("VALIDATION NOTE")
    print("-" * 72)
    print("Decay/alpha are grid-searched at 0.05 resolution, so small deviations")
    print("from true values are expected -- what matters is directional recovery")
    print("(e.g. TV correctly shows the highest decay = most carryover).\n")


if __name__ == "__main__":
    from data_generator import TRUE_PARAMS
    df = pd.read_csv("/home/claude/mmm_project/data/mmm_data.csv")
    fit_results = fit_full_mmm(df)
    report(df, fit_results, TRUE_PARAMS)
