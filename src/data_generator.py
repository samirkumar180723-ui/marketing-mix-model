"""
Synthetic Marketing Mix Model (MMM) Data Generator
------------------------------------------------------------------------
Generates 104 weeks of sales data driven by a baseline trend+seasonality
plus 4 marketing channels, each with:
  - Adstock (carryover): this week's effective spend includes a decayed
    fraction of past weeks' spend, modeling that ad effects linger.
  - Saturation (diminishing returns): effect = beta * (adstocked_spend ^ alpha),
    with alpha < 1, so each extra rupee of spend contributes less than the
    last (realistic -- you can't infinitely scale any one channel).

Channel betas were pre-calibrated (see calibration check in project notes)
so each channel contributes a sensible ~10% of baseline sales at its
typical spend level, BEFORE writing the full generator -- avoiding the
scale-mismatch bugs from earlier projects in this portfolio.

All data is SYNTHETIC with known ground-truth parameters, enabling direct
validation of the fitted MMM against the true generating process.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(1)

WEEKS = 104
BASE_LEVEL = 50_000
TREND_PER_WEEK = 80          # slow underlying growth
SEASONAL_AMPLITUDE = 6_000    # yearly seasonality swing

# True channel parameters (what the model should recover)
TRUE_PARAMS = {
    "TV":      {"adstock_decay": 0.60, "alpha": 0.6, "beta": 15.6068, "typical_spend": 15_000},
    "Digital": {"adstock_decay": 0.20, "alpha": 0.7, "beta": 9.2642,  "typical_spend": 8_000},
    "Social":  {"adstock_decay": 0.40, "alpha": 0.5, "beta": 70.7107, "typical_spend": 5_000},
    "Print":   {"adstock_decay": 0.10, "alpha": 0.8, "beta": 8.2656,  "typical_spend": 3_000},
}


def apply_adstock(spend, decay):
    adstocked = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked


def apply_saturation(adstocked_spend, alpha, beta):
    return beta * np.power(np.maximum(adstocked_spend, 0), alpha)


def generate_channel_spend(typical_spend, weeks=WEEKS, intermittent=False):
    if intermittent:
        # e.g. Print: campaigns run ~40% of weeks
        active = rng.binomial(1, 0.4, size=weeks)
        spend = active * rng.gamma(shape=3, scale=typical_spend / 3, size=weeks)
    else:
        spend = rng.gamma(shape=4, scale=typical_spend / 4, size=weeks)
    return spend


def generate_data():
    week_num = np.arange(WEEKS)
    seasonal = SEASONAL_AMPLITUDE * np.sin(2 * np.pi * week_num / 52)
    trend = TREND_PER_WEEK * week_num
    baseline_noise = rng.normal(0, 1500, size=WEEKS)
    baseline_sales = BASE_LEVEL + trend + seasonal + baseline_noise

    channel_spend = {}
    channel_contribution = {}

    for name, p in TRUE_PARAMS.items():
        intermittent = (name == "Print")
        spend = generate_channel_spend(p["typical_spend"], intermittent=intermittent)
        adstocked = apply_adstock(spend, p["adstock_decay"])
        contribution = apply_saturation(adstocked, p["alpha"], p["beta"])
        channel_spend[name] = spend
        channel_contribution[name] = contribution

    total_contribution = sum(channel_contribution.values())
    sales_noise = rng.normal(0, 800, size=WEEKS)
    total_sales = baseline_sales + total_contribution + sales_noise

    df = pd.DataFrame({"week": week_num, "sales": total_sales, "baseline_sales_true": baseline_sales})
    for name in TRUE_PARAMS:
        df[f"spend_{name}"] = channel_spend[name]
        df[f"true_contribution_{name}"] = channel_contribution[name]

    return df


if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/mmm_data.csv", index=False)

    print(f"Generated {len(df)} weeks of synthetic MMM data.\n")
    print("Sales summary:")
    print(df["sales"].describe())
    print("\nChannel contribution share of total sales (true, for validation later):")
    total_true_contribution = sum(df[f"true_contribution_{name}"].sum() for name in TRUE_PARAMS)
    total_sales_sum = df["sales"].sum()
    for name in TRUE_PARAMS:
        share = df[f"true_contribution_{name}"].sum() / total_sales_sum
        print(f"  {name}: {share:.1%} of total sales")
    print(f"  Baseline: {df['baseline_sales_true'].sum() / total_sales_sum:.1%} of total sales")
