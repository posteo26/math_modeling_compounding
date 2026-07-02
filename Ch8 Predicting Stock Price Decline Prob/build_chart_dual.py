# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 19:49:21 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ============================================
# 1. Load data
# ============================================
df = pd.read_csv("msft_rolling_decline_backtest_fast_features.csv", parse_dates=["pred_date"])

# Filter for 3m horizon and -10% threshold
df = df[(df["horizon"] == "6m") & (df["threshold"] == -0.10)].copy()

# ============================================
# 2. Compute actual future returns if missing
# ============================================
if "actual_decline" not in df.columns or df["actual_decline"].isna().all():

    px = yf.download("MSFT", start="2010-01-01", end="2026-12-31", auto_adjust=True)["Close"]
    px = px.sort_index()

    horizon_days = 63  # 3 months

    actual_returns = []
    actual_declines = []

    for date in df["pred_date"]:
        if date not in px.index:
            actual_returns.append(np.nan)
            actual_declines.append(np.nan)
            continue

        idx = px.index.get_loc(date)
        future_idx = idx + horizon_days

        if future_idx >= len(px):
            actual_returns.append(np.nan)
            actual_declines.append(np.nan)
            continue

        future_date = px.index[future_idx]
        ret = px.loc[future_date] / px.loc[date] - 1

        actual_returns.append(ret)
        actual_declines.append(int(ret <= -0.10))

    df["actual_fut_ret"] = actual_returns
    df["actual_decline"] = actual_declines

# ============================================
# 3. Prepare colors and sizes for scatter
# ============================================
colors = df["actual_decline"].map({
    1: "red",
    0: "green"
}).fillna("gray")

sizes = df["actual_decline"].map({
    1: 150,   # big red dot
    0: 70     # medium green dot
}).fillna(40)  # small gray dot

# ============================================
# 4. Plot RF vs ET probabilities with markers on both lines
# ============================================
plt.figure(figsize=(15, 7))

# RF line + markers
plt.plot(df["pred_date"], df["prob_decline_rf"],
         color="blue", linewidth=2, label="RF predicted probability")

plt.scatter(df["pred_date"], df["prob_decline_rf"],
            c=colors, s=sizes, marker="o", edgecolor="black", linewidth=0.7)

# ET line + markers
plt.plot(df["pred_date"], df["prob_decline_et"],
         color="orange", linewidth=2, label="ET predicted probability")

plt.scatter(df["pred_date"], df["prob_decline_et"],
            c=colors, s=sizes, marker="s", edgecolor="black", linewidth=0.7)

# Labels and title
plt.title("RF vs ET Decline Probabilities Over Time (6m Horizon, -10% Threshold)\nWith Actual Outcomes Highlighted")
plt.xlabel("Prediction Date")
plt.ylabel("Predicted Probability of Decline")
plt.grid(True, alpha=0.3)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='RF predicted probability'),
    Line2D([0], [0], color='orange', lw=2, label='ET predicted probability'),
    Line2D([0], [0], marker='o', color='w', label='RF markers',
           markerfacecolor='blue', markersize=10, markeredgecolor="black"),
    Line2D([0], [0], marker='s', color='w', label='ET markers',
           markerfacecolor='orange', markersize=10, markeredgecolor="black"),
    Line2D([0], [0], marker='o', color='w', label='Decline happened (1)',
           markerfacecolor='red', markersize=12, markeredgecolor="black"),
    Line2D([0], [0], marker='o', color='w', label='No decline (0)',
           markerfacecolor='green', markersize=9, markeredgecolor="black"),
    Line2D([0], [0], marker='o', color='w', label='Outcome unknown',
           markerfacecolor='gray', markersize=7, markeredgecolor="black")
]

plt.legend(handles=legend_elements, loc="upper left")

plt.tight_layout()
plt.show()