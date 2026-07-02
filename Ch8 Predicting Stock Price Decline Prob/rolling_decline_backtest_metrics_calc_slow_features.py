# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 18:37:58 2026

"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix,
    roc_curve, auc
)
from sklearn.calibration import calibration_curve

# ============================================
# 1. Load CSV and download MSFT price data
# ============================================
df = pd.read_csv("msft_rolling_decline_backtest.csv", parse_dates=["pred_date"])

px = yf.download("MSFT", start="2010-01-01", end="2026-12-31", auto_adjust=True)["Close"]
px.name = "price"
px = px.sort_index()

# ============================================
# 2. Compute actual future returns and labels
# ============================================
horizon_days = {"3m": 63, "6m": 126, "12m": 252}

actual_returns = []
actual_declines = []

for i, row in df.iterrows():
    date = row["pred_date"]
    horizon = row["horizon"]
    threshold = row["threshold"]

    if date not in px.index:
        actual_returns.append(np.nan)
        actual_declines.append(np.nan)
        continue

    try:
        idx = px.index.get_loc(date)
        future_idx = idx + horizon_days[horizon]
        if future_idx >= len(px):
            raise IndexError
        future_date = px.index[future_idx]
        ret = px.loc[future_date] / px.loc[date] - 1
        actual_returns.append(ret)
        actual_declines.append(int(ret <= threshold))
    except:
        actual_returns.append(np.nan)
        actual_declines.append(np.nan)

df["actual_fut_ret"] = actual_returns
df["actual_decline"] = actual_declines

df.to_csv("msft_rolling_decline_backtest_updated.csv", index=False)

# ============================================
# 3. Evaluate metrics
# ============================================
df_eval = df[df["actual_decline"].notna()].copy()
results = []

for model_col in ["prob_decline_rf", "prob_decline_et"]:
    for horizon in df_eval["horizon"].unique():
        for threshold in sorted(df_eval["threshold"].unique()):
            sub = df_eval[
                (df_eval["horizon"] == horizon) &
                (df_eval["threshold"] == threshold)
            ].copy()

            if len(sub) == 0:
                continue

            y_true = sub["actual_decline"]
            y_prob = sub[model_col]
            y_pred = (y_prob > 0.5).astype(int)

            acc = accuracy_score(y_true, y_pred)
            brier = brier_score_loss(y_true, y_prob)
            avg_prob = y_prob.mean()
            decline_rate = y_true.mean()

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            calib_true, calib_pred = calibration_curve(y_true, y_prob, n_bins=10)

            results.append({
                "model": model_col,
                "horizon": horizon,
                "threshold": threshold,
                "accuracy": acc,
                "brier_score": brier,
                "avg_pred_prob": avg_prob,
                "actual_decline_rate": decline_rate,
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "roc_auc": roc_auc,
                "fpr": fpr, "tpr": tpr,
                "calib_true": calib_true,
                "calib_pred": calib_pred,
                "n": len(sub)
            })

# ============================================
# 4. Summary table
# ============================================
df_results = pd.DataFrame([{
    k: v for k, v in r.items()
    if k not in ["fpr", "tpr", "calib_true", "calib_pred"]
} for r in results])

print(df_results)

# ============================================
# 5. Side-by-side bar charts (corrected)
# ============================================
pivot_acc = df_results.pivot_table(
    index=["horizon", "threshold"],
    columns="model",
    values="accuracy"
).reset_index()

pivot_brier = df_results.pivot_table(
    index=["horizon", "threshold"],
    columns="model",
    values="brier_score"
).reset_index()

labels = pivot_acc.apply(lambda r: f"{r['horizon']} {r['threshold']}", axis=1)
x = np.arange(len(labels))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].bar(x - width/2, pivot_acc["prob_decline_rf"], width, label="RF")
axes[0].bar(x + width/2, pivot_acc["prob_decline_et"], width, label="ET")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=45)
axes[0].set_title("Accuracy")
axes[0].legend()

# Brier Score
axes[1].bar(x - width/2, pivot_brier["prob_decline_rf"], width, label="RF")
axes[1].bar(x + width/2, pivot_brier["prob_decline_et"], width, label="ET")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=45)
axes[1].set_title("Brier Score")
axes[1].legend()

plt.tight_layout()
plt.show()

