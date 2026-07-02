# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 14:41:21 2026
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ============================================================
# USER SETTINGS
# ============================================================
TICKER = "MSFT"
BENCH  = "SPY"
YEARS  = 8

DECLINE_THRESHOLDS = [-0.07, -0.10]   # -7%, -10%
HORIZONS = {"3m": 63, "6m": 126, "12m": 252}

ROLL_START_DATE = pd.Timestamp("2025-01-01")
OUT_CSV = "msft_rolling_decline_backtest_fast_features.csv"

# ============================================================
# 1. DOWNLOAD RAW DATA
# ============================================================
raw = yf.download([TICKER, BENCH], period=f"{YEARS}y", auto_adjust=True)

print("Raw last date:", raw.index[-1])
print("Raw last prices:\n", raw["Close"].iloc[-1])
print(raw.tail())

# ============================================================
# 2. BUILD df_full (FULL DATASET FOR PREDICTION)
# ============================================================
if isinstance(raw.columns, pd.MultiIndex):
    px_msft = raw["Close"][TICKER].rename("price")
    px_bench = raw["Close"][BENCH].rename("bench")
else:
    px_msft = raw["Close"].rename("price")
    px_bench = None

df_full = pd.concat([px_msft], axis=1).dropna()
if px_bench is not None:
    df_full["bench"] = px_bench.reindex(df_full.index)

df_full["ret_1d"] = df_full["price"].pct_change()

# ============================================================
# 3. BASE TECHNICAL INDICATORS
# ============================================================
df_full["ma_200"] = df_full["price"].rolling(200).mean()
df_full["dist_ma_200"] = df_full["price"] / df_full["ma_200"] - 1

# Standard RSI 14
delta = df_full["price"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df_full["rsi_14"] = 100 - (100 / (1 + rs))

# Standard MACD 12/26
ema12 = df_full["price"].ewm(span=12, adjust=False).mean()
ema26 = df_full["price"].ewm(span=26, adjust=False).mean()
df_full["macd"] = ema12 - ema26
df_full["macd_signal"] = df_full["macd"].ewm(span=9, adjust=False).mean()
df_full["macd_hist"] = df_full["macd"] - df_full["macd_signal"]

# Relative strength vs SPY
if "bench" in df_full.columns:
    rel = df_full["price"] / df_full["bench"]
    df_full["rel_strength"] = rel / rel.iloc[0]

# Rolling 1‑year stats
df_full["roll_1y_ret"] = df_full["price"] / df_full["price"].shift(252) - 1
df_full["roll_1y_vol"] = df_full["ret_1d"].rolling(252).std()
df_full["roll_1y_sharpe"] = df_full["roll_1y_ret"] / df_full["roll_1y_vol"]

# ============================================================
# 4. FAST-REACTING FEATURES
# ============================================================
# Short-term slopes
df_full["slope_5d"] = df_full["price"].diff(5) / 5
df_full["slope_10d"] = df_full["price"].diff(10) / 10

# Rate of change of dist_ma_200
df_full["roc_dist_ma_200_5d"] = df_full["dist_ma_200"].diff(5)
df_full["roc_dist_ma_200_10d"] = df_full["dist_ma_200"].diff(10)

# Fast RSI (5-day)
delta_fast = df_full["price"].diff()
gain_fast = delta_fast.clip(lower=0)
loss_fast = -delta_fast.clip(upper=0)
avg_gain_fast = gain_fast.rolling(5).mean()
avg_loss_fast = loss_fast.rolling(5).mean()
rs_fast = avg_gain_fast / avg_loss_fast
df_full["rsi_5"] = 100 - (100 / (1 + rs_fast))

# Fast MACD
ema6 = df_full["price"].ewm(span=6, adjust=False).mean()
ema13 = df_full["price"].ewm(span=13, adjust=False).mean()
df_full["macd_fast"] = ema6 - ema13
df_full["macd_fast_signal"] = df_full["macd_fast"].ewm(span=4, adjust=False).mean()
df_full["macd_fast_hist"] = df_full["macd_fast"] - df_full["macd_fast_signal"]

# Gap down flag
df_full["gap_down"] = (df_full["price"].pct_change() < -0.03).astype(int)

# ============================================================
# 5. BUILD df_model (TRAINING DATA WITH FUTURE RETURNS)
# ============================================================
df_model = df_full.copy()

for name, h in HORIZONS.items():
    df_model[f"fut_ret_{name}"] = df_model["price"].shift(-h) / df_model["price"] - 1

feature_cols = [
    "dist_ma_200",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "rel_strength",
    "roll_1y_ret", "roll_1y_vol", "roll_1y_sharpe",
    "slope_5d", "slope_10d",
    "roc_dist_ma_200_5d", "roc_dist_ma_200_10d",
    "rsi_5",
    "macd_fast", "macd_fast_signal", "macd_fast_hist",
    "gap_down",
]

df_model = df_model.dropna(subset=feature_cols).copy()

print("\nTRAINING DATA RANGE:", df_model.index.min(), "→", df_model.index.max())
print("FULL DATA RANGE:", df_full.index.min(), "→", df_full.index.max())

# ============================================================
# 6. MONTHLY PREDICTION DATES (USING df_full)
# ============================================================
monthly_steps = pd.date_range(start=ROLL_START_DATE, end=df_full.index[-1], freq="MS")

prediction_dates = []
for d in monthly_steps:
    idx = df_full.index.searchsorted(d)
    if idx < len(df_full.index):
        prediction_dates.append(df_full.index[idx])

prediction_dates = sorted(list(dict.fromkeys(prediction_dates)))

print("\nPrediction dates:", prediction_dates)
print("Total:", len(prediction_dates))

# ============================================================
# 7. ROLLING BACKTEST
# ============================================================
records = []

for pred_date in prediction_dates:
    print(f"\n=== Prediction date: {pred_date.date()} ===")

    # Training data: all rows before pred_date
    df_train = df_model[df_model.index < pred_date]

    if len(df_train) < 300:
        print("Skipping — not enough training data")
        continue

    if pred_date not in df_full.index:
        continue

    x_live = df_full.loc[[pred_date], feature_cols].values
    live_price = df_full.loc[pred_date, "price"]

    for threshold in DECLINE_THRESHOLDS:
        for horizon_name, h in HORIZONS.items():
            label_col = f"fut_ret_{horizon_name}"

            # Horizon must exist in df_model
            if label_col not in df_model.columns:
                continue

            # Horizon‑specific training subset: require label for this horizon
            df_train_h = df_train.dropna(subset=[label_col])
            if len(df_train_h) < 300:
                continue

            X_train_h = df_train_h[feature_cols].values
            y_train = (df_train_h[label_col] <= threshold).astype(int).values

            # Need enough positive samples
            if y_train.sum() < 20:
                continue

            rf = RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_h, y_train)

            et = ExtraTreesClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            et.fit(X_train_h, y_train)

            prob_rf = rf.predict_proba(x_live)[0, 1]
            prob_et = et.predict_proba(x_live)[0, 1]

            # Actual future return: must come from df_model, not df_full
            if pred_date in df_model.index:
                raw_val = df_model.loc[pred_date, label_col]
                fut_ret = raw_val.iloc[0] if isinstance(raw_val, pd.Series) else raw_val
            else:
                fut_ret = np.nan

            actual_decline = int(fut_ret <= threshold) if pd.notna(fut_ret) else np.nan

            print(
                f"{horizon_name} | thr {threshold*100:.0f}% | "
                f"P_RF={prob_rf:.3f} P_ET={prob_et:.3f} | "
                f"fut_ret={fut_ret}"
            )

            records.append({
                "pred_date": pred_date,
                "horizon": horizon_name,
                "threshold": threshold,
                "price_at_pred": live_price,
                "prob_decline_rf": prob_rf,
                "prob_decline_et": prob_et,
                "actual_fut_ret": fut_ret,
                "actual_decline": actual_decline,
            })

# ============================================================
# 8. SAVE RESULTS
# ============================================================
df_out = pd.DataFrame(records)
df_out.sort_values(["pred_date", "threshold", "horizon"], inplace=True)
df_out.to_csv(OUT_CSV, index=False)

print(f"\nSaved to {OUT_CSV}")
print(df_out.tail())


import pandas as pd
import matplotlib.pyplot as plt

# Load df_full from the same feature-building logic
# If you ran the backtest in the same session, you already have df_full in memory.
# Otherwise, rebuild df_full using the same steps as in the script above.

start = "2025-10-01"
end   = "2026-02-01"

df_vis = df_full.loc[start:end].copy()

fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)

axes[0].plot(df_vis.index, df_vis["price"], label="Price", color="black")
axes[0].set_ylabel("Price")
axes[0].legend()

axes[1].plot(df_vis.index, df_vis["dist_ma_200"], label="dist_ma_200", color="blue")
axes[1].set_ylabel("dist_ma_200")
axes[1].legend()

axes[2].plot(df_vis.index, df_vis["slope_10d"], label="slope_10d", color="purple")
axes[2].set_ylabel("slope_10d")
axes[2].legend()

axes[3].plot(df_vis.index, df_vis["roc_dist_ma_200_10d"], label="roc_dist_ma_200_10d", color="orange")
axes[3].set_ylabel("roc_dist_ma_200_10d")
axes[3].legend()

axes[4].plot(df_vis.index, df_vis["rsi_5"], label="RSI 5", color="green")
axes[4].axhline(30, color="red", linestyle="--", alpha=0.4)
axes[4].axhline(70, color="red", linestyle="--", alpha=0.4)
axes[4].set_ylabel("RSI 5")
axes[4].legend()

axes[5].plot(df_vis.index, df_vis["macd_fast_hist"], label="MACD fast hist", color="brown")
axes[5].axhline(0, color="gray", linestyle="--", alpha=0.5)
axes[5].set_ylabel("MACD fast hist")
axes[5].legend()

plt.suptitle("Fast-Reacting Features Around MSFT Decline")
plt.tight_layout()
plt.show()