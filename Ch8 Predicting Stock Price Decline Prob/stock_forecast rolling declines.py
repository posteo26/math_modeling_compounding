# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 13:29:09 2026
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
YEARS  = 15

DECLINE_THRESHOLDS = [-0.07, -0.10]   # -7%, -10%
HORIZONS = {"3m": 63, "6m": 126, "12m": 252}

ROLL_START_DATE = pd.Timestamp("2025-01-01")
OUT_CSV = "msft_rolling_decline_backtest.csv"

# ============================================================
# 1. DOWNLOAD RAW DATA
# ============================================================
raw = yf.download([TICKER, BENCH], period=f"{YEARS}y", auto_adjust=True)

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
# 3. TECHNICAL INDICATORS (computed on df_full)
# ============================================================
df_full["ma_200"] = df_full["price"].rolling(200).mean()
df_full["dist_ma_200"] = df_full["price"] / df_full["ma_200"] - 1

delta = df_full["price"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df_full["rsi_14"] = 100 - (100 / (1 + rs))

ema12 = df_full["price"].ewm(span=12, adjust=False).mean()
ema26 = df_full["price"].ewm(span=26, adjust=False).mean()
df_full["macd"] = ema12 - ema26
df_full["macd_signal"] = df_full["macd"].ewm(span=9, adjust=False).mean()
df_full["macd_hist"] = df_full["macd"] - df_full["macd_signal"]

if "bench" in df_full.columns:
    rel = df_full["price"] / df_full["bench"]
    df_full["rel_strength"] = rel / rel.iloc[0]

df_full["roll_1y_ret"] = df_full["price"] / df_full["price"].shift(252) - 1
df_full["roll_1y_vol"] = df_full["ret_1d"].rolling(252).std()
df_full["roll_1y_sharpe"] = df_full["roll_1y_ret"] / df_full["roll_1y_vol"]

# ============================================================
# 4. BUILD df_model (TRAINING DATA WITH FUTURE RETURNS)
# ============================================================
df_model = df_full.copy()

for name, h in HORIZONS.items():
    df_model[f"fut_ret_{name}"] = df_model["price"].shift(-h) / df_model["price"] - 1

# Drop rows with NaNs in features OR labels
feature_cols = [
    "dist_ma_200",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "rel_strength",
    "roll_1y_ret", "roll_1y_vol", "roll_1y_sharpe",
]

df_model = df_model.dropna(subset=feature_cols)


print("\nTRAINING DATA RANGE:", df_model.index.min(), "→", df_model.index.max())
print("FULL DATA RANGE:", df_full.index.min(), "→", df_full.index.max())

# ============================================================
# 5. MONTHLY PREDICTION DATES (USING df_full)
# ============================================================
monthly_steps = pd.date_range(start=ROLL_START_DATE, end=df_full.index[-1], freq="MS")

prediction_dates = []
for d in monthly_steps:
    idx = df_full.index.searchsorted(d)
    if idx < len(df_full.index):
        prediction_dates.append(df_full.index[idx])

prediction_dates = sorted(list(dict.fromkeys(prediction_dates)))

# ============================================================
# 6. ROLLING BACKTEST (Corrected for Fix A)
# ============================================================

records = []

for pred_date in prediction_dates:
    print(f"\n=== Prediction date: {pred_date.date()} ===")

    # TRAINING DATA: all rows before pred_date
    df_train = df_model[df_model.index < pred_date]

    if len(df_train) < 300:
        print("Skipping — not enough training data")
        continue

    X_train = df_train[feature_cols].values

    # LIVE FEATURES
    if pred_date not in df_full.index:
        continue

    x_live = df_full.loc[[pred_date], feature_cols].values
    live_price = df_full.loc[pred_date, "price"]

    for threshold in DECLINE_THRESHOLDS:
        for horizon_name, h in HORIZONS.items():

            label_col = f"fut_ret_{horizon_name}"

            # If this horizon's future return column doesn't exist, skip
            if label_col not in df_model.columns:
                continue

            # TRAINING LABELS (only rows with valid labels)
            df_train_h = df_train.dropna(subset=[label_col])
            if len(df_train_h) < 300:
                continue

            y_train = (df_train_h[label_col] <= threshold).astype(int).values
            X_train_h = df_train_h[feature_cols].values

            # Need at least some positive samples
            if y_train.sum() < 20:
                continue

            # Train models
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

            # Predict probabilities
            prob_rf = rf.predict_proba(x_live)[0, 1]
            prob_et = et.predict_proba(x_live)[0, 1]

            # Retrieve actual future return for this horizon
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
# 7. SAVE RESULTS
# ============================================================
df_out = pd.DataFrame(records)
df_out.sort_values(["pred_date", "threshold", "horizon"], inplace=True)
df_out.to_csv(OUT_CSV, index=False)

print(f"\nSaved to {OUT_CSV}")
print(df_out.tail())