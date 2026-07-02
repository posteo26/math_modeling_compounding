# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 13:31:59 2025

"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Load dataset
df = pd.read_csv("GE Prices data.csv")

# Split train (2022–2024) and test (2025)
train = df[df['Year'] < 2025]
test = df[df['Year'] == 2025]

y_train = train['Price']
y_test = test['Price']

X_train_rev = train[['Revenue']]
X_test_rev = test[['Revenue']]

X_train_rev_eps = train[['Revenue','Earnings']]
X_test_rev_eps = test[['Revenue','Earnings']]

# --- Model 1: ARIMA baseline (Price only) ---
model_arima = SARIMAX(y_train, order=(1,1,1))
fit_arima = model_arima.fit(disp=False)
forecast_arima = fit_arima.forecast(steps=len(y_test))

# --- Model 2: ARIMAX with Revenue-only ---
model_rev = SARIMAX(y_train, exog=X_train_rev, order=(1,1,1))
fit_rev = model_rev.fit(disp=False)
forecast_rev = fit_rev.forecast(steps=len(y_test), exog=X_test_rev)

# --- Model 3: ARIMAX with Revenue+EPS ---
model_rev_eps = SARIMAX(y_train, exog=X_train_rev_eps, order=(1,1,1))
fit_rev_eps = model_rev_eps.fit(disp=False)
forecast_rev_eps = fit_rev_eps.forecast(steps=len(y_test), exog=X_test_rev_eps)

# --- Model 4: Holt-Winters Exponential Smoothing ---
model_hw = ExponentialSmoothing(y_train, trend='add', seasonal=None)
fit_hw = model_hw.fit()
forecast_hw = fit_hw.forecast(len(y_test))

# --- Compute error metrics ---
def compute_metrics(actual, forecast):
    mape = mean_absolute_percentage_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    return mape, rmse

metrics = {
    "ARIMA (Price only)": compute_metrics(y_test, forecast_arima),
    "ARIMAX (Revenue-only)": compute_metrics(y_test, forecast_rev),
    "ARIMAX (Revenue+EPS)": compute_metrics(y_test, forecast_rev_eps),
    "Holt-Winters (Trend)": compute_metrics(y_test, forecast_hw)
}

# --- Display comparison table ---
print("Train/Test Split (2025 Hold-Out) Model Comparison:")
print("===================================================")
for model, (mape, rmse) in metrics.items():
    print(f"{model:25s} | MAPE: {mape:.4f} | RMSE: {rmse:.2f}")
    
    
# Map numeric index → quarter labels
quarter_labels = {
    12: "Q1 2025",
    13: "Q2 2025",
    14: "Q3 2025",
    15: "Q4 2025"
}


# --- Optional: Plot forecasts vs actual ---
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(test.index, y_test, marker='o', label='Actual 2025 Price')
plt.plot(test.index, forecast_arima, marker='x', linestyle='--', label='ARIMA Forecast')
plt.plot(test.index, forecast_rev, marker='x', linestyle='--', label='ARIMAX Revenue Forecast')
plt.plot(test.index, forecast_rev_eps, marker='x', linestyle='--', label='ARIMAX Revenue+EPS Forecast')
plt.plot(test.index, forecast_hw, marker='x', linestyle='--', label='Holt-Winters Forecast')

# Replace ticks with quarter labels
plt.xticks(test.index, [quarter_labels[i] for i in test.index])

plt.title("GE Aerospace — 2025 Out-of-Sample (Test Set) Forecast Comparison")
plt.suptitle("Actual 2025 Prices vs Model Forecasts (ARIMA, ARIMAX, Holt-Winters)", fontsize=10, y=0.94)
plt.xlabel("2025 Quarter (Test Period Only)")
plt.ylabel("Price (USD)")

plt.figtext(
    0.5, -0.05,
    "Model performance metrics (MAPE, RMSE) shown below reflect accuracy on the 2025 hold-out test set.",
    ha="center", fontsize=9
)


plt.legend()
plt.grid(True)
plt.show()