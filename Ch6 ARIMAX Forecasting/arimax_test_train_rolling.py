# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 13:42:47 2025

"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Load dataset
df = pd.read_csv("GE Prices data.csv")
df = df.sort_values("Year").reset_index(drop=True)

# --- Helper function ---
def compute_metrics(actual, forecast):
    mape = mean_absolute_percentage_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    return mape, rmse

def run_models(train, test):
    y_train, y_test = train['Price'], test['Price']
    X_train_rev, X_test_rev = train[['Revenue']], test[['Revenue']]
    X_train_rev_eps, X_test_rev_eps = train[['Revenue','Earnings']], test[['Revenue','Earnings']]

    # ARIMA baseline
    fit_arima = SARIMAX(y_train, order=(1,1,1)).fit(disp=False)
    forecast_arima = fit_arima.forecast(len(y_test))

    # ARIMAX Revenue-only
    fit_rev = SARIMAX(y_train, exog=X_train_rev, order=(1,1,1)).fit(disp=False)
    forecast_rev = fit_rev.forecast(len(y_test), exog=X_test_rev)

    # ARIMAX Revenue+EPS
    fit_rev_eps = SARIMAX(y_train, exog=X_train_rev_eps, order=(1,1,1)).fit(disp=False)
    forecast_rev_eps = fit_rev_eps.forecast(len(y_test), exog=X_test_rev_eps)

    # Holt-Winters
    fit_hw = ExponentialSmoothing(y_train, trend='add', seasonal=None).fit()
    forecast_hw = fit_hw.forecast(len(y_test))

    # Collect metrics
    metrics = {
        "ARIMA (Price only)": compute_metrics(y_test, forecast_arima),
        "ARIMAX (Revenue-only)": compute_metrics(y_test, forecast_rev),
        "ARIMAX (Revenue+EPS)": compute_metrics(y_test, forecast_rev_eps),
        "Holt-Winters (Trend)": compute_metrics(y_test, forecast_hw)
    }
    return metrics

# --- Split 1: Train 2022–2023, Test 2024 ---
train1 = df[df['Year'] <= 2023]
test1 = df[df['Year'] == 2024]
metrics1 = run_models(train1, test1)

print("Validation Split 1 (Train 2022–2023, Test 2024):")
print("================================================")
for model, (mape, rmse) in metrics1.items():
    print(f"{model:25s} | MAPE: {mape:.4f} | RMSE: {rmse:.2f}")

# --- Split 2: Train 2022–2024, Test 2025 ---
train2 = df[df['Year'] <= 2024]
test2 = df[df['Year'] == 2025]
metrics2 = run_models(train2, test2)

print("\nValidation Split 2 (Train 2022–2024, Test 2025):")
print("================================================")
for model, (mape, rmse) in metrics2.items():
    print(f"{model:25s} | MAPE: {mape:.4f} | RMSE: {rmse:.2f}")