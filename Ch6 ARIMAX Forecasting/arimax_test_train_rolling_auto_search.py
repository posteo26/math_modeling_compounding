# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 13:49:01 2025
pip install pmdarima

"""

import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv("GE Prices data.csv")

# Target variable
y = df['Price']

# Exogenous variables (optional)
X = df[['Revenue','Earnings']]

# --- Auto-ARIMA search ---
# stepwise=True uses a heuristic search to speed things up
model = pm.auto_arima(
    y,
    exogenous=X,              # include Revenue + EPS as exogenous regressors
    start_p=0, start_q=0,
    max_p=3, max_q=3,         # search space for AR and MA terms
    d=None,                   # let auto_arima decide differencing
    seasonal=False,           # no seasonality in quarterly data
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    maxiter=1000
)

print(model.summary())

# --- Forecast next 4 quarters ---
future_exog = pd.DataFrame({
    'Revenue':[12.8,13.25,14.0,14.5],
    'Earnings':[1.80,1.90,2.00,2.05]
})

forecast = model.predict(n_periods=4, exogenous=future_exog)
print("Forecasted Prices (next 4 quarters):", forecast)

# --- In-sample fitted values for error metrics ---
fitted = model.predict_in_sample(exogenous=X)

mape = mean_absolute_percentage_error(y, fitted)
rmse = np.sqrt(mean_squared_error(y, fitted))

print(f"In-sample MAPE: {mape:.4f}, RMSE: {rmse:.2f}")