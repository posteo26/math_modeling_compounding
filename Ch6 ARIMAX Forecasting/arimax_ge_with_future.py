# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 12:09:41 2025

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Example: load your CSV with GE Aerospace data
df = pd.read_csv("GE Prices data.csv")

# Ensure proper datetime index if you have dates
# For now, we'll just use sequential quarters
df['Quarter'] = pd.RangeIndex(start=1, stop=len(df)+1, step=1)

# --- Chart 1: Price Change ---
df['Price Change'] = df['Price'].diff()

plt.figure(figsize=(12,5))
plt.plot(df['Quarter'], df['Price Change'], marker='o', label='Quarterly Change')
plt.title("GE Aerospace Stock Price Change per Quarter")
plt.xlabel("Quarter")
plt.ylabel("Change (USD)")
plt.legend()
plt.grid(True)
plt.show()

# --- Chart 2: Full Price + Forecast ---
# Fit Holt-Winters model for forecasting
model = ExponentialSmoothing(df['Price'], trend='add', seasonal=None)
fit = model.fit()

# Forecast next 4 quarters
forecast = fit.forecast(4)

# Extend quarters for forecast
future_quarters = np.arange(df['Quarter'].iloc[-1]+1, df['Quarter'].iloc[-1]+5)

plt.figure(figsize=(12,5))
plt.plot(df['Quarter'], df['Price'], marker='o', label='Historical Price')
plt.plot(future_quarters, forecast, marker='x', linestyle='--', color='red', label='Forecast')
plt.title("GE Aerospace Stock Price with 4-Quarter Forecast")
plt.xlabel("Quarter")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.grid(True)
plt.show()