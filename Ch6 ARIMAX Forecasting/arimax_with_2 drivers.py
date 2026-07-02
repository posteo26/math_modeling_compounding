# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 12:19:23 2025

"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load your CSV
df = pd.read_csv("GE Prices data.csv")

# Define target series (stock price)
y = df['Price']

# Define exogenous variables (Revenue + EPS)
X = df[['Revenue', 'Earnings']]

# Fit ARIMAX model
model = SARIMAX(y, exog=X, order=(1,1,1))  # (p,d,q) can be tuned
fit = model.fit(disp=False)

# Forecast next 4 points
forecast = fit.forecast(steps=4, exog=[[13.0,1.85],[13.5,1.95],[14.0,2.00],[14.5,2.05]])  # example future exog

# Plot historical + forecast
plt.figure(figsize=(12,5))
plt.plot(df.index, y, label='Historical Price')
plt.plot(range(len(y), len(y)+4), forecast, marker='x', linestyle='--', color='red', label='Forecast (ARIMAX)')
plt.title("GE Aerospace Stock Price Forecast (ARIMAX)")
plt.xlabel("Quarter")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.grid(True)
plt.show()