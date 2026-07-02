# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 09:54:16 2025

"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('GE Prices data.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Ensure stationarity (optional: use differencing)
df['Price_diff'] = df['Price'].diff()
df['Earnings_diff'] = df['Earnings'].diff()
df.dropna(inplace=True)

# Define endogenous and exogenous variables
endog = df['Price_diff']
exog = df[['Earnings_diff']]

# Fit ARIMAX model
model = sm.tsa.SARIMAX(endog, exog=exog, order=(1,0,1))
results = model.fit()

# Forecast
forecast = results.predict(start=len(df)-30, end=len(df)-1, exog=exog[-30:])

# Plot
plt.plot(df.index[-30:], endog[-30:], label='Actual')
plt.plot(df.index[-30:], forecast, label='Forecast')
plt.legend()
plt.title('ARIMAX Forecast')
plt.show()