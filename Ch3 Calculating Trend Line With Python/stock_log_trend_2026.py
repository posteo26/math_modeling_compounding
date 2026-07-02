# -*- coding: utf-8 -*-
"""
Trend Line for Stock Data on Log Scale
"""

import matplotlib.pyplot as plt
import csv
import math
import pandas as pd
import yfinance as yf

import statsmodels.api as sm  
import numpy as np
import time

import os.path


folder_path = "C:\\Users\\Myname\\log_trend\\stock_data_files\\"

#download stock prices, date format ex: "2021-02-01"
def download_stock_prices(ticker, start, end):
    print("Downloading prices for:", ticker)

    df = yf.download(ticker, start=start, end=end)

    # Handle MultiIndex columns (common in yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Drop rows with all NaNs (yfinance sometimes adds these)
    df = df.dropna(how="all")

    # Enforce date filtering (yfinance sometimes returns earlier rows)
    df = df.loc[df.index >= pd.to_datetime(start)]

    # Reset index if needed (optional)
    # df = df.reset_index()

    # Save cleaned data
    out_path = folder_path + f'\\_{ticker}_.csv'
    df.to_csv(out_path)

    time.sleep(4)
    return out_path


# calc value as y = coef0 + coef1 * t
def calculate_value_on_trend_line(params, t):
    
    y = params["const"] + params[0] * t
     
    return y

# calculate estimate trend return        
def calculate_return(params):
    
    return_log = params[0] * 252
    return_actual = math.e ** return_log
    
    return [return_log, return_actual ]

# calculate return and plot data and trendline
# index looks like [0,1,2,3 ......N] like row
# after pd.Series x looks like
# 0   0
# 1   1
# 2   2
# 3   3
# .......
def analyze_data( index, stock_adj_closed_index, data, ticker ):

    x = pd.Series(index)

    y = []
    date_index = []

    for dt in data:
        y.append(dt[1])
        date_index.append(dt[2])
            
    x = sm.add_constant(x)

    model = sm.OLS(y,x)
   
    model_result = model.fit()
    params = model_result.params
    trend = []
    
    for i in range(len(x)):
        
        val = calculate_value_on_trend_line(params, i)
        
        trend.append(val)

    print(model_result.summary())
    
    plt.figure(str(5) + ticker)
    # Convert date strings to datetime
    date_index = pd.to_datetime(date_index)
    
    plt.plot(date_index, y)
    plt.plot(date_index, trend)
    plt.title(ticker + "- Prices at Log Scale and Trend Line")
    
    # Add Y‑axis label
    plt.ylabel("ln(Price)")
    
    # Automatic date ticks
    import matplotlib.dates as mdates
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    

    #this function returns  [return_log, return_actual ]
    myres = calculate_return(params)
    stock_returns_all_times[ticker] = myres[1]
    
    annual_growth_factor = myres[1]
    
    # All-time CAGR based on first and last price
    years = len(data) / 252
    start_price = math.exp(data[0][1])
    end_price = math.exp(data[-1][1])
    cagr_all = (end_price / start_price) ** (1 / years) - 1
    
    # --- 3‑year CAGR ---
    period_3y = 252 * 3
    if len(data) > period_3y:
        start_price_3y = math.exp(data[-period_3y][1])
        end_price_3y = math.exp(data[-1][1])
        cagr_3y = (end_price_3y / start_price_3y) ** (1 / 3) - 1
    else:
        cagr_3y = float('nan')  # not enough data

    # Store all metrics
    stock_returns_all_times[ticker] = {
        "growth_factor": annual_growth_factor,
        "cagr_all": cagr_all,
        "cagr_3y": cagr_3y
    }
    
    returns = []
    ret_dates = []
    
    period = 252
    for i in range(period, len(data)):
        returns.append(math.e ** (data[i][1] - data[i - period][1]))
        ret_dates.append(data[i][2])  # assuming column 2 holds the date string
    
    # Convert to datetime for proper plotting
    ret_dates = pd.to_datetime(ret_dates)
    
    plt.figure(str(6) + ticker)
    plt.plot(ret_dates, returns, label="Rolling Annual Returns")
    plt.axhline(y=1, color='r', linestyle='--', label="0% return")
    plt.axhline(y=2, color='r', linestyle='--', label="100% return")
    plt.title(ticker + " - Returns Rolling Annual")
    
    # Format x-axis with readable dates
    import matplotlib.dates as mdates
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    # Add legend
    plt.legend()

tickers = ["MSFT", "NVDA", "AVGO", "AMD", "GOOG", "VRT", "AMZN", "CAH", "IONQ"]
stock_returns_all_times = {}

for ticker in tickers:

    file = folder_path + "_" + ticker + "_" + ".csv"
    if (os.path.isfile(file) == False):
        download_stock_prices(ticker, "1980-01-01", "2028-09-01")
    
    stock_adj_closed = []
    index = []
    data = []
    stock_adj_closed_index = 5 
    with open(file, 'r') as file:
        csvreader = csv.reader(file)
        headers = next(csvreader)
        
        # Normalize headers for comparison
        normalized_headers = [h.strip().lower() for h in headers]
        
        # Define possible variations for "Adj Close"
        adj_close_variants = ["adj close", "adjusted close", "adjclose", "adjusted_close"]
        
        stock_adj_closed_index = None
        
        # Try to find "Adj Close" first
        for variant in adj_close_variants:
            if variant in normalized_headers:
                stock_adj_closed_index = normalized_headers.index(variant)
                print(f"Found '{variant}' column at index {stock_adj_closed_index}")
                break
        
        # If not found, fall back to "Close"
        if stock_adj_closed_index is None:
            if "close" in normalized_headers:
                stock_adj_closed_index = normalized_headers.index("close")
                print(f"'Adj Close' not found. Using 'Close' column at index {stock_adj_closed_index}")
            else:
                print("⚠️ Warning: Neither 'Adj Close' nor 'Close' column found. "
                      "Downloaded data may be missing required price columns.")
        
        # Optional: handle missing case gracefully
        if stock_adj_closed_index is None:
            raise ValueError("Missing 'Adj Close' or 'Close' column in CSV header.")

        count = 0
        for row in csvreader:
            stock_adj_closed.append(math.log(float(row[stock_adj_closed_index])))
            index.append(count)
            data.append([count, math.log(float(row[stock_adj_closed_index])), row[0]])
            count = count + 1
        
        analyze_data (index, stock_adj_closed_index, data, ticker)

#Optional        
#print (stock_returns_all_times)
  
print("-" * 90)

    
# Sort by growth factor descending
sorted_returns = sorted(
    stock_returns_all_times.items(),
    key=lambda x: x[1]["growth_factor"],
    reverse=True
)

print(f"{'Ticker':<10} {'Growth Factor':>15} {'Annual Return %':>18} "
      f"{'CAGR (All‑Time)':>18} {'CAGR (3 Years)':>18}")
print("-" * 90)

for ticker, metrics in sorted_returns:
    g = metrics["growth_factor"]
    pct = (g - 1) * 100
    cagr_all = metrics["cagr_all"] * 100
    cagr_3y = metrics["cagr_3y"] * 100 if not math.isnan(metrics["cagr_3y"]) else None

    print(f"{ticker:<10} {g:>15.4f} {pct:>17.2f}% "
          f"{cagr_all:>17.2f}% "
          f"{(f'{cagr_3y:>17.2f}%' if cagr_3y is not None else '   N/A'):>18}")