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

    print ("Downloading prices for:")
    print (ticker)

    df =yf.download(ticker,  start , end )

    df.to_csv(folder_path + '\\_'+ticker+'_.csv')
    time.sleep(4) # Sleep for 4 seconds 
    
    return folder_path + '\\_'+ticker+'_.csv'

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
    plt.plot(date_index, y)  # Plot the chart
    plt.plot(date_index, trend)  # Plot the chart
    plt.title(ticker + "- Prices at Log Scale and Trend Line")
    plt.xticks(np.arange(0, len(date_index) + 1, 4000))
    

    #this function returns  [return_log, return_actual ]
    myres = calculate_return(params)
    stock_returns_all_times[ticker] = myres[1]

    returns = []
    ret = []

    period = 252
    for i in range (period, len(data)):
        
        returns.append (math.e ** (data[i][1] - data[i-period][1]))
        ret.append(i-period)
 
    plt.figure(str(6) + ticker)
    plt.plot(ret, returns)  # Plot the chart
    plt.axhline(y=1, color='r', linestyle='--')
    plt.axhline(y=2, color='r', linestyle='--')
    plt.title(ticker + "- Returns Rolling Annual")

# ALL TICKERS
tickers = ["T", "RGTI"]
stock_returns_all_times = {}

for ticker in tickers:
    
    #if file does not exist download or just give message
    file = folder_path + "_" + ticker + "_" + ".csv"
    
    if (os.path.isfile(file) == False):
        download_stock_prices(ticker, "1980-01-01", "2028-09-01")
    
    stock_adj_closed = []
    index = []
    data = []
    stock_adj_closed_index = 5 
    with open(file, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)

        count = 0
        for row in csvreader:
            stock_adj_closed.append(math.log(float(row[stock_adj_closed_index])))
            index.append(count)
            data.append([count, math.log(float(row[stock_adj_closed_index])), row[0]])
            count = count + 1
        
        #index is x labels simply like 0,1,2.... coming from count   
        #data is array count and log are columns
        analyze_data (index, stock_adj_closed_index, data, ticker)
        
print ("OUTPUT")        
print (stock_returns_all_times)