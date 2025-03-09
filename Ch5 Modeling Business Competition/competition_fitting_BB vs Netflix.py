# -*- coding: utf-8 -*-

"""
Competition Model Blockbuster vs Netflix

y1[N+1] =  a*y1[N] - b*y2[N]
y2[N+1] =  c*y2[N]

Blockbuster is y1
Netflix is y2

"""

import math
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from scipy.optimize import minimize

#observations in $K
#first row corresponds to 1999 year
observations_all = [[2701000,633], 
                [2924100,11033], 
                [2736000,26005],
                [3207200,74670],
                [3521900,123883],
                [3553800,170601],
                [3157300,217663],
                [3047800,369675],
                [2864600,419172],
                [2722500,454427],
                [2178200,590998],
                [2171200,805270]]

#observations used for building model, last 3 rows are excluded
observations = [[2701000,633], 
                [2924100,11033], 
                [2736000,26005],
                [3207200,74670],
                [3521900,123883],
                [3553800,170601],
                [3157300,217663],
                [3047800,369675],
                [2864600,419172]
]

init = observations[0]

m = len(observations)

#
# calculate values of gross profit for years starting from initial
#
def calc_estimated_values(params, n, init):
    
    y=[]
    for i in range(m):
        if i == 0:
            y.append(init)
        else:
            y.append( get_value(params, y[i-1]))

    return y

#            
#get gross profit based on prev year values and model params
#
def get_value(params, prev):
    
    a,b,c = params
    return  [a * prev[0] - b * prev[1], c * prev[1]]

#      
#get error for Blockbuster or NFLX
#
def calc_sum(observed, estimated, j):
    sum = 0
    for i in range(m):
        sum = sum + 1000 * ((observed[i][j] - estimated[i][j])*(observed[i][j] - estimated[i][j])) / (observed[i][j] ** 2)
    return math.sqrt(sum) / m
       
#
#function called from minimize to calc error
#
def func(params):

    estimated = calc_estimated_values(params, m, init)

    err_bb = calc_sum(observations, estimated, 0)
    err_nflx = calc_sum(observations, estimated, 1)
    
    err = (err_bb + err_nflx) * 0.5
    
    data_err.append([err_bb, err_nflx, err]) 
    err_arr.append(err)
    return err

#
#OPTIMIZATION
#
initial_guess = [1.1, 1.1, 1.4]

#BB, impact of Netflix to BB, NFLX
bounds = ((0.93, 1.8), (1.1,2.0), (0.4, 2.1))

err_arr = []
data_err = []
          
result = minimize(func, initial_guess, bounds = bounds, method='nelder-mead', options={'xatol': 1e-6})
print ("RESULT:")
print(result.x)

print("\n")
print (tabulate(data_err, headers=["BB", "NFLX", "Total Error"]))

#
#PREDICTION
#
data = []

a,b,k = result.x
prev = [init[0],init[1]]

#prev = init  #observations[0]
y0 = []
x = []
bb = []
nflx = []
bb.append(observations[0][0])
nflx.append(observations[0][1])

xm = []
for i in range (m + 3):
    xm.append(i)
    if i >= 1:
        for j in range (2):
            if j == 0:
                pred = a * prev[0] - b * prev[1]
                prev[j] = pred
                y0.append(pred)
                x.append(i)
            else :
                pred = k * prev[1]
                prev[j] = pred

  
        data.append([i, observations_all[i][0], observations_all[i][1],  prev[0], prev[1]]) 
        bb.append(prev[0])
        nflx.append(prev[1])

print("\n")
print (tabulate(data, headers=["N", "BB", "NFLX", "BB Predicted", "NFLX Predicted"]))

#
#CHARTS
#
x=[]
for xelem in range(m):
    x.append(xelem+1999)

xml=[]
for xelem in xm:
    xml.append(xelem+1999)

plt.ylabel('Gross Profit, $ In Mln') 
plt.plot(x, np.array(observations)[:,0], 'r-', label='Actual BB')
plt.plot(x, np.array(observations)[:,1], 'b-', label='Actual NFLX')
plt.plot(xml, bb, 'g-', label='Model BB')
plt.plot(xml, nflx, 'y-', label='Model NFLX')
plt.legend()
plt.show()

x = []
for i in range(len(err_arr)):
    x.append(i)
    
plt.plot(x, err_arr, label='Error')
plt.legend()
plt.show()



