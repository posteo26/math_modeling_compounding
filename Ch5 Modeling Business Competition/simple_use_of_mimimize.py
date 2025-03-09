# -*- coding: utf-8 -*-
import math
import numpy as np
from scipy.optimize import minimize

observ = [1,4,7,12,20,32,40,59,74,90]
initial_guess = np.array([1.0]) 
bounds = ((0.1, 4.1),)
init = observ[0]
m = len(observ)

#calculate estimated values per model: 
#y[N+1] = y[N] + a*N
#
def calc_estimated_values(params, n, init):
    y=[]
    for i in range(m):
        if i == 0:
            y.append(init)
        else:
            y.append(y[i-1] + params[0] * i)
    return y
       
#function called from minimize to calc error
#
def func(params):

    estimated = calc_estimated_values(params, m, init)
    
    sum = 0
    for i in range(m):
        sum = sum + ((observ[i] - estimated[i])**2) / (observ[i] ** 2)
    err = math.sqrt(sum) / m

    return err
          
result = minimize(func, initial_guess, bounds = bounds, method='nelder-mead')
print ("RESULT:")
print (result)
