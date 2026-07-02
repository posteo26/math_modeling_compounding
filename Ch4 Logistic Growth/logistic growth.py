# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

a = 0.49 #logistic growth rate
init_value = 54179 #number of EV in 2015
c = 140000000 #curve's maximum value

#actual data, starting with 2015
data = [54179, 70466, 94626, 206365, 225741, 233330, 389410, 713145, 1077138]
datax = [0,1,2,3,4,5,6,7,8]

def logistic_recursive_func(prev, a, c):

    new_val = prev + a*(1-prev/c)*prev
    return  new_val

#calc exp growth rate
#the function is used just to confirm
#that growth rate a = 0.49 is correct
def logistic_growth(dt):
    changes = []
    for i in range(1,len(dt)):
        changes.append(dt[i]/dt[i-1])
    print (changes)    
    return np.mean(changes)

print (logistic_growth(data) - 1)

x=[0]
y=[init_value] 
   
for t in range(1,50):
    y.append(logistic_recursive_func(y[t-1], a, c))
    x.append(t)

# Plot the results
plt.title('Number of Households with EV in US')
plt.plot(x, y,  label='Estimated Data')
plt.plot(datax, data,  label='Actual Data', marker=".", markersize=7 )
plt.legend()
plt.show()
