#!/usr/bin/env python
# coding: utf-8

# ## Import Library

# In[1]:


import requests
import pandas as pd
import scipy
import numpy as np
import sys


# ## Data Import

# In[ ]:


Train = pd.read_csv("https://storage.googleapis.com/kubric-hiring/linreg_train.csv")
train = Train.T
train = train.reset_index()
train.columns =['Area', 'Prices']
train.drop(0,inplace=True)
X = train['Area'].apply(lambda x: float(x))
Y = train['Prices'].apply(lambda y: float(y))


# ## Fit data

# In[52]:


x = (X - X.mean()) / X.std()
x = np.c_[np.ones(x.shape[0]), x] 

alpha = 0.01 #Step size
iterations = 2000 #No. of iterations
m = Y.size #No. of data points
np.random.seed(123) #Set the seed
theta = np.random.rand(2) #Pick some random values to start with


#GRADIENT DESCENT
def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = []
    past_thetas = [theta]
    for i in range(iterations):
        prediction = np.dot(x, theta)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        past_costs.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        past_thetas.append(theta)
        
    return past_thetas, past_costs

#Pass the relevant variables to the function and get the new values back...
past_thetas, past_costs = gradient_descent(x, Y, theta, iterations, alpha)
theta = past_thetas[-1]

#Print the results...
print(theta)


# ## Predict

# In[48]:


def predict_price(area):
    Intercept = 1863.71 
    slope = 384.05
    w = [Intercept, slope]
    x1 = (area - area.mean()) / area.std()
    x1 = np.c_[np.ones(x1.shape[0]), x1] 
    pred_price = x1.dot(w)
    return pred_price


# In[ ]:


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")

