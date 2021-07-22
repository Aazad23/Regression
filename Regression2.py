#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import scipy
import numpy as np
import sys

# add branched
TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    #total number of values
    n = len(X)
    # using the formula to calculate the b1 and b0
    numerator = 0
    denominator = 0
    for i in range(n):
        numerator += (X[i] - x_mean) * (Y[i] - y_mean)
        denominator += (X[i] - x_mean) ** 2

    b1 = numerator / denominator
    b0 = y_mean - (b1 * x_mean)
    #printing the coefficient
    print(b1, b0)
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    ...


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


# In[26]:


import requests
import pandas as pd
import scipy
import numpy as np
import sys


# In[4]:


Trai = pd.read_csv("https://storage.googleapis.com/kubric-hiring/linreg_train.csv")


# In[17]:


Trai


# In[11]:


train = train.reset_index()


# In[21]:


train.columns =['Area', 'Prices']


# In[43]:


X = train['Area'].apply(lambda x: int(float(x)))
Y = train['Prices'].apply(lambda y: int(float(y)))


# In[50]:


def mean(values):
    return sum(values) / float(len(values))
 
# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar
 
# Calculate the variance of a list of numbers
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])
 
# Calculate coefficients
def coefficients(x,y):
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]


# In[47]:


def fit(X,Y):
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    #total number of values
    n = len(X)
    # using the formula to calculate the b1 and b0
    numerator = 0
    denominator = 0
    for i in range(n):
        numerator += (X[i] - x_mean) * (Y[i] - y_mean)
        denominator += (X[i] - x_mean) ** 2

    b1 = numerator / denominator
    b0 = y_mean - (b1 * x_mean)
    #printing the coefficient
    return b1, b0


# In[52]:


print((coefficients(X,Y))


# In[53]:


train.describe()


# In[55]:


train.isnull().sum()


# In[ ]:




