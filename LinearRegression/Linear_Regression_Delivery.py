# -*- coding: utf-8 -*-

#Assignment
#LinearRegression-Delivery Time

#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('delivery_time.csv')
X = dataset.iloc[:,1:].values
y = dataset.iloc[:,:-1].values

dataset.describe()
dataset.info()
dataset.corr()

#fitting LinearRegression model 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)

#predict
y_pred = reg.predict(X)

#Visualisation
plt.scatter(X,y,color='red')
plt.plot(X,reg.predict(X),color='blue')
plt.xlabel('Sorting Time Vs Delivery Time')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()