# -*- coding: utf-8 -*-

#Calories Consumed
#Assignment-LinearRegression

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#importing dataset
dataset = pd.read_csv('calories_consumed.csv')
X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

dataset.info()
dataset.describe()
dataset.corr()
dataset.head()
dataset.shape
dataset.skew()
#Fitting Linear Regression model into training dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)


#import statsmodels.api as sm
#X = sm.add_constant(X)
#est = sm.OLS(y, X)
#est2 = est.fit()
#print(est2.summary())

#Predict
y_pred = regressor.predict(X)

#Visualisation
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.xlabel('Churn_out_rate Vs Salary_hike')
plt.xlabel('Salary_hike')
plt.ylabel('Churn_out_rate')
plt.show()

