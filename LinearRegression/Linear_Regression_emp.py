
#Assignment 
#Linear Regression->emp_data

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('emp_data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Fitting the regression model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)


#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#X = sc_X.fit_transform(X)

#Predict 
y_pred = regressor.predict(X)

#Visalisation
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.xlabel('Churn_out_rate Vs Salary_hike')
plt.xlabel('Salary_hike')
plt.ylabel('Churn_out_rate')
plt.show()
