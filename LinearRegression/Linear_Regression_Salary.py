# -*- coding: utf-8 -*-
#Assignment
#LinearRegression-Salary

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#check the dimension of the table
print('The dimensions of the table is:',dataset.shape)

dataset.head()
dataset.info()
dataset.describe()

#Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=101)

#Fitting Linear regression model into training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


#Predicting the test results
y_pred = regressor.predict(X_test)

plt.scatter(y_pred,(y_pred-dataset.YearsExperience),c="r")
plt.hlines(y=0,xmin=0,xmax=300) 
# checking normal distribution for residual
plt.hist(y_pred-dataset.YearsExperience)

#Visulaisation(Training Set)
plt.scatter(X_train,y_train,color='Red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#Visualisation(Test Set)
plt.scatter(X_test,y_test,color='Red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('Salary Vs Experience(Test Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()



# =============================================================================
# Histograms are very good visualization technique to check the distribution 
# of numerical data.
# =============================================================================

