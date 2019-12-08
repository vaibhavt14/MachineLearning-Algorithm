# -*- coding: utf-8 -*-

#Assignment
#MultipleLinearRegression-Startups

#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4:].values

dataset.describe()
dataset.info()
dataset.corr()

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Splitting the training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#fitting the MultipleLinear regression 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)

#Predicting the test results
y_pred = reg.predict(X_test)




