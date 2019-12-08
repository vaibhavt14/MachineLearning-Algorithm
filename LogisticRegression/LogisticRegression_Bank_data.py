# -*- coding: utf-8 -*-

#Logistic Regression
#Assignment-Bank

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('bank-full.csv',sep=';')
dataset.isnull().sum()
dataset.head()
dataset.shape
dataset.corr()
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,16].values

#Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
#X[:,[4,5,6]] = labelencoder_X.fit_transform(X[:,[4,5,6]])
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,3] = labelencoder_X.fit_transform(X[:,3])
X[:,4] = labelencoder_X.fit_transform(X[:,4])
X[:,6] = labelencoder_X.fit_transform(X[:,6])
X[:,7] = labelencoder_X.fit_transform(X[:,7])
X[:,8] = labelencoder_X.fit_transform(X[:,8])
X[:,10] = labelencoder_X.fit_transform(X[:,10])
X[:,15] = labelencoder_X.fit_transform(X[:,15])
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Logistic Regression into training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#predicting
y_pred = classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_squared_error(y_pred,y_test)
mean_absolute_error(y_pred,y_test)

