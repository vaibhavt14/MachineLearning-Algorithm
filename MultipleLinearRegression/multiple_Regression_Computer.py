# -*- coding: utf-8 -*-

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Computer_Data.csv')
dataset = dataset.drop([1441,1701],axis=0)
dataset = dataset.iloc[:, 1:]
X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

dataset.head()
dataset.info()
dataset.describe()
c = dataset.corr()
dataset.columns = ['Price','Speed','HD','RAM','Screen','CD','Multi','Premium','Ads','Trend']
# Taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values ='NaN', strategy='mean',axis=0)
#imputer = imputer.fit(X[:,1:3])
#X[:,1:3] = imputer.transform(X[:,1:3])

# Encoding categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
#X[:,[4,5,6]] = labelencoder_X.fit_transform(X[:,[4,5,6]])
X[:,4] = labelencoder_X.fit_transform(X[:,4])
X[:,5] = labelencoder_X.fit_transform(X[:,5])
X[:,6] = labelencoder_X.fit_transform(X[:,6])
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

# Splitting dataset into trainig set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)



y_pred = lr.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error
mean_squared_error(y_pred,y_test)
mean_absolute_error(y_pred,y_test)





