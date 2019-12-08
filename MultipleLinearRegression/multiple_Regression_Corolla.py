
#Assignment
#MultipleLinearRegression-Corolla

#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('ToyotaCorolla.csv')
dataset.drop(['Id','Model',''],axis=1,inplace=True)
new_dataset=dataset.loc[:,['Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]
new_dataset
X = new_dataset.iloc[:,1:].values
y = new_dataset.iloc[:,:1].values

new_dataset.info()
new_dataset.corr()
new_dataset.head()
new_dataset.shape

#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

#predicting
y_pred = reg.predict(X_test)
