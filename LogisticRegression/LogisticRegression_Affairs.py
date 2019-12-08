# -*- coding: utf-8 -*-

#LogisticRegression-Affairs

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#importing dataset
affairdata =sm.datasets.fair.load_pandas().data
affairdata['affairs'] = (affairdata.affairs > 0).astype(int)
affairdata.groupby('affairs').mean()
affairdata.groupby('rate_marriage').mean()
X = affairdata.iloc[:,:-1].values
y = affairdata.iloc[:,8:].values
affairdata.head()

affairdata.rate_marriage.hist()
plt.title('Histogram for marriage rating')
plt.xlabel('Marriage Rating')
plt.ylabel('frequency')

pd.crosstab(affairdata.rate_marriage, affairdata.affairs.astype(bool)).plot(kind='bar')
plt.title('Marriage Rating Distribution by Affair Status')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')

#Splitting taining set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


#fitting logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#predict
y_pred = classifier.predict(X_test)
print(y_pred)
#check accuracy of training set
classifier.score(X,y)
#What percentage have affair
y.mean()        

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(cm)
#metrics.classification_report(y_pred,y_test))
