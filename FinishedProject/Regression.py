#imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from statistics import mean
import matplotlib.pyplot as plt
import statsmodels.api as sma

#----------------------------------------Read Data in----------------------------------------
df = pd.read_csv("Processed.csv")
#df = df.loc[df["City"] == 'Amaravati']
#df = df.loc[df["State"] == 'Delhi']
#-----------------------------------------Regression-----------------------------------------

#Label Encoder
labelencode = LabelEncoder()
dfencode = df[df.columns[10:21]]#Change nominal values to numeric

#Select numeric features
temp = df['Time']
temp1 = df[df.columns[2:7]]
temp2 = df[df.columns[8:9]]

#combine features
features = dfencode.join(temp)
features = features.apply(labelencode.fit_transform)
features = features.join(temp1)
features = features.join(temp2)

#Selcet value to be predicited
PM25 = df['O3']

#£0 70 test train split
X_train, X_test, y_train, y_test = train_test_split(features, PM25, test_size=0.3)

#Hyper paramter tuning
#Commented out to save on computional power and time
"""
param = {'n_jobs': [1, 100, 1000, 10000]} #set param to test
Tune = GridSearchCV(estimator=LinearRegression(), param_grid=param, cv=10, scoring='r2') #establish classifier and params
Tune.fit(X_train, y_train)#fits data to run on
print("Best Parameters:")
print(Tune.best_estimator_)
"""

#Linear model
LR = LinearRegression(n_jobs=1)#Model
LR.fit(X_train, y_train)#Fit training data
pred = LR.predict(X_test)#Predict

#Accuray scores
print('R2: ',r2_score(y_test, pred))
print('MSE: ', mean_squared_error(y_test, pred))
print('RMSE: ', sqrt(mean_squared_error(y_test, pred)))
print('MAE: ', mean_absolute_error(y_test, pred))
print('Intercept: ', LR.intercept_)
print('Coeffencent: ', LR.coef_)

#Signifcany using Stats model
significany = sma.add_constant(X_test)
significany_test = sma.OLS(y_test, significany)
results = significany_test.fit()
print(results.summary())

#Plot accracy
plt.figure()
plt.title("O3 Prediciton μg/m3")
plt.scatter(y_test, pred)
plt.xlabel("True Value")
plt.ylabel("Predicition")
plt.show()