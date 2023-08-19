import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from scipy import stats
from math import sqrt

#----------------------------------------Read Data in----------------------------------------
df = pd.read_csv("Processed.csv")
df = df.loc[df["City"] == 'Amaravati']
#df = df.loc[df["State"] == 'Delhi']
print(df)
#-----------------------------------------Regression-----------------------------------------

labelencode = LabelEncoder()
dfencode = df[df.columns[10:21]]
temp = df['Time']
temp1 = df[df.columns[3:4]]
temp2 = df[df.columns[4:9]]
features = dfencode.join(temp)
features = features.apply(labelencode.fit_transform)
features = features.join(temp1)
features = features.join(temp2)
PM25 = df['PM2.5']

X_train, X_test, y_train, y_test = train_test_split(features, PM25, test_size=0.3)

LR = LinearRegression(n_jobs=12000)
LR.fit(X_train, y_train)
pred = LR.predict(X_test)
score = r2_score(y_test, pred)
print(features)
print('R2: ',score)
print('MSE: ', mean_squared_error(y_test, pred))
print('RMSE: ', sqrt(mean_squared_error(y_test, pred)))
print('MAE: ', mean_absolute_error(y_test, pred))
print(LR.intercept_)
print(LR.coef_)