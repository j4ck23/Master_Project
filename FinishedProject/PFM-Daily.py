from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

df_new = pd.DataFrame(columns = ['ds', 'y'])#Define data frame for required prophet format
data = []
ds = []

#----------------------------------------Read Data in----------------------------------------
df = pd.read_csv("Processed.csv",usecols=[2])#Change column number to chnage pollutant being read in
df1 = pd.read_csv("Processed.csv",usecols=[22])#Date column
df1 = df1.join(df)#Combine

#--------------------------------------------pre-processing-----------------------------------
#Get mean pollution level for each date
data = df1.groupby("Date")['PM2.5'].mean()
for i in range(len(df1)):
    if df1['Date'][i] not in ds:
        ds.append(df1['Date'][i])
    else:
        pass
#Sort
ds.sort()

#Add pollution level to corresponding date
for i in range(len(ds)):
    temp = {'ds': ds[i], 'y':data[i]}
    df_new = df_new.append(temp, ignore_index=True)

#df_new = df_new.sort_values('ds')

#--------------------------------------------Model--------------------------------------------
#X, y = train_test_split(df_new, test_size=0.3) #Split data test trian 30 70
#Not wosing test train slpit as the data has to be in order for time seires prediction
#Still split 70 30 %, worked out by geeting the lengths of x and y
X = df_new.iloc[:-572]
y = df_new.iloc[1333:]
print(len(y))

#Build Prophet model
model = Prophet()
model.fit(X)#Pollutant data
predict = model.predict(y) #Prediction - outputted as a df
df_cv = cross_validation(model, initial='1275 days', period='31 days', horizon = '5 days')
print(df_cv[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())#Get only accuracy for each predicition
df_p = performance_metrics(df_cv)
print(df_p)

#plot predicition
model.plot(df_cv, uncertainty=True)
plt.title(" SO2 μg/m3")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

model.plot(predict, uncertainty=True)
plt.title(" PM2.5 μg/m3")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

"""
#overall accuray
y_true = y['y'].values
y_pred = df_cv['yhat'].values
print('R2: ',r2_score(y_true, y_pred))
print('MSE: ', mean_squared_error(y_true, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_true, y_pred)))
print('MAE: ', mean_absolute_error(y_true, y_pred))
"""