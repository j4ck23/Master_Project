from prophet import Prophet
from prophet.diagnostics import performance_metrics, cross_validation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


#----------------------------------------Read Data in----------------------------------------
df = pd.read_csv("Processed2.csv",usecols=[8])#Change column number to chnage pollutant being read in
df1 = pd.read_csv("Processed2.csv",usecols=[2])#Date column
df1 = df1.join(df)#Combine

#--------------------------------------------pre-processing-----------------------------------
df1 = df1.rename(columns={"Datetime": "ds", "O3": "y"})#Rename to required column headings
df1 = df1.sort_values('ds')#Dates need to be in order
print(df1)
#--------------------------------------------Model--------------------------------------------
#X, y = train_test_split(df1, test_size=0.3) #Split data test trian 30 70
X = df1.iloc[:-310283]
y = df1.iloc[723993:]

#Build Prophet model
model = Prophet()
model.fit(df1)#Pollutant data
#predict = model.predict(y) #Prediction - outputted as a df
#print(predict[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())#Get only accuracy for each predicition
df5 = cross_validation(model, initial='1275 days', period='31 days', horizon = '5 days')#Cross validation predicion for 5 days to 1 year
df15 = cross_validation(model, initial='1275 days', period='31 days', horizon = '15 days')
df30 = cross_validation(model, initial='1275 days', period='31 days', horizon = '30 days')
df180 = cross_validation(model, initial='1275 days', period='31 days', horizon = '180 days')
df365 = cross_validation(model, initial='1275 days', period='31 days', horizon = '365 days')

df5_p = performance_metrics(df5)#Perfroamce of predcitions
df15_p = performance_metrics(df15)
df30_p = performance_metrics(df30)
df180_p = performance_metrics(df180)
df365_p = performance_metrics(df365)

print(df5_p.iloc[-1])#Display performace for disired day
print(df15_p.iloc[-1])
print(df30_p.iloc[-1])
print(df180_p.iloc[-1])
print(df365_p.iloc[-1])

"""
#plot predicition
model.plot(predict)
plt.title(" O3 μg/m3")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

#overall accuray
y_true = y['y'].values
y_pred = predict['yhat'].values
print('R2: ',r2_score(y_pred, y_true))
print('MSE: ', mean_squared_error(y_true, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_true, y_pred)))
print('MAE: ', mean_absolute_error(y_true, y_pred))

#plot true vs predicited
plt.figure()
plt.title("O3 Prediciton μg/m3")
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
"""