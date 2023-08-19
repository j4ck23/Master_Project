from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

df_new = pd.DataFrame(columns = ['ds', 'y'])
data = []
ds = []

df = pd.read_csv("Processed.csv",usecols=[2])
df1 = pd.read_csv("Processed.csv",usecols=[22])
df1 = df1.join(df)
data = df1.groupby("Date")['PM2.5'].mean()
for i in range(len(df1)):
    if df1['Date'][i] not in ds:
        ds.append(df1['Date'][i])
    else:
        pass

ds.sort()
#df1['ds']= pd.to_datetime(df1['Date'])
#df1 = df1.drop(columns=['Date'], axis=1)
#df1 = df1.rename(columns={'PM2.5': 'y'})

for i in range(len(ds)):
    temp = {'ds': ds[i], 'y':data[i]}
    df_new = df_new.append(temp, ignore_index=True)


#df1['ds']= pd.to_datetime(df1['Date'])
#df1 = df1.drop(columns=['Date'], axis=1)

X, y = train_test_split(df_new, test_size=0.3)


model = Prophet()
model.fit(X)
t = model.predict(y)
print(t[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

model.plot(t)
plt.show()

y_true = y['y'].values
y_pred = t['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
print('MAE: %.3f' % mae)

plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
