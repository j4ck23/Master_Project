import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import *
import matplotlib.pyplot as plt

#------------------------------------------Pre-Processing-------------------------------------
def pre_process(data, look_back = 1):
    #data = data.to_numpy()
    X = []
    y = []
    for i in range(len(data)-look_back-1):
        row = data[i:(i+look_back), 0]
        X.append(row)
        y.append(data[i+look_back, 0])
    return np.array(X), np.array(y)


#----------------------------------------Read Data in----------------------------------------
df = pd.read_csv("Processed.csv",usecols=[2])
data = df.values
data = data.astype('float32')
#--------------------------------------------Model-------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
X, y = pre_process(data)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = Sequential()
model.add(LSTM(16, input_shape=(1,1)))
#model.add(Dense(8))
model.add(Dense(1, 'sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2)

pred = model.predict(X_test)
testScore = np.sqrt(mean_squared_error(y_test, pred))
print('MSE: ', (mean_squared_error(y_test,pred)))
print('RMSE: ', (testScore))

plt.figure()
plt.plot(y_test)
plt.plot(pred, color='r')
plt.show()
#model1 = load_model(model)