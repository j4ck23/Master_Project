import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statistics import mean
import tensorflow as tf
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#------------------------------------------Pre-Processing-------------------------------------
def pre_process(data):#Create data sets for current pollution level and next pollution level
    X = []
    y = []
    for i in range(len(data)-1-1):
        row = data[i:(i+1), 0]#X data - current polluition level at given time
        X.append(row)
        y.append(data[i+1, 0])#y data - pollution level at next time
    return np.array(X), np.array(y)

#------------------------------------------Model----------------------------------------------
def Model1():
    model = Sequential()
    model.add(LSTM(16, input_shape=(1,1)))#Add LSTM layer and input shape
    model.add(Dense(8))
    model.add(Dense(1, 'sigmoid'))#Add output layer
    model.compile(loss='mean_squared_error', optimizer='adam')#Define accracy reading for each epcoh
    return model
    #model.fit(X, y, epochs=epochs, batch_size=batch, verbose=ver)#Pass data and paramters

#----------------------------------------Read Data in----------------------------------------
df = pd.read_csv("Processed.csv",usecols=[7])#Change column number to chnage pollutant being read in
df = df.sample(1000)
data = df.values #Extract values from column
data = data.astype('float32')#Conver to float
#--------------------------------------------prediction---------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))#Set a scalar
data = scaler.fit_transform(data)#Transform data from current states to numbers from 0 to 1
X, y = pre_process(data)#Pass to function

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)#Split test train 30 70


#Hyper parameter tunning - commented out to ave time and computational power
"""
param_FCNN = {'batch_size':[10,20,50],'epochs':[50, 100, 200]}

model_tunning = KerasRegressor(Model1)
TUNE_model = GridSearchCV(estimator=model_tunning, param_grid=param_FCNN, cv=10, refit='accuracy')
gs = TUNE_model.fit(X_train,y_train)

print("HyperParameter results")
print(gs.cv_results_)

print("")
print("Best HyperParameters:")
t = gs.best_estimator_
print(t)
"""

LSTM_Model = KerasRegressor(build_fn=Model1, epochs=50, batch_size=20)#build LTSM passing paraters found from hyper parater tunning
Model_train = LSTM_Model.fit(X_train, y_train)#Fit training data
pred = LSTM_Model.predict(X_test)#Predict using trained model

#Accuracy for model
print("Accracy untransformed:")
print('R2: ',r2_score(y_test, pred))
print('MSE: ', (mean_squared_error(y_test,pred)))
print('RMSE: ', np.sqrt(mean_squared_error(y_test, pred)))
print('MAE: ', mean_absolute_error(y_test, pred))

#Transform predcitions and test data back from 0 to 1 to original numbers
pred_reshape = pred.reshape(-1,1)
pred_transform = scaler.inverse_transform(pred_reshape)
y_test_reshape = y_test.reshape(-1,1)
y_test_transform = scaler.inverse_transform(y_test_reshape)

#Accuracy based on origianl numbers
print("")
print("Accracy transformed:")
print('R2: ',r2_score(y_test_transform, pred_transform))
print('MSE: ', (mean_squared_error(y_test_transform,pred_transform)))
print('RMSE: ', np.sqrt(mean_squared_error(y_test_transform, pred_transform)))
print('MAE: ', mean_absolute_error(y_test_transform, pred_transform))


#Plot accuracy
plt.figure()
red_patch = mpatches.Patch(color='red', label='Predicted Values')
blue_patch = mpatches.Patch(color='blue', label='True values')
plt.title("0-1 O3 Prediciton μg/m3")
plt.xlabel("Time")
plt.ylabel("Value")
plt.plot(y_test)
plt.plot(pred, color='r')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

plt.figure()
red_patch = mpatches.Patch(color='red', label='Predicted Values')
blue_patch = mpatches.Patch(color='blue', label='True values')
plt.title("True Values O3 Prediciton μg/m3")
plt.xlabel("Time")
plt.ylabel("Value")
plt.plot(y_test_transform)
plt.plot(pred_transform, color='r')
plt.legend(handles=[red_patch, blue_patch])
plt.show()