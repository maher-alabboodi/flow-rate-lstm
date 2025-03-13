# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:55:39 2019

@author: Maher
"""

import numpy as np
import tensorflow as tf
import random as rn
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from keras.constraints import max_norm
from math import sqrt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from numpy import concatenate
#%%
def parser(x):
    return datetime.strptime(x, '%Y%m')

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], X.shape[1],1 )
    model = Sequential()
    model.add(LSTM(neurons[0], batch_input_shape=(batch_size, X.shape[1], X.shape[2]) , stateful=True,return_sequences = True))
    model.add(Dropout(0.3))
    model.add(LSTM(neurons[1], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dropout(0.3))
    #model.add(LSTM(neurons[2], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    #model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        print('Epoch:',i)
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model
    


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, len(X), 1)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]


# Update LSTM model
def update_model(model, train, batch_size, updates):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], X.shape[1],1 )
    for i in range(updates):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataset = np.insert(dataset,[0]*look_back,0)    
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY= np.array(dataY)        
    dataY = np.reshape(dataY,(dataY.shape[0],1))
    dataset = np.concatenate((dataX,dataY),axis=1)  
    return dataset


# compute RMSPE
def RMSPE(x,y):
    result=0
    for i in range(len(x)):
        result += ((x[i]-y[i])/x[i])**2
    result /= len(x)
    result = sqrt(result)
    result *= 100
    return result

# compute MAPE
def MAPE(x,y):
    result=0
    for i in range(len(x)):
        result += abs((x[i]-y[i])/x[i])
    result /= len(x)
    result *= 100
    return result

#%%
series = read_csv('OP_A006_TS.csv', header=0,parse_dates=[0],index_col=0, squeeze=True,date_parser=parser)
look_back= 20
neurons= [ 2 , 5 ]
n_epochs= 1187
updates= 10
future_forecast = 24 # fo
    

raw_values = series.values
# transform data to be stationary
diff = difference(raw_values, 1)

# create dataset x,y
dataset = diff.values
dataset = create_dataset(dataset,look_back)


# split into train and test sets
train_size = int(dataset.shape[0] * 0.8)
test_size = dataset.shape[0] - train_size
train, test = dataset[0:train_size], dataset[train_size:]


# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)


# fit the model
lstm_model = fit_lstm(train_scaled, 1, n_epochs, neurons)
# forecast the entire training dataset to build up state for forecasting
print('Forecasting Training Data')   
predictions_train = list()
for i in range(len(train_scaled)):
    # make one-step forecast
    X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(raw_values)-i)
    # store forecast
    predictions_train.append(yhat)
    expected = raw_values[ i+1 ] 
    #PercentageError = (abs(predictions-expected)/expected) *100 # The forecasting had a 20% error. or The forecasting was in error by 12.5%. OR So Forecasting was only 3% off
    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
#print("%.0f%%" % (100 * 1.0/3))
# report performance
rmse_train = sqrt(mean_squared_error(raw_values[:len(train_scaled)], predictions_train))
print('Train RMSE: %.3f' % rmse_train)
#report performance using RMSPE
rmspe_train = RMSPE(raw_values[:len(train_scaled)],predictions_train)
print('Train RMSPE: %.3f' % rmspe_train)
MAE_train = mean_absolute_error(raw_values[:len(train_scaled)], predictions_train)
print('Train MAE: %.5f' % MAE_train)
MAPE_train = MAPE(raw_values[:len(train_scaled)], predictions_train)
print('Train MAPE: %.5f' % MAPE_train)
    
# forecast the test data
print('Forecasting Testing Data')
train_copy = np.copy(train_scaled)
predictions_test = list()
Percentage_Error = []
train_raw = raw_values[0:train_size]
test_raw = raw_values[train_size:]
mytest_raw=test_raw[0:look_back+1]
new_raw_values =  np.copy(mytest_raw)
for i in range(len(test_scaled)):
    # update model
    if i > 0:
        update_model(lstm_model, train_copy, 1, updates)
    # make one-step forecast
        # make one-step forecast
    if i == 0:
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        
        
    else:
        X, y = train_copy[-1, 0:-1], train_copy[-1, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
    
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    if i == 0:
        yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    else:
        yhat = inverse_difference(predictions_test_array, yhat, 1)
    # store forecast
    predictions_test.append(yhat)
    predictions_test_array = np.array(predictions_test)
    
    
    new_raw_values = np.append(new_raw_values,yhat)
    back = look_back+2
    lastrows = new_raw_values[-back:]
    
    mydiff = difference(lastrows, 1).values
    mydiff = mydiff.reshape(1,-1)
    # transform the scale of the data
    scaler, train_scaled, mydiff_scaled = scale(train, mydiff)
    
   
    # add to training set
    train_copy = concatenate((train_copy, mydiff_scaled))
    expected = raw_values[len(train) + i + 1]
    
    PercentageError = (abs(yhat-expected)/expected) *100 
    Percentage_Error.append(PercentageError)
    
    print('Month=%d, Predicted=%f, Expected=%f, The forecasting had a %.0f %% Error' % (i+1, yhat, expected,PercentageError))

# report performance using RMSE
rmse_test = sqrt(mean_squared_error(raw_values[-len(test_scaled):], predictions_test))
print('Test RMSE: %.3f' % rmse_test)
#report performance using RMSPE
rmspe_test = RMSPE(raw_values[-len(test_scaled):],predictions_test)
print('Test RMSPE: %.3f' % rmspe_test)
MAE_test = mean_absolute_error(raw_values[-len(test_scaled):], predictions_test)
print('Test MAE: %.5f' % MAE_test)
MAPE_test = MAPE(raw_values[-len(test_scaled):], predictions_test)
print('Test MAPE: %.5f' % MAPE_test)

################################# forcasting section     
## forecast 12 months 
#print('Forecasting 12 Months ')
#train_copy2 = np.copy(train_copy)
#uodated_raw_values = np.copy(raw_values)
#new_raw_values =  np.copy(raw_values)
#predictions_future = list()
#for i in range(1,future_forecast+1):
#            
#    # update model
#    if i > 1:
#        update_model(lstm_model, train_copy2, 1, updates)
#    # make one-step forecast
#    if i == 1:
#        
#        X, y = test_scaled[-1, 0:-1], test_scaled[-1, -1]
#        
#        yhat = forecast_lstm(lstm_model, 1, X)
#    else:
#        
#        X, y = train_copy2[-1, 0:-1], train_copy2[-1, -1]
#        
#        yhat = forecast_lstm(lstm_model, 1, X)
#    yhat_pred = yhat  
#    # invert scaling
#    yhat = invert_scale(scaler, X, yhat)
#    uodated_raw_values = np.append(uodated_raw_values,yhat)
#    # invert differencing
#    if i == 1:
#        yhat = inverse_difference(raw_values, yhat, 1)
#    else:
#        yhat = inverse_difference(predictions_future_array, yhat, 1)
#        
#    # store forecast
#    predictions_future.append(yhat)
#    predictions_future_array = np.array(predictions_future)
#    
#    new_raw_values = np.append(new_raw_values,yhat)
#    lastrows = new_raw_values[-look_back-2:]
#    
#    mydiff = difference(lastrows, 1).values
#    mydiff = mydiff.reshape(1,-1)
#    # transform the scale of the data
#    scaler, train_scaled, mydiff_scaled = scale(train, mydiff)
#    
#    
#    
#    yhat_pred = yhat_pred.reshape(1,1)
#    X = X.reshape(1,look_back)
#    #stack = np.concatenate((X,yhat_pred),axis=0)
#    stack = np.hstack((X,yhat_pred))
#    #stack = stack.reshape(1,-1)
#    # add to training set
#    train_copy2 = concatenate((train_copy2, mydiff_scaled))
#    #expected = raw_values[len(train) + i + 1]
#    #print('Month=%d, Predicted=%f' % (i, yhat))
#    print('Forecasting Month=%d' % (i))
    
predictions = np.concatenate((predictions_train,predictions_test),axis=0)
#predictions_future = list(np.array(predictions_future).flat)
#predictions = np.concatenate((predictions,predictions_future),axis=0)
# line plot of observed vs predicted
fig, ax = plt.subplots(1,figsize=(12,6))
ax.plot(raw_values, label='Actual', color='blue')
ax.plot(predictions, label='Predictions', color='red')
ax.axvline(x=len(train_scaled)+1,color='gold', linestyle='--',lw = 4,label="Forecasting")
ax.text(len(train_scaled)+6, 9, 'Forecasting',fontsize=14)
ax.legend(loc='upper right')
ax.set_xlabel("Time",fontsize = 16)
ax.set_ylabel('Oil production '+ r'$(10^4 m^3)$',fontsize = 16)
plt.show()

#%%
plt.figure(figsize=(15,7))
plt.plot(predictions, color='red',lw = 4,label="Predictions by LSTM ")
plt.plot(raw_values , color='green',marker=".",markersize=12, label="Actual Field Data")  #plot baseline
plt.axvline(x=len(train_scaled)+1,color='gold', linestyle='--',lw = 4,label="Forecasting")
plt.text(len(train_scaled)+6, 9, 'Forecasting',fontsize=14)
plt.legend(fontsize=30)
plt.title("Actual vs. Prediction of Oil Field Data , Cascading Approach ",fontsize=20,pad=20)
plt.xlabel("Date ",fontsize=12)
plt.ylabel("Oil production '+ r'$(10^4 m^3)$",fontsize=12)
plt.legend()
plt.savefig('Oil production_Smart_LSTM.png', dpi=300)
plt.show()

     
#%%
def run():

    #load dataset
    series = read_csv('oil_production.csv', header=0,parse_dates=[0],index_col=0, squeeze=True,date_parser=parser)
    look_back= 4
    neurons= [ 2 , 5 ]
    n_epochs=5 #1187
    updates= 1
    future_forecast = 12 # forecast 12 months 
    
    

    experiment(series, updates,look_back,neurons,n_epochs,future_forecast)


run()

#%%
