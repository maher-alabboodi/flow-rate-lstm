# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:09:55 2019

@author: Maher

This is Multi input LSTM Model 

"""
import math
from datetime import datetime
import numpy as np
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os.path
#%%
path = r'D:\DevelopmentCodes\ImportData'
oil_production = pd.read_csv(os.path.join(path, "Oil.csv"))
gas_production = pd.read_csv(os.path.join(path, "Gas.csv"))
water_production = pd.read_csv(os.path.join(path, "Water.csv"))
whp = pd.read_csv(os.path.join(path,"WHP.csv"))
choke_size = pd.read_csv(os.path.join(path,"Choke.csv"))
completion = pd.read_csv(os.path.join(path,"Completion.csv"))
pressure = pd.read_csv(os.path.join(path,"Pressure.csv"))
water_satruration = pd.read_csv(os.path.join(path,"Sw.csv"))
petrophysics = pd.read_csv(os.path.join(path,"Petrophysics.csv"))
dop = pd.read_csv(os.path.join(path,"DOP.csv"))
Static_attributes_list=list(petrophysics.iloc[0:0,3:])
#%%    function to start the data from first oil production date by removing the empty columns 
def remove_invild_columns_oil(welldata):
    welldata.fillna(0,inplace=True)
    oilcolumns = list(welldata) 
    zero =[]
    for col in range (1,len(oilcolumns)):  
        if welldata.iloc[0][col]==0:
            zero.append(col)
        elif welldata.iloc[0][col]!=0:
           break 
    welldata.drop(welldata.columns[zero],axis =1, inplace= True)
    return welldata, zero
def remove_invild_columns_other(welldata2):
    
    welldata2.drop(welldata2.columns[zero],axis =1, inplace= True)   
    return welldata2
#%%   
def return_input_output_for_RNN(n_input,lookback,testsize_fraction,input_data):
    attribute = n_input-1
    test_size=int(testsize_fraction * len(input_data))
    X=[]
    y=[]
    for i in range(len(input_data)-lookback-1):
        t=[]
        for j in range(0,lookback):
            
            t.append(input_data[[(i+j)], :])
        X.append(t)
        y.append(input_data[i+ lookback,attribute]) 
        X_all, y_all= np.array(X), np.array(y)
        x_test = X_all[-test_size:]
        y_test = y_all[-test_size:]
        x_train = X_all[:len(X)-test_size]
        y_train = y_all[:len(y)-test_size]
        x_train = x_train.reshape(x_train.shape[0],lookback, n_input)  # orginal approach
        x_test = x_test.reshape(x_test.shape[0],lookback, n_input)      # orginal approach
#        x_train = x_train.reshape(x_train.shape[0], n_input,lookback)
#        x_test = x_test.reshape(x_test.shape[0], n_input,lookback)
        
    return x_train, y_train, x_test, y_test 

#%%
lookback = 20
testsize_fraction = 0.1     
#a = pressure['Well Name']
#wellNames = pressure['Well Name']
#wellNames.drop(wellNames.index[[3, 5, 6, 9, 10, 14, 15]], inplace = True)
wellNames = ['L28-120_CH4']
wellbatch = []
wellspressur = []
for w in wellNames:
    oil = oil_production.groupby(['ISI Name Concat (Oil)']).get_group(w)#
    pr = pressure.groupby(['Well Name']).get_group(w)
    gas = gas_production.groupby(['ISI Name Concat (Gas)']).get_group(w)
    choke = choke_size.groupby(['ISI Name Concat']).get_group(w)
    sw = water_satruration.groupby(['Well Name']).get_group(w)
    perm = petrophysics.groupby(['ISI Name Concat']).get_group(w)['permeability'] 
    por = petrophysics.groupby(['ISI Name Concat']).get_group(w)['porosity']  

    oil_wellData,zero=remove_invild_columns_oil(oil)
    oil_wellData = oil_wellData.iloc[:,1:].values.T
    pr_wellData=remove_invild_columns_other(pr)
    pr_wellData = pr_wellData.iloc[:,1:].values.T
    gas_wellData=remove_invild_columns_other(gas)
    gas_wellData = gas_wellData.iloc[:,1:].values.T
    sw_wellData=remove_invild_columns_other(sw)
    sw_wellData = sw_wellData.iloc[:,1:].values.T
    choke_wellData=remove_invild_columns_other(choke)
    choke_wellData = choke_wellData.iloc[:,1:].values.T
    perm_arry =  np.copy(oil_wellData);perm_arry.fill(float(perm))
    por_arry =  np.copy(oil_wellData);por_arry.fill(float(por)) 
    ##########################################################
    #input_feature = np.hstack((gas_wellData)) # 6 inputs and pressure as output
    input_feature = gas_wellData
    ##########################################################
    input_data = input_feature
    rows,n_input = input_feature.shape
    sc= MinMaxScaler(feature_range=(0,1))
    input_data[:,0:n_input] = sc.fit_transform(input_feature[:,:])
    #############################################################
x_train, y_train, x_test, y_test = return_input_output_for_RNN(n_input,lookback,testsize_fraction,input_data)
  
#%%######################################################### build LSTM model 
np.random.seed(7)
model = Sequential()
model.add(LSTM(units=180, return_sequences= True, input_shape=(x_train.shape[1],n_input)))
model.add(LSTM(units=180, return_sequences=True))
model.add(LSTM(units=180))
model.add(Dense(units=1))
model.summary()
model.compile(optimizer= 'adam', loss='mean_squared_error')
# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=200),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

batch_size_1 = len(y_train)
#batch_size_2 = x_train.shape[1] *2
batch_size_2 = 20

model.fit(x_train,  y_train, epochs=1000, batch_size = batch_size_2, callbacks=callbacks,validation_data=(x_test,y_test))
model.reset_states()                                                                                           
print ('\007')  # sound of alarm 
#%%
model = Sequential()  
model.add(LSTM((1),input_shape=(lookback,n_input),return_sequences= True))
model.add(LSTM((1),return_sequences= False))
model.summary()
#model.add(LSTM((1), return_sequences= True,  input_shape=(lookback,3),activation='sigmoid' ))
#model.add(LSTM(lookback, return_sequences= True, input_shape=(lookback,3),activation='sigmoid' ))
#model.add(LSTM(lookback, return_sequences= True, input_shape=(lookback,3),activation='sigmoid' ))
#model.add(LSTM(lookback, return_sequences= True, input_shape=(lookback,3),activation='sigmoid' ))
#model.add(LSTM(lookback, return_sequences= True, input_shape=(lookback,3),activation='sigmoid' ))
#model.add(LSTM(lookback, return_sequences= True, input_shape=(lookback,3),activation='sigmoid' ))
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,  y_train, epochs=2000, batch_size = 25,validation_data=(x_test,y_test))
#%%
predicted_value= model.predict(x_test)

plt.figure(figsize=(10,5))
plt.plot(predicted_value, color= 'red',label="Predicted")
plt.plot(y_test, color='green',label="Actual"  )
plt.title("Actual vs. Predicted Test Data")
plt.xlabel(" Date (Normalized) ")
plt.ylabel(" Pressure (Normalized)")
plt.legend()
plt.show()

#%%#################################################### cascading the test data 
preds_moving_t2 = [] 
moving_train2_window = [x_test[0,:].tolist()]          # Creating the first test window by taking the first window
moving_train2_window = np.array(moving_train2_window).reshape(1,lookback,n_input)
length = len(y_test)
length_1 = length - 1
for i in range(length):
    
    #preds_one_step = model.predict(moving_test_window)
    
    #preds_one_step = sess.run(outputs, feed_dict={X: moving_train2_window})
    
    preds_one_step= model.predict(moving_train2_window)
    
    #stack_preds_one_step = moving_train2_window[:,-1:,:]
    try: 
        stack = x_test[i+1,-1:,:]
        stack[:,n_input-1] = preds_one_step
        stack = stack.reshape(1,1,n_input)
    except:
        i==length_1
    #stack_preds_one_step[:,:,3] = preds_one_step
    preds_moving_t2.append(preds_one_step[0,0])

    preds_one_step = preds_one_step.reshape(1,1,1)

    moving_train2_window = np.concatenate((moving_train2_window[:,1:,:], stack), axis=1)
    
    

#%%########################################################## plot the cascading the test data  results 
plt.figure(figsize=(12,5))
plt.plot(preds_moving_t2, color= 'red',label="Predicted")
plt.plot(y_test, color='green',label="Actual")
plt.title("Actual vs. Predicted Test Data (Pressure , Cascading )")
plt.xlabel("Date ( Normalized)")
plt.ylabel("Pressure (Normalized)")
plt.legend()
plt.savefig('test data.png', dpi=300)
plt.show()

#%%###################################################### cascading the training data 
preds_moving_train = [] 
moving_train_window = [x_train[0,:].tolist()]          # Creating the first test window by taking the first window
moving_train_window = np.array(moving_train_window).reshape(1,lookback,n_input)
length = len(y_train)# + len(y_test)
length_1 = length - 1
 
for i in range(length):
    
    #preds_one_step = model.predict(moving_test_window)
    
    #preds_one_step = sess.run(outputs, feed_dict={X: moving_train2_window})
    
    preds_one_step= model.predict(moving_train_window)
    
    #stack_preds_one_step = moving_train2_window[:,-1:,:]
    try: 
        sstack = x_train[i+1,-1:,:]
        sstack[:,n_input-1] = preds_one_step
        sstack = sstack.reshape(1,1,n_input)
    except:
        i==length_1
    #stack_preds_one_step[:,:,3] = preds_one_step
    preds_moving_train.append(preds_one_step[0,0])

    preds_one_step = preds_one_step.reshape(1,1,1)

    moving_train_window = np.concatenate((moving_train_window[:,1:,:], sstack), axis=1)
    

#%%###########################################################  plot cascading the training data 
preds_moving_train= model.predict(x_train)
plt.figure(figsize=(12,5))
plt.plot(preds_moving_train, color= 'red',label="Predicted")
plt.plot(y_train, color='green',label="Actual")
#plt.plot(np.hstack((y_train,y_test)) , color='green',label="Actual")  #plot baseline
plt.title("Actual vs. Predicted Training Data (Pressure, Cascading )")
plt.xlabel("Date ( Normalized)")
plt.ylabel("Pressure (Normalized)")
plt.legend()
plt.savefig('training data.png', dpi=300)
plt.show()

#%%
baseline =  np.hstack((y_train,y_test))
baseline=baseline.reshape(len(baseline),1)
preds_moving_train = np.array(preds_moving_train).reshape(len(preds_moving_train),1)
preds_moving_t2 = np.array(preds_moving_t2).reshape(len(preds_moving_t2),1)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(baseline)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[:len(preds_moving_train), :] = preds_moving_train
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(baseline)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[ len(preds_moving_train):, :] = preds_moving_t2
# plot baseline and predictions
#plt.plot(scaler.inverse_transform(dataset))

plt.figure(figsize=(12,5))
plt.plot(testPredictPlot, color= 'red',label="Predicted Test Data")
plt.plot(trainPredictPlot, color='blue',label="Predicted Training Data")
plt.plot(baseline , color='green',label="Actual")  #plot baseline
plt.title("Actual vs. Predicted (Pressure, Cascading )")
plt.xlabel("Date ( Normalized)")
plt.ylabel("Pressure (Normalized)")
plt.legend()
plt.savefig('All.png', dpi=300)
plt.show()
#%%#################################### predict training data set
preds_moving = [] 
moving_train_window = [x_train[0,:].tolist()]          # Creating the first test window by taking the first window
moving_train_window = np.array(moving_train_window).reshape(1,lookback,1)

for i in range(length):   # forcase for one year ahead 
    
    #preds_one_step = model.predict(moving_test_window)
    
    preds_one_step= model.predict(moving_train_window)
    preds_moving.append(preds_one_step[0,0])

    preds_one_step = preds_one_step.reshape(1,1,1)

    moving_test_window = np.concatenate((moving_train_window[:,1:,:], preds_one_step), axis=1)
#%%############################ plot predicted training data 
#preds_moving= model.predict(x_test)   # not cascading 
plt.figure(figsize=(12,5))
plt.plot(preds_moving, color= 'red',label="Predicted")
plt.plot(y_train, color='green',label="Actual")
#plt.plot(np.hstack((y_train,y_test)) , color='green',label="Actual")  #plot baseline
plt.title("Actual vs. Predicted Test Data ( Gas Production, Not Cascading )")
plt.xlabel("Date ( Normalized)")
plt.ylabel("Gas Production (Normalized)")
plt.legend()
plt.savefig('testgdata_Gas_NotCascading.png', dpi=300)
plt.show()

#%%



















