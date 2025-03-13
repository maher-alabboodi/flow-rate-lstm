# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:09:55 2019

@author: Maher

This is Multi input LSTM Model 

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path, files):
    data = {}
    for file in files:
        try:
            key = file.split('.')[0].lower()
            data[key] = pd.read_csv(os.path.join(path, file))
            logging.info(f"Loaded {file} successfully.")
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
    return data

def clean_data(df):
    df.fillna(0, inplace=True)
    non_zero_idx = next((i for i, v in enumerate(df.iloc[0, 1:].values) if v != 0), None)
    return df.iloc[:, non_zero_idx:], non_zero_idx

def remove_invalid_columns(df, zero_idx):
    return df.iloc[:, zero_idx:]

def prepare_rnn_data(input_data, lookback, test_size_fraction):
    n_features = input_data.shape[1]
    test_size = int(test_size_fraction * len(input_data))
    X, y = [], []
    for i in range(len(input_data) - lookback - 1):
        X.append(input_data[i:i + lookback, :])
        y.append(input_data[i + lookback, -1])
    X, y = np.array(X), np.array(y)
    return X[:-test_size], y[:-test_size], X[-test_size:], y[-test_size:]

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(180, return_sequences=True, input_shape=input_shape),
        LSTM(180, return_sequences=True),
        LSTM(180),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_results(y_test, predicted_values):
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_values, color='red', label='Predicted')
    plt.plot(y_test, color='green', label='Actual')
    plt.title('Actual vs. Predicted Test Data')
    plt.xlabel('Date (Normalized)')
    plt.ylabel('Pressure (Normalized)')
    plt.legend()
    plt.show()

def main():
    setup_logging()
    path = r'D:\DevelopmentCodes\ImportData'
    files = ["Oil.csv", "Gas.csv", "Water.csv", "WHP.csv", "Choke.csv", "Completion.csv", "Pressure.csv", "Sw.csv", "Petrophysics.csv", "DOP.csv"]
    
    data = load_data(path, files)
    well_names = ['L28-120_CH4']
    lookback, test_size_fraction = 20, 0.1
    
    for well in well_names:
        try:
            oil, zero_idx = clean_data(data['oil'].groupby('Well Name').get_group(well))
            pressure = remove_invalid_columns(data['pressure'].groupby('Well Name').get_group(well), zero_idx)
            gas = remove_invalid_columns(data['gas'].groupby('Well Name').get_group(well), zero_idx)
            
            input_feature = gas.iloc[:, 1:].values.T
            scaler = MinMaxScaler()
            input_data = scaler.fit_transform(input_feature)
            
            x_train, y_train, x_test, y_test = prepare_rnn_data(input_data, lookback, test_size_fraction)
            
            model = build_lstm_model((lookback, input_data.shape[1]))
            callbacks = [EarlyStopping(monitor='val_loss', patience=200),
                         ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
            
            model.fit(x_train, y_train, epochs=1000, batch_size=20, callbacks=callbacks, validation_data=(x_test, y_test))
            
            predicted_values = model.predict(x_test)
            plot_results(y_test, predicted_values)
            logging.info(f"Model training and evaluation completed for well: {well}")
        except Exception as e:
            logging.error(f"Error processing well {well}: {e}")

if __name__ == "__main__":
    main()


















