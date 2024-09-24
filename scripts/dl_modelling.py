import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from scripts.logger import setup_logger
logger = setup_logger('dl_logger', '../logs/lstm.log')
def check_stationarity(timeseries):
    """Check whether your time Series Data is Stationary."""
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic: {:.10f}'.format(result[0]))
    print('p-value: {:.10f}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {:.10f}'.format(key, value))
    
    # If p-value > 0.05, the time series is non-stationary
    if result[1] > 0.05:
        print("The time series is non-stationary")
    else:
        print("The time series is stationary")


def create_supervised_data(data, n_step=1):
    """Transform the time series data into supervised learning data"""
    X, y = [], []
    for i in range(len(data) - n_step - 1):
        X.append(data[i:(i + n_step), 0])
        y.append(data[i + n_step, 0])

        logger.info(f'Supervised data created with n_step = {n_step}')
        logger.info(f'X shape: {np.array(X).shape}, y shape: {np.array(y).shape}')
    return np.array(X), np.array(y)



def build_lstm_model(n_step):
    # Build LSTM Regression model
    model = Sequential()
    model.add(Input(shape=(n_step, 1)))  # Define the input shape
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    logger.info('LSTM model built with {} steps'.format(n_step))
    return model

def train_lstm_model(model, X, y, epochs=50, batch_size=32, validation_split=0.2):
    # Fit the model and store the history
    logger.info(f'Training LSTM model with epochs={epochs}, batch_size={batch_size}, validation_split={validation_split}')
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    logger.info('Model training complete')
    return history
