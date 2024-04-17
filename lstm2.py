import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt

# Load data 
data = pd.read_csv('NorthSikkim_processed.csv')

# Select
features = ['aqi', "temperature_2m","relativehumidity_2m","dewpoint_2m",'apparent_temperature','precipitation','rain','snowfall','pressure_msl','cloudcover']
data = data[features]

# Normalize 
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# function to prepare data for LSTM
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 1:])  # Exclude AQI from features
        y.append(data[i + time_steps, 0])  # AQI is the first column
    return np.array(X), np.array(y)

# Define time steps
time_steps = 1

# Prepare data for LSTM
X, y = prepare_data(data_normalized, time_steps)

# Split data into train, test, and validation sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=12)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.4, random_state=10)

# model architecture
model = Sequential([
    LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(units=64, return_sequences=True),
    Dropout(0.2),
    LSTM(units=32),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1)  # Adjusted for predicting AQI
])

# Compile
optimizer = 'adam'
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model and store the result
result = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=32, epochs=100, callbacks=[early_stopping], verbose=1)

# Evaluate
prediction = model.predict(X_test)

# Calculate evaluation metrics
rmse = sqrt(mean_squared_error(Y_test, prediction))
print("RMSE Score is", rmse)

mse = mean_squared_error(Y_test, prediction)
print("MSE Score is", mse)

mean_abs_error = mean_absolute_error(Y_test, prediction)
print("Mean absolute error is", mean_abs_error)
