import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dropout, Dense

# Load the data
data = pd.read_csv('EastSikkim_processed.csv')

# Select relevant attributes
relevant_columns = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'aqi']
data = data[relevant_columns]

# Preprocessing the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Define input and output attributes
X = scaled_data[:, :-1]  # Features: AQI attributes (co, no2, o3, pm10, pm25, so2)
y = scaled_data[:, -1]   # Target: AQI

# Reshape the input data to fit the LSTM input shape
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = tf.keras.Sequential([
    LSTM(units=36, input_shape=(1, X_train.shape[2]), activation="relu", return_sequences=True),
    Dropout(0.1),
    Dense(28, activation="relu", kernel_regularizer='l2'),
    Dropout(0.1),
    Dense(22, activation="relu", kernel_regularizer='l2'),
    Dropout(0.1),
    Dense(22, activation="relu", kernel_regularizer='l2'),
    Dropout(0.1),
    Dense(22, activation="relu", kernel_regularizer='l2'),
    Dropout(0.1),
    LSTM(units=16, activation="relu", kernel_regularizer='l2'),
    Dropout(0.1),
    Dense(12, activation="relu", kernel_regularizer='l2'),
    Dropout(0.1),
    Dense(8, activation="relu", kernel_regularizer='l2'),
    Dropout(0.1),
    Dense(6, activation="relu", kernel_regularizer='l2'),
    Dropout(0.1),
    Dense(2, activation="relu", kernel_regularizer='l2'),
    Dropout(0.1),
    Dense(1, activation="relu")
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
