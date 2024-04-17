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

data = pd.read_csv('SouthSikkim_processed.csv')

features = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']
data = data[features]

# Normalize 
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# function to prepare data for LSTM
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
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
    Dense(6)  # Adjusted for predicting 6 attributes
])

# Compile
# Evaluate
prediction = model.predict(X_test)

# Calculate evaluation metrics
rmse = sqrt(mean_squared_error(Y_test, prediction))
print("RMSE Score is", rmse)

mse = mean_squared_error(Y_test, prediction)
print("MSE Score is", mse)

mean_abs_error = mean_absolute_error(Y_test, prediction)
print("Mean absolute error is", mean_abs_error)
# Save the model

# Evaluate
prediction = model.predict(X_test)

# Calculate evaluation metrics
rmse = sqrt(mean_squared_error(Y_test, prediction))
print("RMSE Score is", rmse)

mse = mean_squared_error(Y_test, prediction)
print("MSE Score is", mse)

mean_abs_error = mean_absolute_error(Y_test, prediction)
print("Mean absolute error is", mean_abs_error)

model.save('my_model.keras')

# def plotit(title, x, y, anch):
#     plt.figure(dpi=300)
#     plt.rcParams.update({'font.family': 'Times New Roman'})
#     temp = title.split(" ")
#     f = temp[1]
#     l = temp[3]
#     plt.xlabel("Epochs")
#     plt.ylabel("Value")
#     plt.plot(x, marker='.')
#     plt.plot(y, marker='.')
#     plt.grid(color='#2A3459') 
#     ax = plt.axes()
#     plt.legend([f + ": " + str(round(x[-1], 3)), l + ": " + str(round(y[-1], 3))], loc='right', bbox_to_anchor=anch)
#     plt.savefig(title + ".jpg")
#     plt.savefig(title + ".tiff")

# # Plot accuracy
# plotit("MFCC Training and Validation Accuracy for 10 second samples", result.history['accuracy'], result.history['val_accuracy'], (0.5, 0., 0.5, 0.4))

# # Plot loss
# plotit("MFCC Loss and Validation Loss for 10 second samples", result.history['loss'], result.history['val_loss'], (0.5, 0., 0.5, 1.75))

# plt.show()