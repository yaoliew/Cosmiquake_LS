import tensorflow as tf
import keras as ks
from keras import layers
from keras.layers.experimental import preprocessing
from keras import regularizers
from keras.experimental import LinearModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Read in the CSVs
cr_diff = pd.read_csv('data/cr_diff.csv')
eq_sum = pd.read_csv('data/eq_sum.csv')

# Split the data
train_features = cr_diff.sample(frac=0.8, random_state=0).copy()
test_features = cr_diff.drop(train_features.index)

train_labels = eq_sum.sample(frac=0.8, random_state=0).copy()
test_labels = eq_sum.drop(train_labels.index)

# Normalization
column_mean = cr_diff['Difference'].mean()
column_std = cr_diff['Difference'].std()
cr_diff['Difference'] = (cr_diff['Difference'] - column_mean) / column_std

threshold = 7000

# Create boolean mask
mask = cr_diff['Difference'] > threshold

# Set values to NaN where mask is True
cr_diff.loc[mask, 'Difference'] = np.nan

# Merge cr_diff and eq_sum on Date
merged = pd.merge(cr_diff, eq_sum, on='Date')

# Create X and y with 15 day offset
X = merged['Difference']
y = merged['EQ_Sum'].shift(-15)

# Remove rows with NaN values
nan_mask = np.isnan(y)
y = y[~nan_mask]
X = X[~nan_mask]

# Sequential model
model = ks.models.Sequential([
  layers.Dense(units=1, use_bias=True)
])

# Loss and optimizer
loss = ks.losses.MeanAbsoluteError()
optim = ks.optimizers.Adam(lr=0.01)

model.compile(optimizer=optim, loss=loss)

# Fit model
history = model.fit(
  X, y,
  epochs=100,
  verbose=1,
  validation_split=0.2
)


print(model.evaluate(
    test_features['Difference'],
    test_labels['EQ_Sum'], verbose=1))

def plot_loss(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='loss (training data)')
    plt.plot(history.history['val_loss'], label='val_loss (validation data)')
    plt.title('Loss for Earthquake Prediction Model')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper right")
    plt.show()

plot_loss(history)
plt.show()

# plot line of best fit
range_min = np.min(test_features['Difference']) - 10
range_max = np.max(test_features['Difference']) + 10
x_values = tf.linspace(range_min, range_max, 200)
# Reshape input to 2D
x_values = x_values[:,np.newaxis]
y_pred = model.predict(x_values)

# Convert the TensorFlow tensor to a NumPy array for plotting
x_values = x_values.numpy().flatten()
y_pred = y_pred.flatten()

# Ensure that x_values and y_pred have the same dimensions
assert len(x_values) == len(y_pred), "x and y must have the same length"

# Plotting the predicted line
plt.plot(x_values, y_pred, label='Predicted')

# Plotting the actual data points
plt.scatter(test_features['Difference'], test_labels['EQ_Sum'], color='red', label='Actual Data')

# Adding labels and legend
plt.xlabel('Difference')
plt.ylabel('EQ_Sum')
plt.legend()

plt.show()