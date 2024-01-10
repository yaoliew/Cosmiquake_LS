import tensorflow as tf
import keras as ks
from keras import layers
from keras.layers.experimental import preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# earthquake data
#field_type: integer | datetime | float | float | float | string | string | string | string | string | float | string | string
eq_data = pd.read_csv('data/eq_data.txt', dtype={'EventID': np.int32, 'Time': str, 'Latitude': np.float64, 'Longitude': np.float64, 'Depth': np.float64, 'Author': 'string', 'Catalog': 'string', 'Contributor': 'string', 'ContributorID': 'string', 'MagType': 'string', 'Magnitude': np.float64, 'MagAuthor': 'string', 'EventLocationName': 'string'}, skiprows=4, sep='|')
eq_data['Time'] = pd.to_datetime(eq_data['Time'])
# time test: print(eq_data.iloc[1].time)

# cosmic ray data
cr_data = pd.read_csv('data/short_cr.csv')
cr_data['time'] = pd.to_datetime(cr_data['time'], unit='s')
# time test: print(cr_data.iloc[1].time)


# Data processing

# Create dataframe with date range
date_range = pd.date_range('2006-01-01', '2016-01-01')
cr_diff = pd.DataFrame({'Date': date_range})

# Add difference feature
for i, date in cr_diff.iterrows():
    print('run: ' + str(i))
    start = date['Date'] - pd.Timedelta(5, 'D')
    end = date['Date'] + pd.Timedelta(5, 'D')

    # Convert 'Time' column to timezone-aware datetime
    eq_data['Time'] = eq_data['Time'].dt.tz_convert('UTC')

    # Now 'start' and 'date['Date']' will be timezone-aware
    before = eq_data[(eq_data['Time'] >= start.tz_localize('UTC')) & (
                eq_data['Time'] < date['Date'].tz_localize('UTC'))]
    after = eq_data[(eq_data['Time'] >= date['Date'].tz_localize('UTC')) & (
                eq_data['Time'] < end.tz_localize('UTC'))]

    before_sum = before['Magnitude'].sum()
    after_sum = after['Magnitude'].sum()

    cr_diff.at[i, 'Difference'] = abs(before_sum - after_sum)


# print(df.head())

train_features = cr_diff.sample(frac=0.8, random_state=0).copy()
test_features = cr_diff.drop(train_features.index)

train_labels = eq_data.sample(frac=0.8, random_state=0).copy()
test_labels = eq_data.drop(train_features.index)


# Normalization
column_mean = cr_diff['Difference'].mean()
column_std = cr_diff['Difference'].std()
cr_diff['Difference'] = (cr_diff['Difference'] - column_mean) / column_std

print(cr_diff.head())






# ALT USGS EQ DATA
#
# for i in range(6, 17):
#     temp = pd.read_csv("short_earthquakes/eq_" + str(i))
#     earthquakes = pd.concat(earthquakes, temp)
#
# print(earthquakes)
