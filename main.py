import tensorflow as tf
import keras as ks
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta


# earthquake data
#field_type: integer | datetime | float | float | float | string | string | string | string | string | float | string | string
eq = pd.read_csv('data/eq_data.txt', dtype={'EventID': np.int32, 'Time': str, 'Latitude': np.float64, 'Longitude': np.float64, 'Depth': np.float64, 'Author': 'string', 'Catalog': 'string', 'Contributor': 'string', 'ContributorID': 'string', 'MagType': 'string', 'Magnitude': np.float64, 'MagAuthor': 'string', 'EventLocationName': 'string'}, skiprows=4, sep='|')
eq['Time'] = pd.to_datetime(eq['Time'])
# time test: print(eq_data.iloc[1].time)

# cosmic ray data
cr_data = pd.read_csv('data/scalers.csv')
cr_data['time'] = pd.to_datetime(cr_data['time'], unit='s')
# time test: print(cr_data.iloc[1].time)








# ALT USGS EQ DATA
# cosmic_rays = pd.read_csv("scalers.csv")
#
# for i in range(6, 17):
#     temp = pd.read_csv("short_earthquakes/eq_" + str(i))
#     earthquakes = pd.concat(earthquakes, temp)
#
# print(earthquakes)
