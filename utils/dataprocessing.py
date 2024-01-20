import numpy as np
import pandas as pd

# earthquake data
#field_type: integer | datetime | float | float | float | string | string | string | string | string | float | string | string
eq_data = pd.read_csv('data/eq_data.txt', dtype={'EventID': np.int32, 'Time': str, 'Latitude': np.float64, 'Longitude': np.float64, 'Depth': np.float64, 'Author': 'string', 'Catalog': 'string', 'Contributor': 'string', 'ContributorID': 'string', 'MagType': 'string', 'Magnitude': np.float64, 'MagAuthor': 'string', 'EventLocationName': 'string'}, skiprows=4, sep='|')
eq_data['Time'] = pd.to_datetime(eq_data['Time'])

# Convert 'Time' column to timezone-aware datetime
eq_data['Time'] = eq_data['Time'].dt.tz_convert('UTC')


# cosmic ray data
cr_data = pd.read_csv('data/short_cr.csv')
cr_data['time'] = pd.to_datetime(cr_data['time'], unit='s')
# time test: print(cr_data.iloc[1].time)


# Data processing

# Add CR difference feature
date_range = pd.date_range('2006-01-01', '2016-01-01')
cr_diff = pd.DataFrame({'Date': date_range})

for i, date in cr_diff.iterrows():
    print('cr run: ' + str(i))
    start = date['Date'] - pd.Timedelta(5, 'D')
    end = date['Date'] + pd.Timedelta(5, 'D')

    before = eq_data[(eq_data['Time'] >= start.tz_localize('UTC')) & (eq_data['Time'] < date['Date'].tz_localize('UTC'))]
    after = eq_data[(eq_data['Time'] >= date['Date'].tz_localize('UTC')) & (eq_data['Time'] < end.tz_localize('UTC'))]

    before_sum = before['Magnitude'].sum()
    after_sum = after['Magnitude'].sum()

    cr_diff.at[i, 'Difference'] = abs(before_sum - after_sum)

# Test: print(cr_diff.head())

# Add EQ sum feature

eq_sum = pd.DataFrame()
eq_sum['Date'] = cr_diff['Date']

for i, date in eq_sum.iterrows():
    print('eq run: ' + str(i))

    start = date['Date'] - pd.Timedelta(5, 'D')
    end = date['Date']

    eq_in_range = eq_data[(eq_data['Time'] >= start.tz_localize('UTC')) & (eq_data['Time'] <= end.tz_localize('UTC'))]
    eq_sum.at[i, 'EQ_Sum'] = eq_in_range['Magnitude'].sum()

# Save cr_diff to CSV
cr_diff.to_csv('cr_diff.csv', index=False)

# Save eq_sum to CSV
eq_sum.to_csv('eq_sum.csv', index=False)