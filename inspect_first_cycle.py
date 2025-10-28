from scipy.io import loadmat
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os

print('CWD:', os.getcwd())

# Load file and extract first cycle measurement
data = loadmat('B0005.mat')
battery = data['B0005'][0,0]
cycle_data = battery['cycle'][0]
# line 17 corresponds to: c = cycle_data[0]
c = cycle_data[0]
meas = c['data'][0,0]

# helper to flatten arrays
def to1d(a):
    arr = np.array(a)
    if arr.ndim > 1:
        arr = arr[0]
    return arr.ravel()

voltage = to1d(meas['Voltage_measured'])
current = to1d(meas['Current_measured'])
temp = to1d(meas['Temperature_measured'])
time = to1d(meas['Time'])

df = pd.DataFrame({'Voltage': voltage, 'Current': current, 'Temp': temp, 'Time': time})

print('Data shape:', df.shape)
print('\nMissing values per column:\n', df.isnull().sum())

print('\nBasic stats:\n', df.describe())

# IQR-based outliers
outlier_summary = {}
for col in ['Voltage','Current','Temp']:
    s = df[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (s < lower) | (s > upper)
    outlier_summary[col] = {'count': int(mask.sum()), 'pct': float(mask.mean() * 100), 'lower': float(lower), 'upper': float(upper)}

print('\nIQR outlier summary:')
for k,v in outlier_summary.items():
    print(k, v)

# z-score based outliers (>3)
z_summary = {}
for col in ['Voltage','Current','Temp']:
    s = df[col].dropna()
    if s.std() == 0:
        z_summary[col] = {'count': 0, 'pct': 0.0}
    else:
        zm = np.abs(zscore(s)) > 3
        z_summary[col] = {'count': int(zm.sum()), 'pct': float(zm.mean() * 100)}

print('\nZ-score outlier summary:')
for k,v in z_summary.items():
    print(k, v)

# Show a few extreme rows for each column
for col in ['Voltage','Current','Temp']:
    print(f"\nTop 3 lowest {col} rows:")
    print(df.nsmallest(3, col))
    print(f"Top 3 highest {col} rows:")
    print(df.nlargest(3, col))

# Small sample of rows where any IQR outlier occurs
masks = []
for col in ['Voltage','Current','Temp']:
    s = df[col]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    masks.append((s < lower) | (s > upper))
combined_mask = np.logical_or.reduce(masks)
print('\nNumber of rows with any IQR outlier:', int(combined_mask.sum()))
print('\nSample rows with outliers (up to 10):')
print(df[combined_mask].head(10))
