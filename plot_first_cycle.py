from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os

print('CWD:', os.getcwd())

# Load the same first-cycle data
data = loadmat('B0005.mat')
battery = data['B0005'][0,0]
cycle_data = battery['cycle'][0]
# first cycle
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

df = pd.DataFrame({'Time': time, 'Voltage': voltage, 'Current': current, 'Temp': temp})

# IQR-based mask
def iqr_mask(s):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (s < lower) | (s > upper)

voltage_iqr = iqr_mask(df['Voltage'])
current_iqr = iqr_mask(df['Current'])
temp_iqr = iqr_mask(df['Temp'])

# z-score (>3) mask (handle constant series)
voltage_z = np.abs(zscore(df['Voltage'])) > 3 if df['Voltage'].std() != 0 else np.zeros(len(df), dtype=bool)
current_z = np.abs(zscore(df['Current'])) > 3 if df['Current'].std() != 0 else np.zeros(len(df), dtype=bool)
temp_z = np.abs(zscore(df['Temp'])) > 3 if df['Temp'].std() != 0 else np.zeros(len(df), dtype=bool)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Voltage
ax = axs[0]
ax.plot(df['Time'], df['Voltage'], color='0.75', label='Voltage')
ax.scatter(df['Time'][voltage_iqr], df['Voltage'][voltage_iqr], color='orange', s=25, label='IQR outlier')
ax.scatter(df['Time'][voltage_z], df['Voltage'][voltage_z], color='red', s=40, marker='x', label='Z-score outlier')
ax.set_ylabel('Voltage (V)')
ax.legend(loc='best')
ax.grid(alpha=0.25)

# Current
ax = axs[1]
ax.plot(df['Time'], df['Current'], color='0.75', label='Current')
ax.scatter(df['Time'][current_iqr], df['Current'][current_iqr], color='orange', s=25, label='IQR outlier')
ax.scatter(df['Time'][current_z], df['Current'][current_z], color='red', s=40, marker='x', label='Z-score outlier')
ax.set_ylabel('Current (A)')
ax.legend(loc='best')
ax.grid(alpha=0.25)

# Temperature
ax = axs[2]
ax.plot(df['Time'], df['Temp'], color='0.75', label='Temperature')
ax.scatter(df['Time'][temp_iqr], df['Temp'][temp_iqr], color='orange', s=25, label='IQR outlier')
ax.scatter(df['Time'][temp_z], df['Temp'][temp_z], color='red', s=40, marker='x', label='Z-score outlier')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (°C)')
ax.legend(loc='best')
ax.grid(alpha=0.25)

plt.tight_layout()
# Save an image to the workspace for quick review
out_path = 'first_cycle_outliers.png'
fig.savefig(out_path, dpi=150)
print('Saved plot to', out_path)
plt.show()
