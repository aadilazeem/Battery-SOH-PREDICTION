"""
battery_feature_pipeline.py

Extract physics-informed per-cycle features from the NASA battery .mat files (B0005, B0006, ...),
produce a CSV `features_per_cycle.csv` and basic EDA plots saved to `4_Results_and_Graphs/`.

Features (per cycle):
 - file (cell id), cycle_index
 - n_points
 - time_duration (s)
 - voltage_start, voltage_end, voltage_delta, voltage_slope (V/s) [linear fit slope]
 - current_mean, current_std
 - temp_start, temp_end, temp_change
 - energy_proxy = sum(V * I * dt) (approximate Wh-like proxy; units: V*A*s)
 - capacity (if available), capacity_ratio (w.r.t. first cycle capacity in that file)
 - voltage_std

The script is defensive: it handles missing fields and writes NaN when a value is unavailable.
"""

import os
import glob
from scipy.io import loadmat
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

OUT_DIR = '4_Results_and_Graphs'
os.makedirs(OUT_DIR, exist_ok=True)

def to1d(a):
    arr = np.array(a)
    if arr.ndim > 1:
        arr = arr[0]
    return arr.ravel()

records = []

mat_files = sorted(glob.glob('B*.mat'))
if not mat_files:
    print('No B*.mat files found in', os.getcwd())

for matf in mat_files:
    print('Processing', matf)
    data = loadmat(matf)
    # find the top key starting with 'B'
    key = next((k for k in data.keys() if isinstance(k,str) and k.startswith('B')), None)
    if key is None:
        print('No B* key in', matf)
        continue
    battery = data[key][0,0]
    cycle_data = battery['cycle'][0]
    first_capacity = None
    for ci, c in enumerate(cycle_data):
        # robust extraction of measurement table
        try:
            meas = c['data'][0,0]
        except Exception:
            # try alternate indexing
            try:
                meas = c['data']
            except Exception:
                print(f'  cycle {ci}: no data field')
                continue
        # attempt to extract fields
        def get_field(name):
            if name in meas.dtype.names:
                return to1d(meas[name])
            return None

        V = get_field('Voltage_measured')
        I = get_field('Current_measured')
        T = get_field('Temperature_measured')
        Time = get_field('Time')
        Cap = None
        # capacity sometimes stored in cycle summary or meas
        if 'Capacity' in meas.dtype.names:
            Cap = to1d(meas['Capacity'])
        else:
            # try cycle summary
            try:
                if 'summary' in c.dtype.names:
                    summary = c['summary'][0]
                    if isinstance(summary, np.ndarray):
                        # try to find fields with 'Capacity' in name
                        for fn in summary.dtype.names:
                            if 'cap' in fn.lower():
                                v = summary[fn]
                                try:
                                    Cap = float(v)
                                except Exception:
                                    Cap = None
            except Exception:
                pass

        # Build features if V,I,Time exist
        n_points = 0
        duration = np.nan
        v_start = v_end = v_delta = v_slope = np.nan
        i_mean = i_std = np.nan
        t_start = t_end = t_change = np.nan
        energy = np.nan
        v_std = np.nan

        if V is None or I is None or Time is None:
            # Still try to salvage some stats
            if V is not None:
                n_points = len(V)
                v_std = np.nanstd(V)
            records.append({
                'file': matf,
                'cell_key': key,
                'cycle_index': ci,
                'n_points': n_points,
                'duration_s': duration,
                'voltage_start': v_start,
                'voltage_end': v_end,
                'voltage_delta': v_delta,
                'voltage_slope_V_per_s': v_slope,
                'voltage_std': v_std,
                'current_mean': i_mean,
                'current_std': i_std,
                'temp_start': t_start,
                'temp_end': t_end,
                'temp_change': t_change,
                'energy_proxy_VAs': energy,
                'capacity': np.nan,
                'capacity_ratio': np.nan,
            })
            continue

        # ensure monotonic time - convert to float
        Time = Time.astype(float)
        V = V.astype(float)
        I = I.astype(float)
        n_points = len(Time)
        if n_points >= 2:
            duration = float(Time[-1] - Time[0])
            v_start = float(V[0])
            v_end = float(V[-1])
            v_delta = v_end - v_start
            # linear regression (slope) in V per s
            try:
                slope = linregress(Time, V).slope
                v_slope = float(slope)
            except Exception:
                v_slope = float((V[-1]-V[0]) / (Time[-1]-Time[0])) if (Time[-1]-Time[0])!=0 else np.nan
            i_mean = float(np.mean(I))
            i_std = float(np.std(I))
            t_start = float(T[0]) if T is not None else np.nan
            t_end = float(T[-1]) if T is not None else np.nan
            t_change = t_end - t_start if (not np.isnan(t_start) and not np.isnan(t_end)) else np.nan
            v_std = float(np.std(V))
            # energy proxy: sum V*I*dt (units V*A*s)
            dt = np.diff(Time)
            if len(dt) > 0:
                VI = V[:-1] * I[:-1]
                energy = float(np.sum(VI * dt))
            else:
                energy = np.nan

        # capacity handling
        cycle_capacity = np.nan
        try:
            if Cap is not None:
                # if Cap is an array per timepoint, take last value as cycle capacity
                if hasattr(Cap, '__len__') and len(Cap) > 0:
                    cycle_capacity = float(Cap[-1])
                else:
                    cycle_capacity = float(Cap)
        except Exception:
            cycle_capacity = np.nan

        if first_capacity is None and not np.isnan(cycle_capacity):
            first_capacity = cycle_capacity

        cap_ratio = np.nan
        if not np.isnan(cycle_capacity) and first_capacity is not None and first_capacity != 0:
            cap_ratio = cycle_capacity / first_capacity

        records.append({
            'file': matf,
            'cell_key': key,
            'cycle_index': ci,
            'n_points': n_points,
            'duration_s': duration,
            'voltage_start': v_start,
            'voltage_end': v_end,
            'voltage_delta': v_delta,
            'voltage_slope_V_per_s': v_slope,
            'voltage_std': v_std,
            'current_mean': i_mean,
            'current_std': i_std,
            'temp_start': t_start,
            'temp_end': t_end,
            'temp_change': t_change,
            'energy_proxy_VAs': energy,
            'capacity': cycle_capacity,
            'capacity_ratio': cap_ratio,
        })

# Build DataFrame
if records:
    features = pd.DataFrame.from_records(records)
    features.to_csv(os.path.join(OUT_DIR, 'features_per_cycle.csv'), index=False)
    print('Wrote features to', os.path.join(OUT_DIR, 'features_per_cycle.csv'))

    # Basic EDA plots per file: Capacity_ratio vs cycle and Voltage_slope vs cycle
    for fname, group in features.groupby('file'):
        base = os.path.basename(fname).replace('.mat','')
        # Capacity_ratio vs cycle
        plt.figure(figsize=(8,4))
        if group['capacity_ratio'].notna().any():
            plt.plot(group['cycle_index'], group['capacity_ratio'], marker='o')
            plt.xlabel('Cycle index')
            plt.ylabel('Capacity ratio')
            plt.title(f'{base} - Capacity ratio per cycle')
            plt.grid(alpha=0.25)
            out = os.path.join(OUT_DIR, f'{base}_capacity_ratio.png')
            plt.savefig(out, dpi=150)
            plt.close()
            print('Saved', out)
        # Voltage slope
        plt.figure(figsize=(8,4))
        if group['voltage_slope_V_per_s'].notna().any():
            plt.plot(group['cycle_index'], group['voltage_slope_V_per_s'], marker='o')
            plt.xlabel('Cycle index')
            plt.ylabel('Voltage slope (V/s)')
            plt.title(f'{base} - Voltage slope per cycle')
            plt.grid(alpha=0.25)
            out = os.path.join(OUT_DIR, f'{base}_voltage_slope.png')
            plt.savefig(out, dpi=150)
            plt.close()
            print('Saved', out)

    # Correlation matrix (Pearson)
    corr = features.select_dtypes(float).corr()
    corr.to_csv(os.path.join(OUT_DIR, 'features_correlation.csv'))
    print('Saved correlation matrix to', os.path.join(OUT_DIR, 'features_correlation.csv'))

    # Save a small summary
    summary = features.describe().transpose()
    summary.to_csv(os.path.join(OUT_DIR, 'features_summary.csv'))
    print('Saved summary stats to', os.path.join(OUT_DIR, 'features_summary.csv'))
else:
    print('No feature records were created')
