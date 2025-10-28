"""
battery_model_pipeline.py

Full primary pipeline (Phase 3-7 primary path):
 - Recompute per-cycle features with charge/discharge split
 - Save features to 4_Results_and_Graphs/features_primary.csv
 - Run EDA (plots + correlation)
 - Train RandomForest (leave-one-file-out at cell/file level) to predict capacity_ratio
 - Save CV metrics and averaged feature importances
 - Produce PDPs for top features and save plots

Outputs (saved to 4_Results_and_Graphs):
 - features_primary.csv
 - modeling_metrics.csv
 - feature_importances.csv
 - PDP plots (png)

Notes:
 - Target: capacity_ratio (cycles where capacity is available)
 - If capacity missing, those cycles are excluded from modeling steps.
"""

import os
import glob
from scipy.io import loadmat
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay

OUT_DIR = '4_Results_and_Graphs'
os.makedirs(OUT_DIR, exist_ok=True)

# helper
def to1d(a):
    arr = np.array(a)
    if arr.ndim > 1:
        arr = arr[0]
    return arr.ravel()

# feature extraction with charge/discharge splitting
records = []
mat_files = sorted(glob.glob('B*.mat'))
if not mat_files:
    print('No B*.mat files found in', os.getcwd())

for matf in mat_files:
    data = loadmat(matf)
    key = next((k for k in data.keys() if isinstance(k,str) and k.startswith('B')), None)
    if key is None:
        continue
    battery = data[key][0,0]
    cycle_data = battery['cycle'][0]
    first_capacity = None
    for ci, c in enumerate(cycle_data):
        try:
            meas = c['data'][0,0]
        except Exception:
            continue
        def get_field(name):
            if meas is None:
                return None
            if name in meas.dtype.names:
                return to1d(meas[name]).astype(float)
            return None
        V = get_field('Voltage_measured')
        I = get_field('Current_measured')
        T = get_field('Temperature_measured')
        Time = get_field('Time')
        # capacity attempts (may be missing)
        Cap = None
        if 'Capacity' in meas.dtype.names:
            Cap = to1d(meas['Capacity']).astype(float)
        else:
            # try cycle summary
            try:
                if 'summary' in c.dtype.names:
                    summary = c['summary'][0]
                    if isinstance(summary, np.ndarray):
                        for fn in summary.dtype.names:
                            if 'cap' in fn.lower():
                                try:
                                    Cap = float(summary[fn])
                                except Exception:
                                    Cap = None
            except Exception:
                pass

        # baseline
        cycle_capacity = np.nan
        if Cap is not None:
            try:
                if hasattr(Cap, '__len__'):
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

        # initialize feature placeholders
        n_points = 0
        duration = np.nan
        v_start = v_end = v_delta = v_slope = np.nan
        v_std = np.nan
        i_mean = i_std = np.nan
        t_start = t_end = t_change = np.nan
        energy = np.nan
        # discharge-specific
        discharge_frac = np.nan
        discharge_v_slope = np.nan
        discharge_curvature = np.nan
        plateau_frac = np.nan

        if V is None or I is None or Time is None:
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
                'discharge_frac': discharge_frac,
                'discharge_voltage_slope': discharge_v_slope,
                'discharge_curvature_coeff': discharge_curvature,
                'voltage_plateau_fraction': plateau_frac,
            })
            continue

        Time = Time.astype(float)
        V = V.astype(float)
        I = I.astype(float)
        n_points = len(Time)
        if n_points >= 2:
            duration = float(Time[-1] - Time[0])
            v_start = float(V[0])
            v_end = float(V[-1])
            v_delta = v_end - v_start
            # global slope
            try:
                v_slope = float(linregress(Time, V).slope)
            except Exception:
                v_slope = float((V[-1]-V[0])/(Time[-1]-Time[0])) if (Time[-1]-Time[0])!=0 else np.nan
            v_std = float(np.std(V))
            i_mean = float(np.mean(I))
            i_std = float(np.std(I))
            t_start = float(T[0]) if T is not None else np.nan
            t_end = float(T[-1]) if T is not None else np.nan
            t_change = t_end - t_start if (not np.isnan(t_start) and not np.isnan(t_end)) else np.nan
            # energy proxy
            dt = np.diff(Time)
            if len(dt) > 0:
                VI = V[:-1] * I[:-1]
                energy = float(np.sum(VI * dt))
            # discharge splitting
            dis_mask = I < 0
            discharge_frac = float(np.mean(dis_mask))
            if dis_mask.any():
                t_dis = Time[dis_mask]
                v_dis = V[dis_mask]
                i_dis = I[dis_mask]
                # slope on discharge
                try:
                    discharge_v_slope = float(linregress(t_dis, v_dis).slope)
                except Exception:
                    discharge_v_slope = np.nan
                # curvature: quadratic fit v = a*t^2 + b*t + c  -> curvature ~ a
                if len(t_dis) >= 3:
                    try:
                        coeffs = np.polyfit(t_dis, v_dis, 2)
                        discharge_curvature = float(coeffs[0])
                    except Exception:
                        discharge_curvature = np.nan
                # plateau fraction: fraction of discharge where |dV/dt| < threshold
                dvdt = np.abs(np.diff(v_dis) / np.diff(t_dis)) if len(t_dis) > 1 else np.array([])
                if dvdt.size > 0:
                    thr = 1e-4  # V/s small change threshold; heuristic
                    plateau_frac = float(np.mean(dvdt < thr))
                else:
                    plateau_frac = 0.0
            else:
                discharge_v_slope = np.nan
                discharge_curvature = np.nan
                plateau_frac = 0.0

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
            'discharge_frac': discharge_frac,
            'discharge_voltage_slope': discharge_v_slope,
            'discharge_curvature_coeff': discharge_curvature,
            'voltage_plateau_fraction': plateau_frac,
        })

features = pd.DataFrame.from_records(records)
feat_path = os.path.join(OUT_DIR, 'features_primary.csv')
features.to_csv(feat_path, index=False)
print('Wrote features to', feat_path)

# EDA: distributions and correlations
features.select_dtypes(include=[float]).hist(bins=30, figsize=(12,8))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'features_histograms.png'), dpi=150)
plt.close()

corr = features.select_dtypes(float).corr()
corr.to_csv(os.path.join(OUT_DIR, 'features_primary_correlation.csv'))
plt.figure(figsize=(8,6))
plt.matshow(corr, fignum=1)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Feature correlation matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'features_primary_correlation.png'), dpi=150)
plt.close()

# Modeling: predict capacity_ratio using leave-one-file-out CV
# use only rows where capacity_ratio is finite
model_df = features[~features['capacity_ratio'].isna()].copy()
if model_df.empty:
    print('No cycles with capacity_ratio available for modeling. Exiting modeling steps.')
else:
    feature_cols = [
        'n_points','duration_s','voltage_start','voltage_end','voltage_delta','voltage_slope_V_per_s','voltage_std',
        'current_mean','current_std','temp_start','temp_end','temp_change','energy_proxy_VAs',
        'discharge_frac','discharge_voltage_slope','discharge_curvature_coeff','voltage_plateau_fraction'
    ]
    # keep only available columns
    feature_cols = [c for c in feature_cols if c in model_df.columns]
    model_df = model_df.dropna(subset=feature_cols + ['capacity_ratio'])
    files = model_df['file'].unique()
    metrics = []
    importances = []
    for test_file in files:
        train_df = model_df[model_df['file'] != test_file]
        test_df = model_df[model_df['file'] == test_file]
        if train_df.shape[0] < 5 or test_df.shape[0] < 1:
            print('Skipping LOFO for', test_file, 'insufficient data')
            continue
        X_train = train_df[feature_cols].values
        y_train = train_df['capacity_ratio'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['capacity_ratio'].values
        # scale
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        # RF
        rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        rf.fit(X_train_s, y_train)
        y_pred = rf.predict(X_test_s)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        metrics.append({'test_file': test_file, 'n_test': len(y_test), 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        importances.append(rf.feature_importances_)
        # save predictions per test file
        outpred = test_df[['file','cell_key','cycle_index']].copy()
        outpred['y_true'] = y_test
        outpred['y_pred'] = y_pred
        outpred.to_csv(os.path.join(OUT_DIR, f'predictions_{os.path.basename(test_file)}.csv'), index=False)
        print('Finished LOFO test on', test_file, 'MAE=', mae)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(OUT_DIR, 'modeling_metrics.csv'), index=False)
    print('Saved modeling metrics to', os.path.join(OUT_DIR, 'modeling_metrics.csv'))

    # average feature importances across folds
    if importances:
        mean_imp = np.mean(importances, axis=0)
        imp_df = pd.DataFrame({'feature': feature_cols, 'importance': mean_imp})
        imp_df = imp_df.sort_values('importance', ascending=False)
        imp_df.to_csv(os.path.join(OUT_DIR, 'feature_importances.csv'), index=False)
        print('Saved feature importances to', os.path.join(OUT_DIR, 'feature_importances.csv'))

        # Train final RF on all data for PDPs
        X_all = model_df[feature_cols].values
        y_all = model_df['capacity_ratio'].values
        scaler_all = StandardScaler().fit(X_all)
        X_all_s = scaler_all.transform(X_all)
        rf_all = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
        rf_all.fit(X_all_s, y_all)

        top_feats = imp_df['feature'].tolist()[:3]
        for feat in top_feats:
            feat_idx = feature_cols.index(feat)
            fig, ax = plt.subplots(figsize=(6,4))
            try:
                PartialDependenceDisplay.from_estimator(rf_all, X_all_s, [feat_idx], feature_names=feature_cols, ax=ax)
                plt.tight_layout()
                out = os.path.join(OUT_DIR, f'pdp_{feat}.png')
                fig.savefig(out, dpi=150)
                plt.close(fig)
                print('Saved PDP for', feat, '->', out)
            except Exception as e:
                print('PDP failed for', feat, str(e))

    else:
        print('No importances computed (no LOFO runs)')

print('Pipeline complete.')
import pandas as pd
df = pd.read_csv('features_primary.csv')