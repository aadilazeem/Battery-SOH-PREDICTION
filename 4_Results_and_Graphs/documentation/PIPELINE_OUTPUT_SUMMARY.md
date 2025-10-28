# Battery Feature Pipeline - Output Documentation

**Generated:** 2025-10-26_14:11:40


## 1. Overview

This documentation describes the output of the battery_feature_pipeline.py script,
which extracts physics-informed per-cycle features from NASA battery .mat files.

The pipeline processes raw time-series data (voltage, current, temperature)
and computes degradation-relevant features for each charge/discharge cycle.


**Output Location:** `d:\project battery ML_Apply\first part of nasa dataset\4_Results_and_Graphs`

**Total Cycles Processed:** 2167

**Unique Battery Files:** 4


## 2. Output Files


| Filename | Type | Size (KB) | Description |

|----------|------|-----------|-------------|

| `B0005_capacity_ratio.png` | Visualization | 59.02 | State of Health (SOH) vs cycle number |

| `B0005_voltage_slope.png` | Visualization | 40.93 | Voltage degradation rate vs cycle |

| `B0006_capacity_ratio.png` | Visualization | 58.33 | State of Health (SOH) vs cycle number |

| `B0006_voltage_slope.png` | Visualization | 40.34 | Voltage degradation rate vs cycle |

| `B0007_capacity_ratio.png` | Visualization | 61.13 | State of Health (SOH) vs cycle number |

| `B0007_voltage_slope.png` | Visualization | 39.96 | Voltage degradation rate vs cycle |

| `B0018_capacity_ratio.png` | Visualization | 54.48 | State of Health (SOH) vs cycle number |

| `B0018_voltage_slope.png` | Visualization | 190.75 | Voltage degradation rate vs cycle |

| `battery_sample.tsv` | Other | 1.94 | Miscellaneous output |

| `documentation` | Other | 0 | Miscellaneous output |

| `feature_importances.csv` | CSV Data | 0.63 | Data export |

| `features_correlation.csv` | CSV Data | 5.11 | Pearson correlation between all numeric features |

| `features_histograms.png` | Visualization | 121.4 | EDA plot |

| `features_per_cycle.csv` | CSV Data | 366.13 | Main feature matrix - one row per cycle |

| `features_primary.csv` | CSV Data | 465.25 | Data export |


## 3. Feature Definitions


The `features_per_cycle.csv` contains the following columns:


### `file`

- **Type:** string

- **Description:** Battery .mat filename (e.g., B0005.mat)

- **Physics:** Cell identifier for grouping cycles

- **Units:** N/A

- **Example:** B0005.mat



### `cell_key`

- **Type:** string

- **Description:** MATLAB key for battery object inside .mat file

- **Physics:** Internal data structure identifier

- **Units:** N/A

- **Example:** B0005



### `cycle_index`

- **Type:** integer

- **Description:** Zero-indexed charge/discharge cycle number

- **Physics:** Ordinal position in battery lifetime

- **Units:** cycles

- **Example:** 0, 1, 2, ..., N



### `n_points`

- **Type:** integer

- **Description:** Number of time-series measurement points in this cycle

- **Physics:** Data resolution / sampling density

- **Units:** samples

- **Example:** 1024



### `duration_s`

- **Type:** float

- **Description:** Total time duration of one cycle (Time[-1] - Time[0])

- **Physics:** Cycle execution time; affects thermal effects

- **Units:** seconds

- **Example:** 3600.5



### `voltage_start`

- **Type:** float

- **Description:** Voltage at beginning of cycle (V[0])

- **Physics:** Initial state-of-charge indicator

- **Units:** Volts

- **Example:** 4.1



### `voltage_end`

- **Type:** float

- **Description:** Voltage at end of cycle (V[-1])

- **Physics:** Final state-of-charge indicator

- **Units:** Volts

- **Example:** 2.0



### `voltage_delta`

- **Type:** float

- **Description:** Total voltage change during cycle (V_end - V_start)

- **Physics:** Charge/discharge depth; impacts degradation rate

- **Units:** Volts

- **Example:** -2.1



### `voltage_slope_V_per_s`

- **Type:** float

- **Description:** Linear regression slope of V vs Time

- **Physics:** ★ KEY INDICATOR: voltage degradation rate

- **Units:** V/s

- **Example:** -0.0006

- **Interpretation:** More negative = faster discharge = more aged battery



### `voltage_std`

- **Type:** float

- **Description:** Standard deviation of voltage measurements

- **Physics:** Voltage stability; high values indicate noise/artifacts

- **Units:** Volts

- **Example:** 0.045



### `current_mean`

- **Type:** float

- **Description:** Mean current over entire cycle

- **Physics:** Average charge/discharge intensity

- **Units:** Amperes

- **Example:** 1.8



### `current_std`

- **Type:** float

- **Description:** Standard deviation of current measurements

- **Physics:** Current variability; control quality indicator

- **Units:** Amperes

- **Example:** 0.15



### `temp_start`

- **Type:** float

- **Description:** Temperature at cycle start

- **Physics:** Initial thermal state

- **Units:** Celsius

- **Example:** 24.5



### `temp_end`

- **Type:** float

- **Description:** Temperature at cycle end

- **Physics:** Final thermal state after charge/discharge

- **Units:** Celsius

- **Example:** 27.8



### `temp_change`

- **Type:** float

- **Description:** Temperature rise during cycle (T_end - T_start)

- **Physics:** ★ KEY INDICATOR: electrochemical heating; drives aging

- **Units:** Celsius

- **Example:** 3.3



### `energy_proxy_VAs`

- **Type:** float

- **Description:** Energy throughput approximation: sum(V[i] * I[i] * dt[i])

- **Physics:** ★ KEY INDICATOR: total work done; coulombic throughput proxy

- **Units:** V·A·s (Joule equivalent)

- **Example:** 18500.2

- **Interpretation:** Cumulative energy enables cycle-to-cycle aging assessment



### `capacity`

- **Type:** float

- **Description:** Measured battery capacity at end of cycle

- **Physics:** ★ GOLD STANDARD: direct measure of state-of-health (SOH)

- **Units:** Amp-hours (Ah)

- **Example:** 1.85



### `capacity_ratio`

- **Type:** float

- **Description:** Normalized capacity: Capacity[i] / Capacity[0] (first cycle)

- **Physics:** ★ DEGRADATION METRIC: tracks SOH relative to brand new battery

- **Units:** dimensionless (0.0 to 1.0)

- **Example:** 0.95

- **Interpretation:** 0.95 = 95% of original capacity; 0.80 = 80% (end-of-life)




## 4. Feature Statistics


Summary statistics for all numeric features:


| Feature | Count | Mean | Std | Min | Max |

|---------|-------|------|-----|-----|-----|

| cycle_index | 2167 | 285.640 | 176.061 | 0.000 | 615.000 |

| n_points | 2167 | 964.029 | 1445.181 | 0.000 | 3900.000 |

| duration_s | 1280 | 6747.041 | 3696.237 | 12.656 | 10815.375 |

| voltage_start | 1280 | 3.883 | 0.421 | 0.236 | 8.393 |

| voltage_end | 1280 | 3.750 | 0.524 | 1.813 | 4.213 |

| voltage_delta | 1280 | -0.134 | 0.853 | -4.191 | 3.977 |

| voltage_slope_V_per_s | 1279 | 0.001 | 0.018 | -0.000 | 0.394 |

| voltage_std | 1279 | 0.178 | 0.125 | 0.016 | 2.276 |

| current_mean | 1279 | -0.605 | 1.230 | -2.000 | 1.007 |

| current_std | 1279 | 0.564 | 0.137 | 0.001 | 0.865 |


## 5. Key Physics Metrics


### capacity_ratio

State of Health indicator


**Range:** 0.0 to 1.0


**Interpretation:**

- 1.0: Brand new battery (first cycle)

- 0.95: Excellent health (~95% capacity retained)

- 0.90: Good health (95-100 cycles typical)

- 0.80: Degraded (near end-of-life threshold)

- 0.70: Critical (battery usually retired)



### voltage_slope_V_per_s

Voltage degradation rate during discharge


**Range:** Negative values, typically -0.0001 to -0.001


**Interpretation:**

- -0.00001: Excellent health, stable discharge profile

- -0.0001: Normal operation, new-to-mid-life

- -0.001: Degraded, increased internal resistance

- -0.01: Critical condition



### energy_proxy_VAs

Cumulative energy throughput


**Range:** Varies by cycle depth; typically 5000-50000 V·A·s


**Interpretation:**

- low: Shallow cycle (e.g., 10% depth)

- high: Deep cycle (e.g., 100% depth)



### temp_change

Temperature rise during cycle


**Range:** 0 to 20°C typical


**Interpretation:**

- 0-2°C: Low rate, cool operation

- 2-5°C: Normal operation

- 5-10°C: Elevated heating, accelerated aging

- >10°C: Excessive thermal stress




## 6. Usage Examples


### Load feature matrix in Python


```python

import pandas as pd
features = pd.read_csv('4_Results_and_Graphs/features_per_cycle.csv')

# Filter specific battery
b0005 = features[features['file'] == 'B0005.mat']

# Get capacity fade curve
import matplotlib.pyplot as plt
plt.plot(b0005['cycle_index'], b0005['capacity_ratio'])
plt.show()

```


### Simple RUL prediction using capacity ratio


```python

# Fit capacity fade curve
from scipy.optimize import curve_fit
import numpy as np

def decay_model(x, a, b):
    return a * np.exp(-b * x)

cycles = b0005['cycle_index'].values
capacity = b0005['capacity_ratio'].values

params, _ = curve_fit(decay_model, cycles, capacity)

# Predict when capacity reaches 0.80 (end-of-life)
rul_cycles = -np.log(0.80 / params[0]) / params[1]
print(f'Predicted end-of-life at cycle: {rul_cycles:.0f}')

```


### Analyze thermal effects on degradation


```python

# Correlate temp_change with capacity_fade
high_temp = features[features['temp_change'] > 5]
low_temp = features[features['temp_change'] <= 5]

print(f'High temp capacity fade rate: {high_temp["capacity_ratio"].mean():.3f}')
print(f'Low temp capacity fade rate: {low_temp["capacity_ratio"].mean():.3f}')

# High-temp batteries degrade faster

```



## 7. Visualizations


### State of Health (SOH) Trend (`B0005_capacity_ratio.png`)


**Physics Insight:**
Shows how battery capacity fades with cycling. 
Early flat region = calendar aging. 
Later steep decline = end of life approaching.
Capacity_ratio = 0.80 is typical end-of-life threshold.


**What to Look For:**

- Linear degradation vs non-linear

- Inflection points indicating phase changes

- Comparison between different battery cells



### Voltage Degradation Rate (`B0005_voltage_slope.png`)


**Physics Insight:**
Voltage slope becomes more negative as battery ages.
This indicates reduced ability to maintain voltage during discharge.
Early cycles: stable voltage slope. 
Later cycles: increasingly negative = rapid degradation.


**What to Look For:**

- Transition from stable to degrading region

- Exponential vs linear change in slope

- Correlation with capacity fade



### State of Health (SOH) Trend (`B0006_capacity_ratio.png`)


**Physics Insight:**
Shows how battery capacity fades with cycling. 
Early flat region = calendar aging. 
Later steep decline = end of life approaching.
Capacity_ratio = 0.80 is typical end-of-life threshold.


**What to Look For:**

- Linear degradation vs non-linear

- Inflection points indicating phase changes

- Comparison between different battery cells



### Voltage Degradation Rate (`B0006_voltage_slope.png`)


**Physics Insight:**
Voltage slope becomes more negative as battery ages.
This indicates reduced ability to maintain voltage during discharge.
Early cycles: stable voltage slope. 
Later cycles: increasingly negative = rapid degradation.


**What to Look For:**

- Transition from stable to degrading region

- Exponential vs linear change in slope

- Correlation with capacity fade



### State of Health (SOH) Trend (`B0007_capacity_ratio.png`)


**Physics Insight:**
Shows how battery capacity fades with cycling. 
Early flat region = calendar aging. 
Later steep decline = end of life approaching.
Capacity_ratio = 0.80 is typical end-of-life threshold.


**What to Look For:**

- Linear degradation vs non-linear

- Inflection points indicating phase changes

- Comparison between different battery cells



### Voltage Degradation Rate (`B0007_voltage_slope.png`)


**Physics Insight:**
Voltage slope becomes more negative as battery ages.
This indicates reduced ability to maintain voltage during discharge.
Early cycles: stable voltage slope. 
Later cycles: increasingly negative = rapid degradation.


**What to Look For:**

- Transition from stable to degrading region

- Exponential vs linear change in slope

- Correlation with capacity fade



### State of Health (SOH) Trend (`B0018_capacity_ratio.png`)


**Physics Insight:**
Shows how battery capacity fades with cycling. 
Early flat region = calendar aging. 
Later steep decline = end of life approaching.
Capacity_ratio = 0.80 is typical end-of-life threshold.


**What to Look For:**

- Linear degradation vs non-linear

- Inflection points indicating phase changes

- Comparison between different battery cells



### Voltage Degradation Rate (`B0018_voltage_slope.png`)


**Physics Insight:**
Voltage slope becomes more negative as battery ages.
This indicates reduced ability to maintain voltage during discharge.
Early cycles: stable voltage slope. 
Later cycles: increasingly negative = rapid degradation.


**What to Look For:**

- Transition from stable to degrading region

- Exponential vs linear change in slope

- Correlation with capacity fade




---

*Documentation auto-generated by BatteryPipelineDocumenter*
