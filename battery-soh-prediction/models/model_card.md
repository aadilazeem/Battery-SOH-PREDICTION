# BATTERY SOH PREDICTION MODEL - MODEL CARD

## Model Overview
- **Name:** Stacking Ensemble with GB Meta-Learner
- **Type:** Regression (Battery State of Health prediction)
- **Base Learners:** RandomForest, GradientBoosting, XGBoost, SVR
- **Meta-Learner:** GradientBoosting

## Performance
- **R² Score:** 0.843 (84.3% variance explained)
- **MAE:** 0.0341 (3.41% absolute error)
- **RMSE:** 0.0381
- **Cross-Validation:** 5-Fold, Stable (Std < 0.008)

## Input Features (5 Required)
1. energy_proxy_VAs (Energy integral, V·A·s)
2. duration_s (Cycle duration, seconds)
3. discharge_curvature_coeff (Voltage curve shape)
4. voltage_start (Initial voltage, V)
5. current_mean (Average discharge current, A)

## Output
- **Capacity_Ratio:** 0.0-1.0 (fraction of original capacity)
- **Confidence Interval:** 95% CI automatically calculated

## Known Limitations
- ⚠️ Data quality: 40% missing electrochemical features
- ⚠️ Outliers: 8.2% extreme values present
- ⚠️ Distribution: Right-skewed (not normal)
- ⚠️ Designed for: Li-ion batteries (similar chemistry)

## Trained Data
- **Samples:** 636 capacity measurements
- **Cells:** 4 battery files
- **Cycles:** 0-615 per cell
- **Chemistry:** 18650-type Li-ion cells

## Deployment
- **Framework:** scikit-learn
- **Python Version:** 3.8+
- **Dependencies:** pandas, numpy, scikit-learn
- **Inference Time:** <10ms per prediction
- **Memory:** ~2MB model + 500KB scaler

## Maintenance
- **Retraining:** Every 6 months with new data
- **Monitoring:** Track prediction error distribution
- **Drift Detection:** Alert if MAE increases >5%