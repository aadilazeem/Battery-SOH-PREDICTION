# Battery State-of-Health (SOH) Prediction: Project Report

## 1. Introduction
This project aims to develop a robust, interpretable machine learning pipeline for predicting the State-of-Health (SOH) of lithium-ion batteries using the NASA battery dataset. The workflow is designed to be transparent, physics-informed, and reproducible, providing actionable insights for battery research and management.

## 2. Data Preparation and Exploration
- **Data Source:** NASA battery datasets (MAT files for multiple cells/cycles).
- **Initial Steps:**
    - Loaded raw MAT files and extracted per-cycle measurement tables (Voltage, Current, Temperature, Time).
    - Performed exploratory data analysis (EDA) to check for missing values, outliers, and anomalies using both IQR and z-score methods.
    - Visualized the first cycle to understand data structure and flagged outliers.
    - Documented recommendations for handling outliers and segmenting charge/discharge events.

## 3. Feature Engineering
- **Physics-Informed Features:**
    - Created features such as cycle duration, voltage start/end/delta, voltage slope, current mean/std, temperature change, energy proxy, discharge curve shape, and capacity ratio.
    - Each feature was documented with its physical meaning and relevance to battery degradation mechanisms (e.g., SEI growth, LLI, LAM, thermal effects).
    - Saved the feature matrix for downstream ML modeling.

## 4. Exploratory Data Analysis (EDA)
- **Trend Analysis:**
    - Visualized key features (capacity ratio, energy proxy, discharge voltage slope) over cycles.
    - Used histograms, boxplots, and correlation analysis to identify trends and relationships.
    - Noted that capacity ratio declines over cycles, and voltage slope/energy proxy are strong indicators of SOH decline.

## 5. Machine Learning Modeling
- **Model Selection:**
    - Compared several models: Random Forest, Decision Tree, Gradient Boosting, Stacking, and Support Vector Machine (SVM).
    - Used leave-one-file-out cross-validation to ensure robust evaluation and avoid data leakage.
    - SVM with RBF kernel was selected as the best-performing model after hyperparameter tuning.

## 6. Model Interpretation
- **Feature Importance:**
    - Used permutation importance to identify the most influential features for SVM predictions.
    - Top features: energy processed per cycle, cycle duration, discharge curvature, starting voltage, and mean current—all physically meaningful.
- **Visualization:**
    - Plotted SVM predictions vs. true SOH for all batteries.
    - Bar chart of top 10 feature importances.

## 7. Reporting and Export
- **Reproducibility:**
    - Exported SVM predictions and top features as CSV files for reporting and further analysis.
    - Provided clear documentation and summary for supervisors and collaborators.

## 8. Conclusion
- The developed pipeline accurately predicts battery SOH using interpretable, physics-based features.
- SVM with RBF kernel generalizes well across different battery cells, with low error and high R².
- The most important features align with known battery degradation mechanisms, increasing confidence in the model's physical relevance.
- This approach is robust, transparent, and can be extended to other battery datasets or used to guide experimental research.

**Final Takeaway:**
A transparent, physics-informed ML pipeline can provide actionable and interpretable SOH predictions, supporting both research and practical battery management. The workflow and results are ready for reporting, presentation, or further extension.
