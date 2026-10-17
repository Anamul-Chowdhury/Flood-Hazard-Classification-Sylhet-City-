# Flood Hazard Classification (Sylhet City)

Paper: https://ieeexplore.ieee.org/document/11491386/

Reproduces the methodology of the ICCIT 2025 paper "Flood Hazard Classification
Using Machine Learning and a Proposed Early-Warning System for Risk Management
in Sylhet City, Bangladesh".

## Dataset

`sylhet_climate_2013_2024.csv` in the same folder. Columns needed:
RH_Avg, RH_Max, RH_Min, AT_Avg, AT_Max, AT_Min,
SR_Avg, SR_Max, SR_Min, RF_Avg, RF_Max, RF_Min,
Flood_Risk_Status (values: Minimal, Minor, Significant, Severe).

## What each cell does

1. Preprocessing: drops Year/Month, seasonal-pattern imputation then MICE with
   Random Forest, domain constraints (rainfall >= 0, humidity <= 100), duplicate
   removal, z-score standardization. Saves `flood_processed.csv`.
2. Training: Random Forest (200 trees, depth 12, balanced weights),
   Gradient Boosting (200 rounds, lr 0.1, depth 6), AdaBoost (200 learners,
   lr 0.1), Decision Tree (depth 12, balanced weights). No resampling is used;
   class imbalance is handled with class weights. Reports accuracy, precision,
   recall, F1 (macro) and saves `flood_model_results.csv`.
3. Analysis of the best model: class-wise report, normalized confusion matrix
   (`flood_confusion_matrix.png`), SHAP beeswarm plot (`flood_shap.png`).

## Requirements

pandas, numpy, scikit-learn, shap, matplotlib, seaborn.

## Expected outcome

Random Forest should be the top model, matching the paper (accuracy ~99.31%,
maximum and average rainfall as top SHAP drivers).
