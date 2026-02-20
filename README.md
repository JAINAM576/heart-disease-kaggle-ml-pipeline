# ❤️ Heart Disease Prediction – Kaggle ML Pipeline

Advanced machine learning pipeline for predicting heart disease using gradient boosting models, stacking ensembles, and experiment tracking with MLflow.

Built for large-scale synthetic dataset (630,000 samples) derived from the UCI Heart Disease dataset.

---

## 📌 Competition Overview

- **Task**: Binary classification – Predict Heart Disease
- **Train Size**: 630,000 rows
- **Features**: 14 clinical features
- **Evaluation Metric**: ROC-AUC (primary), Accuracy
- **Dataset Source**: Synthetic data generated from UCI Heart Disease dataset

---

## 🧠 Modeling Strategy

### 1️⃣ Base Models
- LightGBM
- XGBoost
- CatBoost

### 2️⃣ Cross Validation
- Stratified K-Fold
- Repeated Stratified K-Fold

### 3️⃣ Feature Engineering
-remaining 
### 4️⃣ Stacking Architecture
- Out-of-Fold predictions
- Meta-model (Logistic Regression / LightGBM)
- Test matrix blending

### 5️⃣ Experiment Tracking
- MLflow logging
- Metrics logging
- Confusion matrix
- ROC curve
- Classification report
- Threshold optimization (F1)

---


(Results from cross-validation)

---

## 📂 Project Structure

