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
-Done
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

```
heart-disease-kaggle-ml-pipeline/
│
├── configs/                          # Configuration & bootstrap
│   ├── bootstrap.py                  # Project bootstrapping utilities
│   └── config.py                     # Global configuration settings
│
├── data/                             # Data directory (raw → processed)
│   ├── raw/                          # Original competition data
│   │   ├── train.csv                 # Training set (630k rows)
│   │   ├── test.csv                  # Test set
│   │   └── sample_submission.csv     # Submission format reference
│
├── notebooks/                        # Jupyter notebooks
│   ├── EDA.ipynb                     # Exploratory Data Analysis
│   ├── data_process.ipynb            # Data processing & cleaning
│   ├── model_trainning.ipynb         # Model training & evaluation
│   └── model_mlflow_pipeline.ipynb   # MLflow experiment pipeline
│
├── src/                              # Source code modules
│   ├── __init__.py                   # Package initializer
│   ├── dataloader.py                 # Data loading utilities
│   ├── features.py                   # Feature engineering functions
│   ├── preprocess.py                 # Preprocessing pipeline
│   ├── submission_formatter.py       # Kaggle submission formatting
│   ├── utils.py                      # General utility functions
│   └── models/                       # Model definitions
│       ├── trainers.py               # Model training logic
│       ├── stacking.py               # Stacking ensemble builder
│       ├── linear_stacking.py        # Linear stacking strategy
│       └── combine_models.py         # Model combination utilities
│
├── mlflow_utils/                     # MLflow tracking utilities
│   ├── mlflowhandler.py              # Remote MLflow handler
│   ├── mlflowhandler_local.py        # Local MLflow handler
│   └── mlflowregister.py             # Model registry utilities
│
├── app.py                            # Application entry point
├── main.py                           # Main execution script
├── create_kaggle_template.py         # Kaggle notebook template generator
├── problem_research.py               # Dataset research & Bayes error analysis
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation
```

---

## 🏆 Best Score

| Detail              | Value                              |
|----------------------|------------------------------------|
| **Competition Score** | **0.95368** (ROC-AUC)             |
| **Model Type**       | StackingClassifier                 |
| **Base Estimators**  | CatBoost ×3 + XGBoost ×3          |
| **Final Estimator**  | RidgeClassifier                    |
| **Dataset**          | Heart Disease (Kaggle Synthetic)   |
| **Highlight**        | 🥇 Highest Accuracy Model          |

> **Note:** This score was achieved using a **StackingClassifier** ensemble with three CatBoost and three XGBoost models as base estimators, combined through a **RidgeClassifier** meta-learner. 

---

## 🚀 Streamlit Deployment

The project includes a **Streamlit** web app (`app.py`) for real-time heart disease prediction using the deployed MLflow stacking model.

### Prerequisites

| Dependency       | Purpose                                  |
|------------------|------------------------------------------|
| `streamlit`      | Web app framework                        |
| `pandas`         | Data manipulation                        |
| `numpy`          | Numerical computations                   |
| `mlflow`         | Model loading from MLflow registry       |
| `scikit-learn`   | ML utilities (required by the model)     |
| `python-dotenv`  | Load environment variables from `.env`   |
| `catboost`       | CatBoost base estimator (model dependency) |
| `xgboost`        | XGBoost base estimator (model dependency)  |

### Setup & Run

1. **Install dependencies**
   ```bash
   pip install -r streamlit_deploy_requirements.txt
   ```

2. **Configure environment variables** — Create a `.env` file in the project root:
   ```env
   MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
   MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Access** — Open [http://localhost:8501](http://localhost:8501) in your browser.

### How It Works

- Loads the **production** stacking model from the MLflow registry on [DagsHub](https://dagshub.com/JAINAM576/heart-disease-kaggle-ml-pipeline.mlflow)
- Accepts 13 clinical features via an interactive form
- Uses a calibrated **decision threshold (0.4574)** from PR curve analysis
- Outputs risk probability and heart disease prediction

---
