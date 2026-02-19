RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "Heart Disease"
ID_COL = "id"

MLFLOW_EXPERIMENT = "heart_disease_lgbm"

# GOT USING OPTUNA 
LGB_PARAMS = {
    "n_estimators": 1600,
    "learning_rate": 0.04689367370315809,
    "num_leaves": 49,
    "subsample": 0.7223117664101711,
    "colsample_bytree": 0.8352262295510223,
    "reg_alpha": 2.248822478298761,
    "reg_lambda": 0.17728212280528755,
    "min_child_samples": 32,
    "max_depth": 4,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

LOGISTIC_PARAMS = {
    "max_iter": 1000,
    "random_state": 42
}

LINEAR_PARAMS_RIDGE={
    "alpha": 1.0
}