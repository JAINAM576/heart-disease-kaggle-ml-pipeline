
from lightgbm import LGBMClassifier
from src.dataloader import  preprocess
from src.models.trainers import train_any_model_with_fold
from src.models.stacking import stack_models
from src.mlflowhandler import log_experiment
from configs.config import LGB_PARAMS
from pathlib import Path


PROJECT_ROOT = Path("..").resolve()
TRAIN_PATH= PROJECT_ROOT / "data" / "raw" / "train.csv"
TEST_PATH= PROJECT_ROOT / "data" / "raw" / "test.csv"
submission_path= PROJECT_ROOT / "data" / "raw" / "sample_submission.csv"

def main():

    X, y, X_test = preprocess(TRAIN_PATH, TEST_PATH)

    oof,test_preds, mean_auc, mean_metrics, models = train_any_model_with_fold(X, y, X_test,model_class=LGBMClassifier, PARAMS=LGB_PARAMS)

    log_experiment("LightGBM", mean_auc, mean_metrics,LGB_PARAMS)

    # If stacking multiple models:
    # stacked_test, stack_auc = stack_models([lgb_oof], [lgb_test], y)

if __name__ == "__main__":
    # main()
    pass

