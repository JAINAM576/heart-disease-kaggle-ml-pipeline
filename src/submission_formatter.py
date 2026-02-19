import pandas as pd 
from pathlib import Path


def formatter(test_preds,name):
    print("Formatting submission...")
    print("Test predictions shape:", test_preds.shape)

    PROJECT_ROOT = Path("..").resolve()
    submission_path= PROJECT_ROOT / "data" / "raw" / "sample_submission.csv"
    submission_df = pd.read_csv(submission_path) 

    submission_df["Heart Disease"] = test_preds
    submission_df.to_csv(name, index=False)

    print("Submission saved:", name)