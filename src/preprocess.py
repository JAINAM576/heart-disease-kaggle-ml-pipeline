import numpy as np

def cap_outliers(df):
    df["Cholesterol"] = np.clip(df["Cholesterol"], 126, 450)
    return df


def log_transform(df):
    df["ST depression"] = np.log1p(df["ST depression"])
    return df


def preprocess(df):
    # df = cap_outliers(df)
    # df = log_transform(df)
    return df