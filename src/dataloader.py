
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(".."))
from configs.config import TARGET, ID_COL


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    return train, test


def prepare_target(train):
    train[TARGET] = train[TARGET].map({
        "Presence": 1,
        "Absence": 0
    })

    print("Class balance:\n", train[TARGET].value_counts(normalize=True))
    return train


def split_features(train, test):
    X = train.drop([TARGET, ID_COL], axis=1)
    y = train[TARGET]
    X_test = test.drop([ID_COL], axis=1)

    return X, y, X_test

def preprocess(TRAIN_PATH, TEST_PATH):
    train, test = load_data(TRAIN_PATH, TEST_PATH)
    train = prepare_target(train)
    X, y, X_test = split_features(train, test)

    return X, y, X_test