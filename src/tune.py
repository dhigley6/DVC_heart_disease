"""Tune Logistic Regression Hyperparameter with Cross Validation
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')
    lr = 