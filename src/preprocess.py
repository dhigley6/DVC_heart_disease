"""Preprocess raw data for modeling
"""

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = pd.read_csv('data/raw/heart_cleveland_upload.csv')
    train, test = train_test_split(data, test_size=0.3)
    train.to_csv('data/processed/training.csv')
    test.to_csv('data/processed/test.csv')
    