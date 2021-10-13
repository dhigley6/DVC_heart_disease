"""Evaluate trained model
"""

import json
import pickle
import pandas as pd

from sklearn import metrics
from sklearn.calibration import calibration_curve

if __name__ == "__main__":
    with open('models/logistic_regression.pickle', 'rb') as f:
        lr = pickle.load(f)
    test = pd.read_csv('data/processed/test.csv')
    X_test, y_test = test.drop(['condition'], axis=1), test['condition']
    prediction_probs = lr.predict_proba(X_test)
    precision, recall, prc_thresholds = metrics.precision_recall_curve(y_test, prediction_probs)
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, prediction_probs)
    prob_true, prob_pred = calibration_curve(y_test, prediction_probs)

    with open()