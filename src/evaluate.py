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
    prediction_classes = lr.predict(X_test)
    prediction_probs = lr.predict_proba(X_test)[:, 1]
    accuracy = metrics.accuracy_score(y_test, prediction_classes)
    roc_auc = metrics.roc_auc_score(y_test, prediction_probs)
    results_scores = {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    with open('results/scores.json', 'w') as f:
        json.dump(results_scores, f, indent=4)
    precision, recall, prc_thresholds = metrics.precision_recall_curve(y_test, prediction_probs)
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, prediction_probs)
    prob_true, prob_pred = calibration_curve(y_test, prediction_probs, n_bins=10)
    with open('results/pr_curve.json', 'w') as f:
        json.dump(
            {
                "pr": [
                    {'precision': p, 'recall': r, 'threshold': t}
                    for p, r, t in zip(precision, recall, prc_thresholds)
                ]
            },
            f,
            indent=4
        )
    with open('results/roc_curve.json', 'w') as f:
        json.dump(
            {
                "roc": [
                    {'fpr': a, 'tpr': b, 'threshold': c}
                    for a, b, c in zip(fpr, tpr, roc_thresholds)
                ]
            },
            f,
            indent=4
        )
    with open('results/calibration_curve.json', 'w') as f:
        json.dump(
            {
                "calibration": [
                    {'prob_true': a, 'prob_predicted': b}
                    for a, b in zip(prob_true, prob_pred)
                ]
            },
            f,
            indent=4
        )
