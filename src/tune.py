"""Tune Logistic Regression Hyperparameter with Cross Validation
"""

import pandas as pd
import numpy as np
import json

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    train = pd.read_csv('data/processed/training.csv')
    X_train, y_train = train.drop(['condition'], axis=1), train['condition']
    test = pd.read_csv('data/processed/test.csv')
    X_test, y_test = test.drop(['condition'], axis=1), test['condition']
    lr = Pipeline([
        ('scale', StandardScaler()),
        ('lr', LogisticRegression())
    ])
    param_grid = {'lr__C': np.logspace(-3, 3, 20)}
    tuned_lr = GridSearchCV(lr, return_train_score=True, param_grid=param_grid, scoring='accuracy', cv=10)
    tuned_lr.fit(X_train, y_train)
    with open('results/tuned_hyperparameters.json', 'w') as outfile:
        json.dump(tuned_lr.best_params_, outfile)