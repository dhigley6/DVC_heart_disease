"""Train Logistic Regression model with tuned hyperparameters
"""

import json
import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    with open('results/tuned_hyperparameters.json') as json_file:
        hyperparams = json.load(json_file)
    train = pd.read_csv('data/processed/training.csv')
    X_train, y_train = train.drop(['condition'], axis=1), train['condition']
    lr = Pipeline([
        ('scale', StandardScaler()),
        ('lr', LogisticRegression())
    ])
    lr.set_params(**hyperparams)
    lr.fit(X_train, y_train)
    with open('models/logistic_regression.pickle', 'wb') as f:
        pickle.dump(lr, f)
