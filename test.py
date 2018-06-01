import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from util import *

# scores a model on the data [X y]
def score_model(X, y, model):
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    model.fit(train_X, train_y.values.ravel()) # train the model
    return model.score(test_X, test_y)

def score_model_across_params(X, y, model, params, n_times=10):
    res = {}
    for p in product_dict(params):
        print(p)
        m = model(**p)
        res[str(p)] = average(lambda: score_model(X, y, model(**p)), n_times=n_times)
    return res

from sklearn.ensemble import GradientBoostingClassifier
def get_gradient_boosting_classifier():
    params = {
        'n_estimators': [10, 50, 100, 500],
        'min_impurity_decrease': np.arange(0., 0.005, 0.00025),
        'max_features': [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
        'learning_rate': [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0],
        'loss': ['deviance'],
        'random_state': [324089]
    }
    return (GradientBoostingClassifier, params)


