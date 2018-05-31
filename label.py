#/usr/share/python3

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sb
import pandas as pd

from pmlb import fetch_data, classification_dataset_names

import util

# from os.path import exists, join
# from os import makedirs

# scores a model on the data [X y]
def score_model(X, y, model):
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    model.fit(train_X, train_y) # train the model
    return model.score(test_X, test_y)

# returns dict of scores (keyed by names) after running each model on the provided data
@util.timeout(180)
def compare(X, y, model_list, model_names, n_times=10):
    total = {}
    for i, m in enumerate(model_list):
        print("  Tring model {}: ".format(i), end="", flush=True)
        results = []
        for t in range(n_times):
            results.append(score_model(X, y, m()))
        mean = np.mean(results)
        print(mean)
        total[model_names[i]] = [mean]
    return total

def main():
    ds_names = classification_dataset_names
    models = [LogisticRegression, GradientBoostingClassifier]
    model_names = ["LogisticRegression", "GradientBoosting"]
    results = {}
    for i, n in enumerate(ds_names):
        try:
            print("Iteration: {}/{} '{}'".format(i+1, len(ds_names), n))
            X, y = fetch_data(n, return_X_y=True)
            results = util.merge_dicts(results,
                                       compare(X, y, models, model_names)) # updates results
            pd.DataFrame(results).to_pickle('labels.pkl')
        except util.TimeoutError:
            print("Timed Out!")
    print("Done!")
    df = pd.DataFrame(results)
    df = df.rename(index=util.list_to_idx_dict(ds_names))
    df.to_pickle("labels.pkl")

if __name__ == "__main__":
    main()
