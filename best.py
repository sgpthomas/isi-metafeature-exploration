import numpy as np
import seaborn as sb
import pandas as pd

from util import *
from math import isnan


def table_subset_row(data, col, val):
    return data.loc[data['dataset'] == 'adult']

def max_row(data, col):
    return data.loc[data[col].idxmax()]

# res = {}
# for name in dataset_names:
#     sub = table_subset_row(data, 'dataset', name)
#     res[name] = max_row(sub, 'accuracy').classifier

def best_classifiers(data):
    col = 'accuracy'
    data = data.groupby(['dataset','classifier'])[col].max().reset_index()
    data[col] = data[col].apply(lambda x: round(x, 3))

    dataset_best_models = {}
    for name, group_data in data.groupby('dataset'):
        norm_accuracy = group_data.copy()
        norm_accuracy[col] = group_data[col] / group_data[col].max()
        dataset_best_models[name] = norm_accuracy.loc[
            norm_accuracy[col] >= 0.99, 'classifier'
        ].values
    return dataset_best_models

columns = ['dataset', 'classifier', 'parameters', 'accuracy', 'macrof1', 'bal_accuracy']

data = pd.read_csv('sklearn-benchmark5-data-edited.tsv.gz',
                   sep='\t',
                   names=columns).fillna('')

metafeatures = pd.read_pickle('sgt43_metafeatures.pkl')

dataset_names = data['dataset'].unique()

X = filter_dataframe_by_column(lambda x: isnan(x), metafeatures)
y = pd.DataFrame(list((map_dict(lambda v: v[0], best_classifiers(data))).values()))


