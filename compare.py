#/usr/share/python3

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import seaborn as sb
import pandas as pd

from pmlb import fetch_data, classification_dataset_names

from os.path import exists, join
from os import makedirs

def cca_fig(method, name, data):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    path = join('figs', method)
    if not exists(path):
        makedirs(path)

    pca = PCA(n_components=2)
    trans_X = pd.DataFrame(pca.fit_transform(data))
    sb.regplot(x=trans_X[0], y=trans_X[1], fit_reg=False)
    plt.savefig(join(path, name) + ".png", dpi=400)
    plt.clf()

def compare_methods(names, m1, m2, m1_name="model1", m2_name="model2"):
    results = {m1_name: [], m2_name: []}
    row_names = {}
    for i, dataset in enumerate(names):
        row_names[i] = dataset

        X, y = fetch_data(dataset, return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y)

        m1.fit(train_X, train_y)
        m2.fit(train_X, train_y)

        m1_score = m1.score(test_X, test_y)
        m2_score = m2.score(test_X, test_y)
        print("m1: {}, m2: {}".format(m1_score, m2_score))
        if m1_score > m2_score:
            results[m1_name].append(1)
            results[m2_name].append(0)
            cca_fig(m1_name, dataset, pd.DataFrame(X))
        else:
            results[m1_name].append(0)
            results[m2_name].append(1)
            cca_fig(m2_name, dataset, pd.DataFrame(X))

    df = pd.DataFrame(results)
    return df.rename(index=row_names)

results = compare_methods(classification_dataset_names[:10],
                          LogisticRegression(),
                          GradientBoostingClassifier(),
                          m1_name="Logistic Regression",
                          m2_name="Gradient Boosting")


# print("M1 = {}; M2 = {}".format("Logistic Regression", "Gradient Boosting"))
# print("M1: ", _scores)
# print("M2: ", m2_test_scores)

# import matplotlib.pyplot as plt
# sb.boxplot(data=[logit_test_scores, gnb_test_scores], notch=True)
# plt.xticks([0, 1], ['LogisticRegression', 'GaussianNB'])
# plt.ylabel('Test Accuracy')
# plt.show()
