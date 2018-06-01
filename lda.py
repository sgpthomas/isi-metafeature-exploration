#/usr/bin/env python3

from pmlb import fetch_data

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from util import *

def graph_lda(X, y, subplot="111"):
    ax = plt.subplot(subplot)
    trans_X = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
    unique = unique_values(y)
    unique.sort()
    for i in unique:
        ax.scatter(trans_X[y == i, 0], trans_X[y == i, 1], label=str(i), lw=0.1)
    return ax

def wrapper(name, save=False, loc=None):
    X, y = fetch_data(name, return_X_y=True)
    y_rand = randomize_labels(y)

    norm = graph_lda(X, y, subplot="121")
    norm.set_title("Real Labels")
    rand = graph_lda(X, y_rand, subplot="122")
    rand.set_title("Random Labels")

    plt.suptitle("LDA of {}".format(name))

    if save:
        filename = "{}.png".format(name)
        if loc != None:
            filename = "{}/{}".format(loc, filename)
        plt.savefig(filename, dpi=500)
        plt.clf()
    else:
        plt.show()
