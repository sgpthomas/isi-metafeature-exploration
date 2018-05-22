#/usr/share/python3

from pmlb import fetch_data, dataset_names
from progressbar import ProgressBar, Percentage, Bar, ETA

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

adult_X, adult_labels = fetch_data('adult', return_X_y=True)
adult_Xdf = pd.DataFrame(adult_X)

### PCA
from sklearn.decomposition import PCA

# plt.show()

def cca_fig(name, data):
    pca = PCA(n_components=2)
    trans_X = pd.DataFrame(pca.fit_transform(data))
    sb.regplot(x=trans_X[0], y=trans_X[1], fit_reg=False)
    plt.savefig(name + ".png", dpi=400)
    plt.clf()

names = dataset_names
pbar = ProgressBar(widgets=[Percentage(),
                            Bar(right="| "),
                            ETA()],
                   maxval=len(names)).start()
for i, n in enumerate(names):
    X, y = fetch_data(n, return_X_y=True)
    Xdf = pd.DataFrame(X)
    cca_fig(n, Xdf)
    pbar.update(i)
pbar.finish()

### CCA
# from sklearn.cross_decomposition import CCA

# cca = CCA(n_components=2)

# sb.regplot(x=adult_X[0], y=adult_X[2], fit_reg=False)
# plt.show()
