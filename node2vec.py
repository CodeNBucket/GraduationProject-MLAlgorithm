import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing

import numpy as np

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph

from gensim.models import Word2Vec

import warnings
import collections
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt

metadata = pd.read_csv('data/metadata.txt', header=None, delim_whitespace=True)
metadata.columns = ['Date', 'ReviewId', 'ReviewerId', 'ProductId', 'Label', 'Useful', 'Funny', 'Cool', 'Rating']

edgeData = metadata[['ReviewerId', 'ProductId', 'Rating']]
edgeData.columns = ['source', 'target', 'weight']
reviewerNodes = pd.DataFrame(index=metadata['ReviewerId'].unique())
productNodes = pd.DataFrame(index=metadata['ProductId'].unique())

graph = StellarGraph({"reviewer": reviewerNodes, "product": productNodes}, {"rating": edgeData})

labelReviewer = metadata[['ReviewerId', 'Label']]
labelReviewer = labelReviewer.drop_duplicates(subset=['ReviewerId'])
labelReviewer.set_index('ReviewerId', inplace=True)

labelProduct = metadata[['ProductId', 'Label']]
labelProduct = labelProduct.drop_duplicates(subset=['ProductId'])
labelProduct.set_index('ProductId', inplace=True)

frames = [labelReviewer, labelProduct]
labelData = pd.concat(frames)
labelData = labelData.squeeze()

rw = BiasedRandomWalk(graph)
weighted_walks = rw.run(
    nodes=list(graph.nodes()),
    length=10,
    n=10,
    p=1.0,
    q=1.0,
    weighted=True,
    seed=42,
)

weighted_model = Word2Vec(
    weighted_walks, vector_size=384, window=10
)

node_ids = weighted_model.wv.index_to_key
weighted_node_embeddings = (
    weighted_model.wv.vectors
)
node_targets = labelData.loc[node_ids].astype("category")

X = weighted_node_embeddings
y = np.array(node_targets)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, test_size=None, random_state=42
)

clf = LogisticRegressionCV(
    Cs=10,
    cv=10,
    tol=0.001,
    max_iter=1000,
    scoring="roc_auc",
    verbose=False,
    multi_class="ovr",
    random_state=5434,
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
