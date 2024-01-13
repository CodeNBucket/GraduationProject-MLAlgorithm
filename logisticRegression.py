import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

metadata = pd.read_csv(
    'data/metadata.txt',
    header=None,
    delim_whitespace=True
)

metadata.columns = [
    'Date', 'ReviewId', 'ReviewerId',
    'ProductId', 'Label', 'Useful',
    'Funny', 'Cool', 'Rating'
]

# nodeEmbeddings = KeyedVectors.load_word2vec_format(
#     'models/node2vec.wordvectors',
#     binary=True
# )
# nodeVectors = [None] * len(metadata.axes[0])
# nodeTargets = [None] * len(metadata.axes[0])
# 
# for i in metadata.index:
#     nodeVectors[i] = np.append(
#         nodeEmbeddings.get_vector(metadata['ReviewerId'][i]),
#         nodeEmbeddings.get_vector(metadata['ProductId'][i])
#     )
#     nodeTargets[i] = metadata['Label'][i]
# 
# nodeVectors = np.array(nodeVectors)
# nodeTargets = np.array(nodeTargets)
# 
# X_train, X_test, y_train, y_test = train_test_split(
#     nodeVectors,
#     nodeTargets,
#     train_size=0.95,
# )
# 
# clf = LogisticRegressionCV(max_iter=8000)
# clf.fit(X_train, y_train)
# 
# clf_probs = clf.predict_proba(X_test)
# clf_probs = clf_probs[:, 1]
# clf_auc = roc_auc_score(y_test, clf_probs)
# print(f'AUC-ROC-SCORE: {clf_auc}')
