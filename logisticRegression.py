import joblib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix,accuracy_score

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

nodeEmbeddings = KeyedVectors.load_word2vec_format(
    'models/node2vec.wordvectors',
    binary=True
)

docEmbeddings = KeyedVectors.load_word2vec_format(
    'models/doc2vec.wordvectors',
    binary=True
)

nodeVectors = [None] * len(metadata.axes[0])
nodeTargets = [None] * len(metadata.axes[0])

for i in metadata.index:
    nodeVectors[i] = np.append(
        nodeEmbeddings.get_vector(metadata['ReviewerId'][i]),
        nodeEmbeddings.get_vector(metadata['ProductId'][i])
    )
    nodeVectors[i] = np.append(
        nodeVectors[i],
        docEmbeddings[metadata['ReviewId'][i]]
    )
    nodeTargets[i] = metadata['Label'][i]


nodeVectors = np.array(nodeVectors)
nodeTargets = np.array(nodeTargets)

scale = StandardScaler()

model = LogisticRegressionCV(
    class_weight={'Y': 0.13, 'N': 0.87},
    max_iter=8000
)

nodeVectors = scale.fit_transform(nodeVectors)

model.fit(nodeVectors, nodeTargets)

joblib.dump(model, 'models/LRM.sav')

X_train, X_test, y_train, y_test = train_test_split(
    nodeVectors,
    nodeTargets,
    train_size=0.80,
)

X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

model.fit(X_train, y_train)

y_pred_prob_test = model.predict_proba(X_test)[:, 1]
y_pred_test = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_test)
print(y_pred_test)

print("confusion Matrix is :\n\n", cm)
print("\n")
print("ROC-AUC score  test dataset: ",
      roc_auc_score(y_test,
                    y_pred_prob_test))
print("Precision score  test dataset: ",
      precision_score(y_test,
                      y_pred_test,
                      pos_label='N'))
print("Recall score  test dataset: ",
      recall_score(y_test,
                   y_pred_test,
                   pos_label='N'))
print("F1 score  test dataset : ",
      f1_score(y_test,
               y_pred_test,
               pos_label='N'))
print("Accuracy",accuracy_score(y_test,y_pred_test))
# clf = LogisticRegressionCV(max_iter=8000)
# clf.fit(X_train, y_train)
# 
# clf_probs = clf.predict_proba(X_test)
# clf_probs = clf_probs[:, 1]
# clf_auc = roc_auc_score(y_test, clf_probs)
# print(f'AUC-ROC-SCORE: {clf_auc}')
