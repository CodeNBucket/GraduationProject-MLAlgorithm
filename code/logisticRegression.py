import joblib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

metadata = pd.read_csv("../data/metadata.txt", header=None, sep="\\s+")

metadata.columns = [
    "Date",
    "ReviewId",
    "ReviewerId",
    "ProductId",
    "Label",
    "Useful",
    "Funny",
    "Cool",
    "Rating",
]

nodeEmbeddings = KeyedVectors.load_word2vec_format(
    "../models/node2vec.wordvectors", binary=True
)

docEmbeddings = KeyedVectors.load_word2vec_format(
    "../models/doc2vec.wordvectors", binary=True
)

nodeVectors = [None] * len(metadata.axes[0])
nodeTargets = [None] * len(metadata.axes[0])

nodeVectors = docEmbeddings[metadata["ReviewId"]]
nodeTargets = metadata["Label"]

# for i in metadata.index:
# nodeVectors[i] = np.append(
#     nodeEmbeddings.get_vector(metadata["ReviewerId"][i]),
#     nodeEmbeddings.get_vector(metadata["ProductId"][i]),
# )


nodeVectors = np.array(nodeVectors)
nodeTargets = np.array(nodeTargets)

scale = StandardScaler()

model = LogisticRegressionCV(max_iter=8000, class_weight={"N": 0.14, "Y": 0.86})

nodeVectors = scale.fit_transform(nodeVectors)

model.fit(nodeVectors, nodeTargets)

joblib.dump(model, "../models/LRM.sav")

X_train, X_test, y_train, y_test = train_test_split(
    nodeVectors,
    nodeTargets,
    train_size=0.80,
)

X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)

print("Accuracy", accuracy_score(y_test, y_pred_test))
