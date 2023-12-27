import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,fbeta_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

file_path=r"data/metadata.txt"
meta_file=open(file_path)
meta=meta_file.readlines()#meta contains all the meta_data and I used it just to get the labels
meta_file.close()

labels=[]
for label in meta:
    labels.append(label.split()[4]) #labels of the revies are added to 'labels' one by one
model=Doc2Vec.load(r"models/doc2vec_model")#load the model

vectors=[]
for i in range(model.corpus_count):
    vector=model.dv[i]
    vectors.append(vector)#vectors contains vector embeddings of dataset

X_train,X_test, y_train, y_test = train_test_split( vectors, labels, random_state=1,stratify=labels)

LogisticRegressionModel = LogisticRegression(class_weight={'N': 1, 'Y': 6},C=0.01)
LogisticRegressionModel.fit(X_train, y_train)
y_pred=LogisticRegressionModel.predict(X_test)
y_true=y_test
print("Test set score: {:.2f}".format(LogisticRegressionModel.score(X_test, y_test)))
#scores = cross_val_score(LogisticRegressionModel, vectors, labels, cv=10)
#print("Cross  validation %0.2f" % (scores.mean()))
# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_true, y_pred,pos_label='N')
print("Precision:", precision)

# Recall
recall = recall_score(y_true, y_pred,pos_label='N')
print("Recall:", recall)

# F1-Score
f1 = f1_score(y_true, y_pred,pos_label='N')
print("F1-Score:", f1)
fb = fbeta_score(y_true, y_pred, pos_label='N', beta=0.5)
print("Fbeta-Score:", fb)
label_encoder = LabelEncoder()
label_encoder.fit(y_true) #Assigns numerical values 0 1 for labels N Y

y_true_binary = label_encoder.transform(y_true)
y_pred_binary = label_encoder.transform(y_pred)

roc_auc = roc_auc_score(y_true_binary, y_pred_binary)
print("ROC-AUC:", roc_auc)
model_filename = r"models/logistic_regression.joblib"
joblib.dump(LogisticRegressionModel, model_filename)