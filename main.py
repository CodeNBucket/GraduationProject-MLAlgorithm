from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,fbeta_score
from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from main2 import dict
import gensim
import numpy as np
file_path=r"C:\Users\Turgut\Desktop\Dataset\output_meta_yelpHotelData_NRYRcleaned.txt"
meta_file=open(file_path)
meta=meta_file.readlines()#meta contains all the meta_data and I used it just to get the labels
meta_file.close()

labels=[]
for label in meta:
    labels.append(label.split()[4]) #labels of the revies are added to 'labels' one by one
model=Doc2Vec.load(r"C:\Users\Turgut\Desktop\Dataset\trained_model")#load the model
accuracy_percentage = (dict[0] / (dict[0] + dict[1])) * 100
formatted_accuracy = "{:.2f}".format(accuracy_percentage)
print("While there are 5854 reviews used for training the model, while testing the models vectorization accuracy, "
      "of the vectors, " + str(dict[0]) + " have been found most similar with itself. "
      "When the trained model and inferred vectors created afterwards are compared, "
      "the Doc2Vec has %" + formatted_accuracy + " accuracy which is accaptable")





vectors=[]
for i in range(model.corpus_count):
    vector=model.dv[i]
    vectors.append(vector)#vectors contains vector embeddings of dataset

X_train,X_test, y_train, y_test = train_test_split( vectors, labels, random_state=1,stratify=labels)



C_values=[0.001,0.01,0.1,1,10,100,1000]
for C in C_values:

    LogisticRegressionModel = LogisticRegression(class_weight={'N': 1, 'Y': 6},C=C,max_iter=2000)
    LogisticRegressionModel.fit(X_train, y_train)
    y_pred=LogisticRegressionModel.predict(X_test)
    y_true=y_test
    print("Test set score: {:.2f}".format(LogisticRegressionModel.score(X_test, y_test)))
    print(C)
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


LogisticRegressionModel = LogisticRegression(class_weight={'N': 1, 'Y': 6},C=1,max_iter=2000)
LogisticRegressionModel.fit(X_train, y_train)
review=input("Write the review you want to test:")
review = gensim.utils.simple_preprocess(review)
review_vector= model.infer_vector(review)
review_vector_reshaped = np.array(review_vector).reshape(1, -1)
prediction = LogisticRegressionModel.predict(review_vector_reshaped)
print("Predicted Label:", prediction)