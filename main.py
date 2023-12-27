from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,fbeta_score
from sklearn.model_selection import cross_val_score
import gensim
import numpy as np
import joblib
from gensim.models.doc2vec import Doc2Vec

model_filename = r"models/logistic_regression.joblib"#load the LogisticRegression model
LogisticRegressionModel=joblib.load(model_filename)
model=Doc2Vec.load(r"models/doc2vec_model")#load the Doc2Vec model
review=input("Write the review you want to test:")
review = gensim.utils.simple_preprocess(review)
review_vector= model.infer_vector(review)
review_vector_reshaped = np.array(review_vector).reshape(1, -1)
prediction = LogisticRegressionModel.predict(review_vector_reshaped)
print("Predicted Label:", prediction)