import joblib
from flask import Flask, jsonify, request
from flask_cors import cross_origin
from gensim.utils import simple_preprocess

from node2vec import *

logModel = joblib.load("models/LRM.sav")
docModel = Word2Vec.load("models/doc2vec.model")

app = Flask(__name__)


@app.route("/detection_server", methods=["POST"])
@cross_origin(origin="*", headers=["Content-Type", "Authorization"])
def detectionServer():
    review = "Rambagh is wonderful. Great Property, clean room,excellent hospitality amazing services. Services are quick. Helping staff,specially I would like to thanks siddhi, Shiv, Sanchit, riddhi , komal , vikas and shubham from front office department & Danish from housekeeping for there service. Definetly I recommend visitors to get experience of beautiful service and gesture from Fairmont Jaipur."
    review = simple_preprocess(review)
    review_vectors = docModel.infer_vector(review)
    review_vectors = np.array(review_vectors).reshape(1, -1)

    prediction = logModel.predict(review_vectors)
    print("Predicted label: ", prediction)
    return prediction
