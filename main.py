import gensim
import numpy as np
import joblib
from gensim.models.doc2vec import Doc2Vec
from flask import Flask, request
model_filename = r"C:\Users\Turgut\Desktop\Dataset\trained_logistic_regression_model.joblib"#load the LogisticRegression model
LogisticRegressionModel=joblib.load(model_filename)
model=Doc2Vec.load(r"C:\Users\Turgut\Desktop\Dataset\trained_model")#load the Doc2Vec model



app = Flask(__name__)

@app.route('/get_char', methods=['POST'])
def get_char():
    sentence = request.form.get('sentence', '')#GET
    review=sentence
    review = gensim.utils.simple_preprocess(review)
    review_vector = model.infer_vector(review)
    review_vector_reshaped = np.array(review_vector).reshape(1, -1)
    prediction = LogisticRegressionModel.predict(review_vector_reshaped)
    print("Predicted Label:", prediction)
    prediction=str(prediction)
    return prediction


if __name__ == '__main__':
    # Run the Flask server on port 5000
    app.run(port=5000)

