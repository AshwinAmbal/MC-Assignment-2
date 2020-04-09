from flask import Flask, jsonify, request
from cheroot.wsgi import Server as WSGIServer
from src.utils import jsonToCSV, INDEX_TO_GESTURE
from src.feature_vector_mother import feature_vector_mother
from src.feature_vector_fun import feature_vector_fun
import pickle
import os
import numpy as np
import pandas as pd

app = Flask(__name__)


def get_result_from_model(model_index, gesture_id, feature_vector):
    model = pickle.load(open(os.path.abspath('../models/model_{}/{}/model.pkl'.format(model_index, gesture_id)), 'rb'))
    pca = pickle.load(open(os.path.abspath('../models/model_{}/{}/pca.pkl'.format(model_index, gesture_id)), 'rb'))
    minmax = pickle.load(open(os.path.abspath('../models/model_{}/{}/minmax.pkl'.format(model_index, gesture_id)), 'rb'))
    feature_vector = pd.DataFrame([feature_vector])
    feature_vector = minmax.transform(feature_vector)
    feature_vector = pca.transform(feature_vector)
    return model.predict_proba(feature_vector)[0][1]


@app.route('/',  methods=['GET', 'POST'])
def predict():
    received_data = request.get_json()
    extracted_df = jsonToCSV(received_data)
    # TODO: Change below feature vector calls to respective functions. Uncomment for debugging temporarily
    # buy = feature_vector_fun(extracted_df, test=True)
    # communicate = feature_vector_fun(extracted_df, test=True)
    fun = feature_vector_fun(extracted_df, test=True)
    # hope = feature_vector_mother(extracted_df, test=True)
    mother = feature_vector_mother(extracted_df, test=True)
    # really = feature_vector_mother(extracted_df, test=True)
    feature_vectors = [buy, communicate, fun, hope, mother, really]
    results = dict()
    for i in range(1, 5):
        scores = [0] * 6
        for gesture_id in INDEX_TO_GESTURE:
            scores[ord(gesture_id)-65] = get_result_from_model(i, gesture_id, feature_vectors[ord(gesture_id)-65])
        results[i] = INDEX_TO_GESTURE[chr(ord('A') + np.argmax(scores))]
    return jsonify(results)


if __name__ == '__main__':
    server = WSGIServer(bind_addr=('127.0.0.1', 9696), wsgi_app=app, numthreads=100)
    try:
        print("Serving on {}".format(server))
        server.start()
    except KeyboardInterrupt:
        server.stop()
