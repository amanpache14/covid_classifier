from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Detector.h5'
# Load your trained model
global sess
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
global graph
graph = tf.get_default_graph()
with graph.as_default():
    set_session(sess)
    model = load_model(MODEL_PATH)
    model.compile(loss=tf.keras.losses.binary_crossentropy,optimizer="adam",metrics=['accuracy'])

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])
    with graph.as_default():
        set_session(sess)
        preds = model.predict_classes(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        if preds[0][0] == 0:
            return str("COVID INFECTED")
        elif preds[0][0] == 1:
            return str("NOT INFECTED WITH COVID")
    return None


if __name__ == '__main__':
    app.run(debug=True)

