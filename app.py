from flask import Flask, request, render_template

#import aiohttp, asyncio
import requests, os
from io import BytesIO

# fastai
from fastai import *
from fastai.vision import *
import fastai

app = Flask(__name__)
classes = ['sailboat', 'catamaran', 'motorboat']

@app.route('/')
def hello_world():
    return 'Boat Classifier'

@app.route('/predict', methods=['GET'])
def predict():
    url = ''
    pred_class = ''

    if (request.args):
        url = request.args['url']
        response = requests.get(url)

        img_class = open_image(BytesIO(response.content))
        #pred_class,pred_idx,outputs = learn.predict(img_class)

    return render_template('predict.html', url = url, boat_type = pred_class)

