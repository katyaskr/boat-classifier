from flask import Flask, request, render_template

import aiohttp, asyncio
import requests, os
from io import BytesIO

# fastai
from fastai import *
from fastai.vision import *
import fastai

app = Flask(__name__)
classes = ['sailboat', 'catamaran', 'motorboat']
export_file_url = 'https://drive.google.com/uc?export=download&id=1kSUp-9Q2f6fUPx4133sKmq2_jp0Ft1bj'
export_file_name = 'export.pkl'
path = Path('.')

learn = load_learner(path)

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

#loop = asyncio.get_event_loop()
#tasks = [asyncio.ensure_future(setup_learner())]
#learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
#loop.close()

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
        pred_class,pred_idx,outputs = learn.predict(img_class)

    return render_template('predict.html', url = url, boat_type = pred_class)

