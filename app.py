import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

df = pd.read_csv('caltech101_binary_codes.csv')
df['binary_codes'] = df['binary_codes'].apply(lambda x: np.array(json.loads(x), dtype=np.int32))

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img_arr = np.array(img).astype(np.float32)
    img_arr = preprocess_input(img_arr)
    return img_arr

model = load_model('model/caltech101_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            return redirect(url_for('results', filename=file.filename))
    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_arr = preprocess_image(image_path)
    img_arr = np.expand_dims(img_arr, axis=0)
    feature = model.predict(img_arr)
    
    binary_code = np.where(feature < 0.5, 0, 1).flatten()
    
    distances = df['binary_codes'].apply(lambda x: np.sum(np.abs(x - binary_code)))
    similar_indices = distances.nsmallest(10).index
    
    similar_images = []
    for idx in similar_indices:
        similar_img_path = df.iloc[idx]['filepaths']
        relative_path = os.path.relpath(similar_img_path, 'D:/Project/Flask').replace('\\', '/')
        similar_images.append(relative_path)
    
    return render_template('results.html', query_image=filename, similar_images=similar_images)

if __name__ == '__main__':
    app.run(debug=True)