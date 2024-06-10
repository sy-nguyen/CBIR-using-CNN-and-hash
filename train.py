import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model

path = 'static/data/caltech-101'
filepaths = []
labels = []
classlist = sorted(os.listdir(path))

for klass in classlist:
    classpath = os.path.join(path, klass)
    flist = os.listdir(classpath)
    for f in flist:
        fpath = os.path.join(classpath, f)
        filepaths.append(fpath)
        labels.append(klass)

df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
df = df[df['labels'] != 'BACKGROUND_Google'].reset_index(drop=True)

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img_arr = np.array(img).astype(np.float32)
    img_arr = preprocess_input(img_arr)
    return img_arr

def preprocess_data(df):
    preprocessed_images = [preprocess_image(row['filepaths']) for index, row in df.iterrows()]
    return np.array(preprocessed_images)

images = preprocess_data(df)

def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(2048, activation='leaky_relu')(x)
    return Model(inputs=base_model.input, outputs=x)

model = create_model()

model.save('caltech101_model.h5')
features = model.predict(images, batch_size=32, verbose=1)
binary_codes = np.where(features < 0.5, 0, 1)

df['binary_codes'] = [json.dumps(code.tolist()) for code in binary_codes]
df.to_csv('caltech101_binary_codes.csv', index=False)

model.save('model/caltech101_model.h5')