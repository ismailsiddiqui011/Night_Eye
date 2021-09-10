import numpy as np
import tensorflow as tf
import cv2
from skimage.transform import resize
import re
import collections
import warnings
import os
warnings.filterwarnings('ignore')
import streamlit as st
import model_arch
from PIL import Image

st.title('Night Eye')
st.image('https://i.imgur.com/7nocKNV.png', width = 300)

model = model_arch.model_arch()

def clean(file):
    return int(re.findall('[0-9]+', file)[0])

files = {}
weight_dir = 'https://github.com/ismailsiddiqui011/Night_Eye/tree/main/Weights'
files = dict(zip(list(map((clean), os.listdir(weight_dir))), os.listdir(weight_dir)))
files = list(collections.OrderedDict(sorted(files.items())).values())
for i, file in enumerate(files):
    path = os.path.join(weight_dir, file)
    weights = np.load(path, allow_pickle = True)
    model.weights[i] = weights
    model.weights[i].assign(tf.reshape(weights, model.weights[i].numpy().shape))

img_file = st.file_uploader("Upload an image", type = ["png", "jpg", "jpeg"])

def process(image):
    if image is not None:
        img = np.array(Image.open(image))
        img = img/255
        img = resize(img, (256, 256))
        return img
    else:
        st.text('Upload a Image')	
if img_file is not None:
    img = process(img_file)

def predict(image):
    img = np.expand_dims(image, axis = 0)   
    pred = model.predict(img)[0]
    pred = np.clip(pred, 0, 1)
    return pred

if img_file is not None:
    pred = predict(img)
    st.image([img, pred], caption = ['Input', 'Prediction'], width = 256)
