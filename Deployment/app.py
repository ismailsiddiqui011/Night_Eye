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

choice = st.selectbox('Choose one of the following', ('URL', 'Upload Image'))
if choice == 'URL':
    image_path = st.text_input('Enter image URL...')
    try:
      img = imread(image_path)
      img = resize(img, (256, 256))
    except:
      st.markdown('Enter a Valid URL!!!')

if choice == 'Upload Image':
    img = st.file_uploader('Upload an Image')
    if img == None:
      st.markdown('Upload Image')
    else:
      img = Image.open(img)
      img = np.array(img)/255
      img = resize(img, (256, 256))
try:
    pred = model.predict(np.expand_dims(img, 0))[0]
    pred = np.clip(pred, 0, 1)
    st.image([img, pred], caption = ['Input', 'Prediction'], width = 256)
except:
    pass
