import numpy as np
import pandas as pd
import tensorflow as tf
for device in tf.config.get_visible_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import os
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import cv2
from skimage.io import imshow, imsave, imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import *
import tensorflow.keras.backend as K
import re
import collections
import warnings
warnings.filterwarnings('ignore')
tf.config.experimental_run_functions_eagerly(True)

def model_arch():
  '''Creates a model'''
  inp = Input(shape = (256, 256, 3))

  conv1 = Conv2D(32, kernel_size = 5, padding = 'same', name='Conv_1', dilation_rate = 2)(inp)
  conv1 = BatchNormalization()(conv1)
  conv1 = LeakyReLU()(conv1)
  conv2 = Conv2D(32, kernel_size = 5, padding = 'same', name='Conv_2', dilation_rate = 3)(conv1)
  conv2 = BatchNormalization()(conv2)
  conv2 = LeakyReLU()(conv2)
  pool1 = MaxPool2D(2, padding = 'same')(conv2)

  conv3 = Conv2D(64, kernel_size = 5, padding = 'same', name='Conv_3', dilation_rate = 2)(pool1)
  conv3 = BatchNormalization()(conv3)
  conv3 = LeakyReLU()(conv3)
  conv4 = Conv2D(64, kernel_size = 5, padding = 'same', name='Conv_4', dilation_rate = 3)(conv3)
  conv4 = BatchNormalization()(conv4)
  conv4 = LeakyReLU()(conv4)
  pool2 = MaxPool2D(2, padding = 'same')(conv4)

  conv5 = Conv2D(128, kernel_size = 5, padding = 'same', name='Conv_5', dilation_rate = 2)(pool2)
  conv5 = BatchNormalization()(conv5)
  conv5 = LeakyReLU()(conv5)
  conv6 = Conv2D(128, kernel_size = 5, padding = 'same', name='Conv_6', dilation_rate = 3)(conv5)
  conv6 = BatchNormalization()(conv6)
  conv6 = LeakyReLU()(conv6)
  pool3 = MaxPool2D(2, padding = 'same')(conv6)

  conv7 = Conv2D(256, kernel_size = 5, padding = 'same', name='Conv_7', dilation_rate = 2)(pool3)
  conv7 = BatchNormalization()(conv7)
  conv7 = LeakyReLU()(conv7)
  conv8 = Conv2D(256, kernel_size = 5, padding = 'same', name='Conv_8', dilation_rate = 3)(conv7)
  conv8 = BatchNormalization()(conv8)
  conv8 = LeakyReLU()(conv8)
  pool4 = MaxPool2D(2, padding = 'same')(conv8)

  ###################################### Upsampling ############################

  up1 = Conv2DTranspose(256, kernel_size = 5, strides = 2, padding = 'same',
                             kernel_initializer = TruncatedNormal())(pool4)
  up1 = Concatenate()([conv8, up1])
  up1 = BatchNormalization()(up1)
  up1 = Conv2D(256, kernel_size = 5, padding = 'same')(up1)
  up1 = LeakyReLU()(up1)
  up1 = Conv2D(256, kernel_size = 5, padding = 'same')(up1)
  up1 = LeakyReLU()(up1)

  up2 = Conv2DTranspose(128, kernel_size = 5, strides = 2, padding = 'same',
                             kernel_initializer = TruncatedNormal())(up1)
  up2 = Concatenate()([conv6, up2])
  up2 = BatchNormalization()(up2)
  up2 = Conv2D(128, kernel_size = 5, padding = 'same')(up2)
  up2 = LeakyReLU()(up2)
  up2 = Conv2D(128, kernel_size = 5, padding = 'same')(up2)
  up2 = LeakyReLU()(up2)

  up3 = Conv2DTranspose(64, kernel_size = 5, strides = 2, padding = 'same',
                             kernel_initializer = TruncatedNormal())(up2)
  up3 = Concatenate()([conv4, up3])
  up3 = BatchNormalization()(up3)
  up3 = Conv2D(64, kernel_size = 5, padding = 'same')(up3)
  up3 = LeakyReLU()(up3)
  up3 = Conv2D(64, kernel_size = 5, padding = 'same')(up3)
  up3 = LeakyReLU()(up3)

  up4 = Conv2DTranspose(32, kernel_size = 5, strides = 2, padding = 'same',
                             kernel_initializer = TruncatedNormal())(up3)
  up4 = Concatenate()([conv2, up4])
  up4 = BatchNormalization()(up4)
  up4 = Conv2D(32, kernel_size = 5, padding = 'same')(up4)
  up4 = LeakyReLU()(up4)
  up4 = Conv2D(32, kernel_size = 5, padding = 'same')(up4)
  up4 = LeakyReLU()(up4)

  out = Conv2D(3, kernel_size = 5, padding = 'same')(up4)

  model = Model(inputs = inp, outputs = out)
  
  
  return model
