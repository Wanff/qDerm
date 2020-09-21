import sys 
import numpy as np
#had to install pandas on python2 cuz thtat's what it uses?
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from PIL import Image

lr = 1e-5
init = 'normal'
activ = 'relu'
optim = 'adam'
epochs = 50
batch_size = 64
input_shape = (224,224,3)

checkpoint_path = "path"
im = Image.open("moles/test.jpg")
width, height = im.size

if width == 224 & height == 224:
    model = build_model(lr=lr, init= init, activ= activ, optim=optim, input_shape= input_shape)
    # model.load_weights(checkpoint_path)
    print("hi")

    prediction = model.predict(im)
    print(str(prediction))
else:
    print("Image does not possess the right dimensions :/")


def build_model(input_shape= (224,224,3), lr = 1e-3, num_classes= 2, init= 'normal', activ= 'relu', optim= 'adam'):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3),padding = 'Same',input_shape=input_shape,
                     activation= activ, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.MaxPool2D(pool_size = (2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3),padding = 'Same', 
                     activation =activ, kernel_initializer = 'glorot_uniform'))
    model.add(keras.layers.MaxPool2D(pool_size = (2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    model.summary()

    if optim == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(lr=lr)

    else:
        optimizer = keras.optimizers.Adam(lr=lr)

    model.compile(optimizer = optimizer ,loss = "binary_crossentropy", metrics=["accuracy"])
    return model