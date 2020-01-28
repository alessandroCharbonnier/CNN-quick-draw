#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import psutil
import glob
import os


# In[2]:


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = f'{iteration} / {max}'
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


# In[3]:


IMAGE_ROWS = 28
IMAGE_COLS = 28
IMAGE_SIZE = IMAGE_ROWS * IMAGE_ROWS
IMAGE_SHAPE = (IMAGE_ROWS, IMAGE_COLS, 1)

IMAGE_DIR = "./data/"

BATCH_SIZE = 2048


# In[4]:


with open('short_categories.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
LABEL_MAP = [(x.strip(), i) for i, x in enumerate(content)]
NUM_CLASSES = len(LABEL_MAP)


# In[5]:


model = Sequential()
model.add(Conv2D(input_shape=IMAGE_SHAPE, filters=64, kernel_size=(3, 3), padding='same', activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(1,1),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=2048,activation="relu"))
model.add(Dense(units=2048,activation="relu"))
model.add(Dense(NUM_CLASSES, activation="softmax"))

print(model.summary())


# In[6]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])


# In[ ]:


x_train = np.ndarray(shape=(0, IMAGE_SIZE), dtype=np.float16)
y_train = np.array([], dtype=np.uint16)

i = 0
max = len(LABEL_MAP)

for label in LABEL_MAP:
    printProgressBar(i, max)
    x_train = np.concatenate((x_train, np.load(f'{IMAGE_DIR}{label[0]}.npy')))
    '''labels = np.full(elements.shape[0], label[1])
    y_train = np.append(y_train, labels)'''
    i += 1

x_train, x_validate, y_train, y_validate = tts(x_train, y_train, test_size=.2)


# ### elements[0]

# In[ ]:




