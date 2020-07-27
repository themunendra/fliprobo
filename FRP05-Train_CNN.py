#!/usr/bin/env python
# coding: utf-8

import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Activation, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow.keras.backend as k
import cv2

num_classes = 3 #3 celebrities
img_rows, img_cols = 224, 224
batch_size = 256 # images in one go
dir_path = 'celeb_clean'
train_data_dir = './celeb_clean/train'
validation_data_dir = './celeb_clean/validation'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

traindata = train_datagen.flow_from_directory(train_data_dir,
                                              target_size=(img_rows, img_cols),
                                              batch_size=batch_size,
                                              class_mode='categorical')

testdata = validation_datagen.flow_from_directory(validation_data_dir,
                                                  target_size=(img_rows, img_cols),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

#Clear session to cleanup older models
k.clear_session()

#create model
model = Sequential()

# First CONV-ReLU Layer
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(BatchNormalization())

# Second CONV-ReLU Layer
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(BatchNormalization())

# Max Pooling with Dropout
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
          
# 3rd set of CONV-ReLU Layers         
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

# 4th Set of CONV-ReLU Layers
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

# Max Pooling with Dropout
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# 5th Set of CONV-ReLU Layers
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

# 6th Set of CONV-ReLU Layers
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

# Max Pooling with Dropout
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# 7th Set of CONV-ReLU Layers
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

# 8th Set of CONV-ReLU Layers
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

# Max Pooling with Dropout
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# 9th Set of CONV-ReLU Layers
model.add(Conv2D(filters=1024, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

# 10th Set of CONV-ReLU Layers
model.add(Conv2D(filters=1024, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

# Max Pooling with Dropout
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# First set of FC or Dense Layers
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Second set of FC or Dense Layers
model.add(Dense(units=512,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Final Dense Layer
model.add(Dense(units=num_classes, activation="softmax"))

#Compile the model
from keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

#Print model Summary
model.summary()

#Define Checkpoints and early stopping
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("m_cnn.h5", 
                             monitor='val_accuracy', 
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=False, 
                             mode='auto', 
                             period=1)

early = EarlyStopping(monitor='val_accuracy', 
                      min_delta=0, 
                      patience=3, 
                      verbose=1, 
                      mode='auto')

#Fit the model
hist = model.fit_generator(steps_per_epoch=100, #Number of training images//batch size
                           generator=traindata, 
                           validation_data= testdata, 
                           validation_steps=20, #Number of test images//batch size,
                           epochs=10,
                           callbacks=[checkpoint,early])
