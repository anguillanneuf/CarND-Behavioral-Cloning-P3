#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:08:31 2016

@author: tz
"""
import os
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

import pickle
from keras.preprocessing.image import ImageDataGenerator
import h5py

with open('./data/train.p', mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']

with open('./data/validation.p', mode='rb') as f:
    validation = pickle.load(f)
X_val, y_val = validation['features'], validation['labels']

gen = ImageDataGenerator()

def get_model(time_len=1):
  h,w,c=32,64,3  

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(h,w,c),
            output_shape=(h,w,c)))
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model

def main():

  model = get_model()
  model.fit_generator(
    gen.flow(X_train, y_train, batch_size=128),
    samples_per_epoch=len(X_train),
    nb_epoch=10,
    validation_data=gen.flow(X_val, y_val, batch_size=128),
    nb_val_samples=1500
  )

  print("Saving model weights and configuration file.")

  if not os.path.exists("./outputs"):
      os.makedirs("./outputs")

  model.save_weights("./outputs/model_03.h5")
  with open('./outputs/model_03.json', 'w') as f:
    json.dump(model.to_json(), f)


if __name__ == '__main__':
    main()