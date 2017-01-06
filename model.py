#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 0.0119
"""
Created on Fri Dec 30 17:08:31 2016

@author: tz
"""
import os
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint

import pickle; import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import h5py;

with open('./data/train.p', mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']

with open('./data/test.p', mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']

gen = ImageDataGenerator()

def get_model(time_len=1):
    h,w,c=32,64,3#48,96,3#32,64,3  

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(h,w,c),
              output_shape=(h,w,c)))
    model.add(Convolution2D(3, 1, 1, subsample=(1, 1), border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(16, 7, 7, subsample=(4, 4), border_mode="same"))
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
    
    if not os.path.exists("./outputs"): os.makedirs("./outputs")
    
    model = get_model()
    
    b = 64

    checkpointer = ModelCheckpoint("./outputs/model.hdf5", verbose=1, 
                               save_best_only=True)
      
    model.fit_generator(gen.flow(X_train, y_train, batch_size=b),
                samples_per_epoch=len(X_train),
                nb_epoch=8,
                validation_data=gen.flow(X_test, y_test, batch_size=b),
                nb_val_samples=len(X_test),
                callbacks=[checkpointer]
                )
    
    model.save_weights("./outputs/model06.h5")
    with open('./outputs/model06.json', 'w') as f:
        json.dump(model.to_json(), f)

if __name__ == '__main__':
    main()