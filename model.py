#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:08:31 2016

@author: tz
"""
import os
import json
import h5py
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization


# Load data. 
with open('./data/train.p', mode='rb') as f:
    train = pickle.load(f)
X_train_, y_train_ = train['features'], train['labels']

with open('./data/test.p', mode='rb') as f:
    test = pickle.load(f)
X_test_, y_test_ = test['features'], test['labels']


# Python generator. There is a version I wrote on my own which can be 
# found in sandbox.py. But for my final model, I used keras's. 
gen_train = ImageDataGenerator(height_shift_range=0.2)
gen_test = ImageDataGenerator()


def get_model():
    h,w,c=32,64,3

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(h,w,c),
              output_shape=(h,w,c)))
    model.add(Convolution2D(3, 1, 1, subsample=(1, 1), border_mode='same',
                            init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(16, 5, 5, subsample=(4, 4), border_mode="same",
                            init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same",
                            init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same", 
                            init = 'he_normal'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

    
# Limit the number of near-zero angles (defined by offset β) by a factor of α.
def limit(X, y, β=.08, α=.5):  
    # indices of near-zero steering angles
    bad = [k for k,v in enumerate(y) if v >=-β and v <=β]
    # indices of bigger steering angles
    good = list(set(range(0, len(y)))-set(bad))
    n = len(bad)
    # merges the good indices with a portion of the bad indices
    new = good + [bad[i] for i in np.random.randint(0, n, int(n*α))]
    
    return X[new,], y[new]
    

def main():
    
    if not os.path.exists("./output"): 
        os.makedirs("./output")
    
    model = get_model()
    
    b = 128

    checkpointer = ModelCheckpoint("./output/model.hdf5", verbose=1, 
                               save_best_only=True)
    
    X, y = limit(X_train_, y_train_, .5)
    
    model.fit_generator(gen_train.flow(X, y, batch_size=b),
                samples_per_epoch=len(X),
                nb_epoch=10,
                validation_data=gen_test.flow(X_test_, y_test_, batch_size=b),
                nb_val_samples=len(X_test_),
                callbacks=[checkpointer]
                )
    
    model.save_weights("model.h5")
    with open('model.json', 'w') as f:
        json.dump(model.to_json(), f)

if __name__ == '__main__':
    main()
