#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 0.0119
"""
Created on Fri Dec 30 17:08:31 2016

@author: tz
"""
import os
import json; import h5py; import pickle;
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
#from keras.initializations import he_normal
from keras.layers.normalization import BatchNormalization

import numpy as np
import matplotlib.pyplot as plt


with open('./data/train.p', mode='rb') as f:
    train = pickle.load(f)
X_train_, y_train_ = train['features'], train['labels']

with open('./data/test.p', mode='rb') as f:
    test = pickle.load(f)
X_test_, y_test_ = test['features'], test['labels']

#with open('./mydata/mydat.p', mode='rb') as f:
#    mydat = pickle.load(f)
#X_add, y_add = mydat['features'], mydat['labels']
#
#with open('./mydata/train.p', mode='rb') as f:
#    mytrain = pickle.load(f)
#X_mytrain, y_mytrain = mytrain['features'], mytrain['labels']
#
#with open('./mydata/test.p', mode='rb') as f:
#    mytest = pickle.load(f)
#X_mytest, y_mytest = mytest['features'], mytest['labels']
#
#X_train = np.append(X_train_, X_mytrain, axis = 0)
#y_train = np.append(y_train_, y_mytrain)
#X_test = np.append(X_test_, X_mytest, axis = 0)
#y_test = np.append(y_test_, y_mytest)

#plt.hist(y_train, bins=50)

# gen = ImageDataGenerator()
gen_train = ImageDataGenerator(height_shift_range=0.2)
gen_test = ImageDataGenerator()


def get_model(time_len=1):
    h,w,c=32,64,3 #48,96,3 #32,64,3  

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
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

    
# limit the number of examples with close to zero steering angles
def limit(X, y, α=.5):
    bad = [k for k,v in enumerate(y) if v >=-.08 and v <=.08]
    good = list(set(range(0, len(y)))-set(bad))
    n = len(bad)
    new = good + [bad[i] for i in np.random.randint(0, n, int(n*α))]
    X,y = X[new,], y[new]
    return X, y
    

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