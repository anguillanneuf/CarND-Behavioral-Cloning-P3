#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:08:31 2016

@author: tz
"""
import numpy as np; import pandas as pd; import os; import pickle
import cv2; import tensorflow as tf
from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS

    
# command line flags.
flags.DEFINE_integer('height', 32, "Resize image height")
flags.DEFINE_integer('width', 64, "Reisze image width")
flags.DEFINE_float('epsilon', 0.08, "Adjust left/right images by ε")


def main(_):

    raw = pd.read_csv('./data/driving_log.csv')
    
    # strips white spaces in text fields.
    for i in raw.columns:
        if isinstance(raw[i][1], str):
            raw[i]=raw[i].map(str.strip)
    
    tot = raw.shape[0]*3*2
    h = FLAGS.height; w = FLAGS.width; ε = FLAGS.epsilon
    
    # create empty dictionary. 
    dat = {'features':np.zeros(shape=[tot,h,w,3]), 
           'labels':np.zeros(shape=tot), 
           'position': ['' for i in range(tot)], 
           'notes': ['org' for i in range(int(tot/2))]+
                    ['aug' for i in range(int(tot/2))]}
    
    for i, j in enumerate(os.listdir('./data/IMG')):

        # original/flipped images.
        img = cv2.cvtColor(cv2.imread('./data/IMG/'+j.strip()), 
                           cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        flipped = cv2.flip(img,1)
        
        # offset angles for left/right images.
        # flip angles for flipped images.
        
        # position in text.
        pos_old = ('left' if j.find('left')>=0 else 
                   'right' if j.find('right')>=0 else 'center')    
        pos_new = ('right' if j.find('left')>=0 else 
                   'left' if j.find('right')>=0 else 'center')
        # numerical angles.
        old = raw[raw[pos_old]=='IMG/'+j]['steering']
        adjusted = np.clip(old+ε if pos_old=='left' else 
                           old-ε if pos_old=='right' else old, -1., 1.)
        new = adjusted*-1.
        
        # write to dictionary.
        dat['features'][i] = img
        dat['labels'][i] = adjusted
        dat['position'][i] = pos_old
        dat['features'][i+int(tot/2)] = flipped
        dat['labels'][i+int(tot/2)] = new
        dat['position'][i+int(tot/2)] = pos_new
    
    # this takes care of shuffling.
    X_train, X_test, y_train, y_test = \
    train_test_split(dat['features'], dat['labels'], test_size=0.1)
    
    train = {'features': X_train, 'labels': y_train}
    test = {'features': X_test, 'labels': y_test}
    
    with open("./data/train.p", "wb") as f:
        pickle.dump(train, f)
    with open("./data/test.p", "wb") as f:
        pickle.dump(test, f)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()