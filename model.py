#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:08:31 2016

@author: tz
"""

import numpy as np; import pandas as pd; import os; import pickle
import cv2; # import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

raw = pd.read_csv('./data/driving_log.csv')

tot = raw.shape[0]*2
h = 64; w = 128; c = 3; s = [1,h,w,3]; ε = .25

dat = {'features':np.zeros(shape=[tot,h,w,3]), 'labels':np.zeros(shape=tot), 
       'position': ['' for i in range(tot)], 
       'notes': ['org' for i in range(int(tot/2))]+['aug' for i in range(tot)]}

for i, j in enumerate(os.listdir('./data/IMG')):
    # original/flipped images.
    img = cv2.cvtColor(cv2.imread('./data/IMG/'+j.strip()), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w,h))
    flipped = cv2.flip(img,1)
    
    # adjusted angles for left/right images/new angles for flipped images
    pos_old = ('left' if j.find('left')>=0 else 
               'right' if j.find('right')>=0 else 'center')    
    pos_new = ('right' if j.find('left')>=0 else 
               'left' if j.find('right')>=0 else 'center')
    
    old = raw[raw[pos_old]=='IMG/'+j]['steering']
    adjusted = np.clip(old+ε if pos_old=='left' else 
                       old-ε if pos_old=='right' else old, -1., 1.)
    new = adjusted*-1.
    
    dat['features'][i] = img
    dat['labels'][i] = adjusted
    dat['position'][i] = pos_old
    dat['features'][i+int(tot/2)] = flipped
    dat['labels'][i+int(tot/2)] = new
    dat['position'][i+int(tot/2)] = pos_new

X_train, X_test, y_train, y_test = \
train_test_split(dat['features'], dat['labels'], test_size=0.2)

X_train, X_val, y_train, y_val = \
train_test_split(X_train, y_train, test_size=0.2)

train = {'features': X_train, 'labels': y_train}
validation = {'features': X_val, 'labels': y_val}
test = {'features': X_test, 'labels': y_test}

with open("./data/train.p", "wb") as f:
    pickle.dump(train, f)
with open("./data/validation.p", "wb") as f:
    pickle.dump(validation, f)
with open("./data/test.p", "wb") as f:
    pickle.dump(test, f)