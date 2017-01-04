#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:08:31 2016

@author: tz
"""

import numpy as np; import pandas as pd; import os; import pickle
import cv2

raw = pd.read_csv('./data/driving_log.csv')
h = 160; w = 320; c = 3; s = [1,h,w,3]
ε = .25
dat = {'features':np.zeros(shape=s), 'labels':np.array([0]), 
       'notes': ['unk'], 'position'; ['unk']}

for j in os.listdir('./data/IMG')[:100]:
    
    img = cv2.cvtColor(cv2.imread('./data/IMG/' + j.strip()), 
                       cv2.COLOR_BGR2RGB)
    flipped = cv2.flip(img,1)
    
    pos = ('left' if j.find('left')>=0 else 
           'right' if j.find('right')>=0 else 'center')
    
    old = raw[raw[pos]=='IMG/'+j]['steering']
    old_adjusted = np.clip(old+ε if pos=='left' else 
                           old-ε if pos=='right' else old, -1, 1)
    new = old_adjusted*-1
    
    dat['features']=np.append(dat['features'], img.reshape(s), axis=0)
    dat['labels']=np.append(dat['labels'], old_adjusted)
    dat['notes'].append('original')
    dat['position'].append(pos)
    dat['features']=np.append(dat['features'], flipped.reshape(s), axis=0)
    dat['labels']=np.append(dat['labels'], new)
    dat['notes'].append('transformed')
    dat['position'].append(pos)
    
with open("./data/dat.p", "wb") as f:
    pickle.dump(dat, f)
    
# validation and test should come from original+center