#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:08:31 2016
@author: tz

"To remove a bias towards driving straight the training data includes a higher 
proportion of frames that represent road curves."
"""


import numpy as np; import pandas as pd

import cv2; import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
df = pd.read_csv('./data/driving_log.csv')
h = 160; w = 320; c = 3

# center image
img = cv2.imread('./data/' + df.iloc[4042,0].strip())
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# flipped image
flipped = cv2.flip(img, 1)
plt.imshow(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))

# steering angle distribution
print(np.mean(df['steering']==0))
print(np.sum(df['steering']!=0)*3*2)
plt.hist(df['steering'], bins=50, color='#FF69B4')

# steering angle distribiton after removing 0 angles
plt.hist(df['steering'][df['steering']!=0], bins=50, color='#FF69B4')

# adjust steering angles for left/right images: left +ε, right -ε
ε=.25

# shuffle data at the beginning of each epoch
n_sample_zeros = 500
nonzeros = df[df['steering']!=0]
zeros = df[df['steering']==0].sample(n_sample_zeros)
mydf = pd.concat([nonzeros, zeros], ignore_index=True)
shuffled = mydf.reindex(np.random.permutation(mydf.index))

# melt shuffled to create model-ready df
small = shuffled[['center','left','right','steering']]
melted = pd.melt(small, id_vars=['steering'], var_name='position')
def epsilon(row):
    a = (row['steering'] if row['position']=='center' else 
            row['steering']+ε if row['position']=='left' else
            row['steering']-ε)
    return np.clip(a, -1, 1)
melted['steering_new'] = melted.apply(epsilon, 1)
melted.tail()

# to-do: the ordering of my pre-processing is wrong
# 1. mirror
# 2. steering_new for left/right
# 3. concat zeros and nonzeros
# 4. shuffle for model