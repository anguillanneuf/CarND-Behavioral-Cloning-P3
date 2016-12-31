#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:08:31 2016

@author: tz
"""


import numpy as np; import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import cv2; import matplotlib.pyplot as plt


# load dataset
df = pd.read_csv('./data/driving_log.csv')
h = 160; w = 320; c = 3

# sanity check
img = cv2.imread('./data/' + df.iloc[4042,2].strip())
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

flipped = cv2.flip(img, 1)
plt.imshow(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))

plt.hist(df['steering'], bins=50, color='#FF69B4')

# adjust steering angles for left/right images by ε=.25

shuffled = df.reindex(np.random.permutation(df.index))

#==============================================================================
# "To remove a bias towards driving straight the training data includes a 
# higher proportion of frames that represent road curves."
#==============================================================================

print(np.mean(df['steering']==0))
print(np.sum(df['steering']==0)*3*2)
plt.hist(df['steering'][df['steering']!=0], bins=50, color='#FF69B4')