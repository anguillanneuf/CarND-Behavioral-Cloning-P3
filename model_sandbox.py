#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:08:31 2016
@author: tz

"To remove a bias towards driving straight the training data includes a higher 
proportion of frames that represent road curves."
"""

import pickle

with open('./data/train.p', mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']

import matplotlib.pyplot as plt
plt.hist(y_train, color='#FF69B4')

import numpy as np
index_new = np.random.permutation(np.arange(X_train.shape[0]))
X_train, y_train = X_train[index_new], y_train[index_new]

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

def get_model(time_len=1):
  channel, height, width = 3, 160, 320  # camera format

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(channel, height, width),
            output_shape=(channel, height, width)))
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

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasRegressor

#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline

#from keras import backend as ktf

#h = 160; w = 320; c = 3
#batch_size = 32;
#
## center image
#j = 'left_2016_12_01_13_30_48_287.jpg'
#img = cv2.imread('./data/' + raw.iloc[4042,0].strip())
#img = cv2.imread('./data/' + j)
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
#img_small = cv2.resize(img, (128, 64))
#plt.imshow(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
#
## flipped image
#flipped = cv2.flip(img, 1)
#plt.imshow(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
#
## steering angle distribution
#print(np.mean(df['steering']==0))
#print(np.sum(df['steering']!=0)*3*2)
#plt.hist(df['steering'], bins=50, color='#FF69B4')
#
## steering angle distribiton after removing 0 angles
#plt.hist(df['steering'][df['steering']!=0], bins=50, color='#FF69B4')
#
## shuffle data at the beginning of each epoch
#n_sample_zeros = 500
#nonzeros = df[df['steering']!=0]
#zeros = df[df['steering']==0].sample(n_sample_zeros)
#mydf = pd.concat([nonzeros, zeros], ignore_index=True)
#shuffled = mydf.reindex(np.random.permutation(mydf.index))
#
## melt shuffled to create model-ready df
#small = shuffled[['center','left','right','steering']]
#melted = pd.melt(small, id_vars=['steering'], var_name='position')
#def epsilon(row):
#    a = (row['steering'] if row['position']=='center' else 
#            row['steering']+ε if row['position']=='left' else
#            row['steering']-ε)
#    return np.clip(a, -1, 1)
#melted['steering_new'] = melted.apply(epsilon, 1)
#melted.tail()


# 1. mirror
# 2. steering_adjusted for left/right
# 3. sample zeros and combine with nonzeros
# 4. shuffle for model

#==============================================================================
# 1. flip and save, this doubles my training data (ε=0.25)
#==============================================================================
#import numpy as np; import pandas as pd; import os; import cv2
#
## for each image, flip, save.
#if not os.path.exists('./data/IMG_new/'):
#    os.mkdir('./data/IMG_new/')
#
#for j in os.listdir('./data/IMG'):
#    flipped = cv2.flip(cv2.imread('./data/IMG/' + j.strip()),1)
#    j = (j.replace('left', 'right') if j.find('left')>=0 else
#          j.replace('right', 'left') if j.find('right')>=0 else j)
#    cv2.imwrite('./data/IMG_new/'+j.strip()[:-4]+'_new.jpg', flipped)
#
## for df_new, update steering/steering_adjusted, update position/value, save.
#raw = pd.read_csv('./data/driving_log.csv')
#df = pd.melt(raw[['center', 'left', 'right', 'steering']], 
#             id_vars=['steering'], var_name='position')
#
#def epsilon(row, ε=.25):
#    a = (row['steering']+ε if row['position']=='left' else
#         row['steering']-ε if row['position']=='right' else row['steering'])
#    return np.clip(a, -1, 1)
#    
#df['steering_adjusted'] = df.apply(epsilon,1,ε=0.25)
#
#def flip(row):
#    row['steering']*=-1
#    row['steering_adjusted']*=-1
#    row['position'] = ('right' if row['position']=='left' else
#                       'left' if row['position']=='right' else 'center')
#    t = 'IMG_new/'+row['value'].strip()[4:-4]+'_new.jpg'
#    row['value'] = (t.replace('left', 'right') if t.find('left')>=0 else
#                    t.replace('right', 'left') if t.find('right')>=0 else t)
#    return row
#
#df_new = df.apply(flip,1)
#
#pd.concat([df, df_new]).to_csv('./data/driving_log_new.csv', index=False)

#==============================================================================
# 2. function to sample more balanced data, find corresponding images
#==============================================================================
#import pandas as pd; import matplotlib.pyplot as plt
#
## unbalanced bins
#df = pd.read_csv('./data/driving_log_new.csv')
#plt.hist(df['steering_adjusted'], bins=100, color='#FF69B4')
#
## for each epoch, I can sample 1000 examples from steering=+/-.25/0
#def generate_epoch(df, n_c=1000, n_l=1000, n_r=1000, ε=0.25):
#    c = df[(df['steering_adjusted']==0)].sample(n_c)
#    l = df[(df['steering_adjusted']==-ε)].sample(n_l)
#    r = df[(df['steering_adjusted']==ε)].sample(n_r)    
#    m = df[(df['steering_adjusted']!=0)&
#           (df['steering_adjusted']!=ε)&
#           (df['steering_adjusted']!=-ε)]
#    mydf = pd.concat([c,l,r,m], ignore_index=True)
#    return mydf.reindex(np.random.permutation(mydf.index))
#
#mydf = generate_epoch(df,1000,1000,1000,0.25)
#plt.hist(mydf['steering_adjusted'],bins=50,color='#FF69B4')
#
##==============================================================================
## 3. crate pickle, split training.p (0.6), validation.p (0.2), test.p (0.2)
##==============================================================================
#import pandas as pd; import numpy as np; import matplotlib.pyplot as plt
#import pickle
#
#data = {'features':np.array([]), 'labels':np.array([])}
#         
#for k,v in enumerate(df['value']):
#    img = plt.imread('./data/'+v)
#    data['features']=np.append(data['features'], img)
#    data['labels']=np.append(data['labels'], df.iloc[k,3])
#
#with open("./data/data.p", "wb") as f:
#    pickle.dump(data, f)