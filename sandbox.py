#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:08:31 2016
@author: tz

"To remove a bias towards driving straight the training data includes a higher 
proportion of frames that represent road curves." (Source: Nvidia paper)
"""

#==============================================================================
# Original self-written generator.
#==============================================================================

import numpy as np

def gen(X, y, batch_size = 64):
    h, w, c, cursor = 32, 64, 3, 0
    
    XX, yy = np.zeros((batch_size, h,w,c)), np.zeros(batch_size)
    idx = np.random.permutation(np.arange(len(X))) 
    X, y = X[idx,], y[idx] # random shuffling.
    
    while True:
        for i in range(batch_size):
            cursor = cursor % len(X)
            XX[i], yy[i] = X[cursor,], y[cursor]
            cursor += 1
        yield XX, yy

        
#==============================================================================
# Code to merge self-collected data with Udacity-provided data.
#==============================================================================

import pickle; import matplotlib.pylot as plt

with open('./mydata/mydat.p', mode='rb') as f:
    mydat = pickle.load(f)
X_add, y_add = mydat['features'], mydat['labels']

with open('./mydata/train.p', mode='rb') as f:
    mytrain = pickle.load(f)
X_mytrain, y_mytrain = mytrain['features'], mytrain['labels']

with open('./mydata/test.p', mode='rb') as f:
    mytest = pickle.load(f)
X_mytest, y_mytest = mytest['features'], mytest['labels']

X_train = np.append(X_train_, X_mytrain, axis = 0)
y_train = np.append(y_train_, y_mytrain)
X_test = np.append(X_test_, X_mytest, axis = 0)
y_test = np.append(y_test_, y_mytest)

plt.hist(y_train, bins=50, color='#FF69B4')



#==============================================================================
# Failed attempt to train the model by incrementally increase the number of 
# near-zero training examples. 
#==============================================================================

#def limit(X, y, s = 700):
#    bad = [k for k,v in enumerate(y) if v in [0, -.25, .25]]
#    good = list(set(range(0, len(y)))-set(bad))
#    new = good + [bad[i] for i in np.random.randint(0,len(bad),s)]
#    X,y = X[new,], y[new]
#    return X, y
#    
#def main():
#    
#    if not os.path.exists("./outputs"): os.makedirs("./outputs")
#    
#    model = get_model()
#    
#    b = 64
#    
#    for i in range(8):
#        X, y = limit(X_train, y_train, 700 + i*100)
#        checkpointer = ModelCheckpoint("./outputs/model.hdf5", verbose=1, 
#                               save_best_only=True)
#        if i > 0:
#            model.fit_generator(gen.flow(X, y, batch_size=b),
#                        samples_per_epoch=len(X),
#                        nb_epoch=1,
#                        validation_data=gen.flow(X_val, y_val, batch_size=b),
#                        nb_val_samples=len(X_val),
#                        callbacks=[checkpointer]
#                        )
#    
#        else: 
#            model.fit_generator(gen.flow(X, y, batch_size=b),
#                        samples_per_epoch=len(X),
#                        nb_epoch=1,
#                        validation_data=gen.flow(X_val, y_val, batch_size=b),
#                        nb_val_samples=len(X_val),
#                        callbacks=[checkpointer])
#
#    model.save_weights("./outputs/model05.h5")
#    with open('./outputs/model05.json', 'w') as f:
#        json.dump(model.to_json(), f)
#


#==============================================================================
# Exploratory code to check if images are flipped correctly and to look at 
# the distribution of steering angles
#==============================================================================

import cv2; import matplotlib.pyplot as plt; import pandas as pd

raw = pd.read_csv('./data/driving_log.csv')
for i in raw.columns:
    if isinstance(raw[i][1], str):
        raw[i]=raw[i].map(str.strip)

# center image
img = cv2.imread('./data/' + raw.iloc[4042,0].strip())
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

img_small = cv2.resize(img, (128, 64))
plt.imshow(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))

# flipped image
flipped = cv2.flip(img, 1)
plt.imshow(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))

# steering angle distribution
print("Before flipping, zeros: {}".format(np.mean(raw['steering']==0)))
print("After flipping, none-zero: {}".format(np.sum(raw['steering']!=0)*3*2))

plt.hist(df['steering'], bins=50, color='#FF69B4')



#==============================================================================
# (archived) 1. flip and save, this doubles my training data (ε=0.25)
#==============================================================================
import numpy as np; import pandas as pd; import os; import cv2

# for each image, flip, save.
if not os.path.exists('./data/IMG_new/'):
    os.mkdir('./data/IMG_new/')

for j in os.listdir('./data/IMG'):
    flipped = cv2.flip(cv2.imread('./data/IMG/' + j.strip()),1)
    j = (j.replace('left', 'right') if j.find('left')>=0 else
          j.replace('right', 'left') if j.find('right')>=0 else j)
    cv2.imwrite('./data/IMG_new/'+j.strip()[:-4]+'_new.jpg', flipped)

# for df_new, update steering/steering_adjusted, update position/value, save.
raw = pd.read_csv('./data/driving_log.csv')
df = pd.melt(raw[['center', 'left', 'right', 'steering']], 
             id_vars=['steering'], var_name='position')

def epsilon(row, ε=.25):
    a = (row['steering']+ε if row['position']=='left' else
         row['steering']-ε if row['position']=='right' else row['steering'])
    return np.clip(a, -1, 1)
    
df['steering_adjusted'] = df.apply(epsilon,1,ε=0.25)

def flip(row):
    row['steering']*=-1
    row['steering_adjusted']*=-1
    row['position'] = ('right' if row['position']=='left' else
                       'left' if row['position']=='right' else 'center')
    t = 'IMG_new/'+row['value'].strip()[4:-4]+'_new.jpg'
    row['value'] = (t.replace('left', 'right') if t.find('left')>=0 else
                    t.replace('right', 'left') if t.find('right')>=0 else t)
    return row

df_new = df.apply(flip,1)

pd.concat([df, df_new]).to_csv('./data/driving_log_new.csv', index=False)

#==============================================================================
# (archived) 2. function to sample more balanced data/find corresponding images
#==============================================================================
import pandas as pd; import matplotlib.pyplot as plt

# unbalanced bins
df = pd.read_csv('./data/driving_log_new.csv')
plt.hist(df['steering_adjusted'], bins=100, color='#FF69B4')

# for each epoch, I can sample 1000 examples from steering=+/-.25/0
def generate_epoch(df, n_c=1000, n_l=1000, n_r=1000, ε=0.25):
    c = df[(df['steering_adjusted']==0)].sample(n_c)
    l = df[(df['steering_adjusted']==-ε)].sample(n_l)
    r = df[(df['steering_adjusted']==ε)].sample(n_r)    
    m = df[(df['steering_adjusted']!=0)&
           (df['steering_adjusted']!=ε)&
           (df['steering_adjusted']!=-ε)]
    mydf = pd.concat([c,l,r,m], ignore_index=True)
    return mydf.reindex(np.random.permutation(mydf.index))

mydf = generate_epoch(df,1000,1000,1000,0.25)
plt.hist(mydf['steering_adjusted'],bins=50,color='#FF69B4')

#==============================================================================
# (archived) 3. crate pickle
#==============================================================================
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt
import pickle

data = {'features':np.array([]), 'labels':np.array([])}
         
for k,v in enumerate(df['value']):
    img = plt.imread('./data/'+v)
    data['features']=np.append(data['features'], img)
    data['labels']=np.append(data['labels'], df.iloc[k,3])

with open("./data/data.p", "wb") as f:
    pickle.dump(data, f)