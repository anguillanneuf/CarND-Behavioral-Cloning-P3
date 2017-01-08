#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:08:31 2016

@author: tz
"""
import numpy as np; import pandas as pd; import os; import pickle
import cv2; # import tensorflow as tf; # import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

def main():

    raw = pd.read_csv('./data/driving_log.csv')
    for i in raw.columns:
        if isinstance(raw[i][1], str):
            raw[i]=raw[i].map(str.strip)
    
    today = [i for i in os.listdir('./data/IMG')if i.find('2017_01_08')>=0]
    
    tot = len(today)*3*2
    h = 32; w = 64; ε = 0.08
    
    dat = {'features':np.zeros(shape=[tot,h,w,3]), 
           'labels':np.zeros(shape=tot), 
           'position': ['' for i in range(tot)], 
           'notes': ['org' for i in range(int(tot/2))]+
                    ['aug' for i in range(int(tot/2))]}
    
    for i, j in enumerate(today):

        # original/flipped images.
        img = cv2.cvtColor(cv2.imread('./data/IMG/'+j.strip()), 
                           cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        flipped = cv2.flip(img,1)
        
        # offset angles for left/right, flipped angles for flipped images
        pos_old = ('left' if j.find('left')>=0 else 
                   'right' if j.find('right')>=0 else 'center')    
        pos_new = ('right' if j.find('left')>=0 else 
                   'left' if j.find('right')>=0 else 'center')
        newpath = '/Users/tz/Documents/carnd/CarND-Behavioral-Cloning-P3/data/IMG/'
        old = raw[raw[pos_old]==newpath+j]['steering']
        adjusted = np.clip(old+ε if pos_old=='left' else 
                           old-ε if pos_old=='right' else old, -1., 1.)
        new = adjusted*-1.
        
        dat['features'][i] = img
        dat['labels'][i] = adjusted
        dat['position'][i] = pos_old
        dat['features'][i+int(tot/2)] = flipped
        dat['labels'][i+int(tot/2)] = new
        dat['position'][i+int(tot/2)] = pos_new
    
    with open("./data/additional.p", "wb") as f:
        pickle.dump(dat, f)
        

if __name__ == '__main__':
    main()