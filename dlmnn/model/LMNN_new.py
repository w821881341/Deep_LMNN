#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:11:15 2018

@author: nsde
"""

#%%
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import get as get_optimizer

#%%
class lmnn(object):
    def __init__(self, session=None, dir_loc=None):
        # Initilize session and tensorboard dirs 
        self.session = tf.Session() if session is None else session
        self.dir_loc = './logs' if dir_loc is None else dir_loc
        self.train_writer = None
        self.val_writer = None
        
        # Initialize feature extractor
        self.extractor = Sequential()
        
        # Set variables for later training
        self.optimizer = self.verbose = self.margin = None
        
    def add(self, layer):
        self.extractor.add(layer)
        
    def build(self, optimizer='adam'):
        self.optimizer = get_optimizer(optimizer)
        pass    
    
    def fit(self, ):
        pass
    
    def transform(self, ):
        pass
        
    def predict(self, ):
        pass
    
    def evaluate(self, ):
        pass
    
#%%
if __name__ == '__main__':
    from tensorflow.python.keras.layers import Dense, InputLayer
    
    # Construct model
    model = lmnn()
    
    # Add feature extraction layers
    model.add(InputLayer(input_shape=(50,)))
    model.add(Dense(50, use_bias=False, kernel_initializer='identity'))
    
    # Build graph
    model.build()
    
    # Fit to data
    model.fit()