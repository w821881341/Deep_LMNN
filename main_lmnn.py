#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:36:27 2018

@author: nsde
"""
#%%
from dlmnn import lmnn
from dlmnn.helper.argparser import lmnn_argparser
from dlmnn.helper.layers import InputLayer, Flatten, Conv2D, MaxPool2D, \
                                Dense, LeakyReLU
from dlmnn.data.get_img_data import get_dataset


#%% 
if __name__ == '__main__':
    # Get input arguments
    args = lmnn_argparser()
    print(args)
    
    # Get some data 
    X_train, y_train, X_test, y_test = get_dataset('cifar10') 
    input_shape=X_train.shape[1:]
    
    # Make model 
    model = lmnn() 
    model.add(InputLayer(input_shape=input_shape)) 
    
    model.add(Conv2D(16, kernel_size=(3,3), padding='same')) 
    model.add(LeakyReLU(alpha=0.3)) 
    model.add(MaxPool2D(pool_size=(2,2))) 
    
    model.add(Conv2D(32, kernel_size=(3,3), padding='same')) 
    model.add(LeakyReLU(alpha=0.3)) 
    model.add(MaxPool2D(pool_size=(2,2))) 
    
    model.add(Conv2D(64, kernel_size=(3,3), padding='same')) 
    model.add(LeakyReLU(alpha=0.3)) 
    model.add(MaxPool2D(pool_size=(2,2))) 
    
    model.add(Flatten()) 
    model.add(Dense(128)) 
    model.add(LeakyReLU(alpha=0.3)) 
    
    # Compile model 
    model.compile(k=args.k, optimizer='adam', learning_rate=args.lr,  
                  mu=args.mu, margin=args.margin)
     
    # Fit model and save result 
    model.fit(X_train, y_train,  
              maxEpoch=args.n_epochs, 
              batch_size=args.batch_size, 
              val_set=[X_test, y_test], 
              snapshot=5) 