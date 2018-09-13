#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:14:23 2018

@author: nsde
"""

#%%
from dlmnn import sequental_lmnn
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
    
    # Add layers to model
    model = sequental_lmnn()
    model.add(InputLayer, input_shape=input_shape)
    model.add(Conv2D, filters=16, kernel_size=(3,3), padding='same')
    model.add(LeakyReLU, alpha=0.3)
    model.add(MaxPool2D, pool_size=(2,2))
    model.add(Conv2D, filters=32, kernel_size=(3,3), padding='same')
    model.add(LeakyReLU, alpha=0.3)
    model.add(MaxPool2D, pool_size=(2,2))
    model.add(Conv2D, filters=64, kernel_size=(3,3), padding='same')
    model.add(LeakyReLU, alpha=0.3)
    model.add(MaxPool2D, pool_size=(2,2))
    model.add(Flatten)
    model.add(Dense, units=128)
    model.add(LeakyReLU, alpha=0.3)
    
    model.set_model_list([3, 6, ], [10, 11, 12])
    
    # Fit model
    model.fit_sequential(X_train, y_train, 
                         epochs_pr_model=args.n_epochs, 
                         batch_size=args.batch_size, 
                         verbose=2, 
                         snapshot=5, 
                         val_set=[X_test, y_test], 
                         k=args.k, 
                         optimizer='adam', 
                         learning_rate=args.lr, 
                         mu=args.mu, 
                         margin=args.margin)