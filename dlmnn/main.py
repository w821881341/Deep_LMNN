#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:36:27 2018

@author: nsde
"""
#%%
from dlmnn.model.LMNN import lmnn
from dlmnn.model.LMNN_new import lmnn as lmnn_new
from dlmnn.helper.argparser import lmnn_argparser
from dlmnn.data.get_img_data import get_mnist
from dlmnn.helper.tf_funcs import KerasTransformer

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, InputLayer, Flatten
from tensorflow.python.keras.utils import to_categorical

#%% 
if __name__ == '__main__':
    # Get input arguments
    args = lmnn_argparser()
    
    # Get some data
    X_train, y_train, X_test, y_test = get_mnist()
    
    # Construct and train normal conv net
#    nn = Sequential()
#    nn.add(InputLayer(input_shape=(28, 28, 1)))
#    nn.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
#    nn.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
#    nn.add(MaxPool2D(pool_size=(2,2)))
#    nn.add(Flatten())
#    nn.add(Dense(128, activation='relu'))
#    nn.add(Dense(10, activation='softmax'))
#    nn.compile(optimizer=Adam(lr=1e-4), 
#               loss='categorical_crossentropy', 
#               metrics=['accuracy'])
#    nn.fit(X_train, to_categorical(y_train, 10), 
#           epochs=10,
#           validation_data=(X_test, to_categorical(y_test, 10)))

    # Construct keras feature extractor
#    kt = KerasTransformer(input_shape=(28, 28, 1))
#    kt.add(InputLayer(input_shape=(28, 28, 1)))
#    kt.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
#    kt.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
#    kt.add(MaxPool2D(pool_size=(2,2)))
#    kt.add(Flatten())
#    kt.add(Dense(128, activation='relu'))
#    trans_func = kt.get_function()

   
    # Define class
#    model = lmnn(tf_transformer = trans_func,
#                 margin = args['m'], 
#                 dir_loc = 'results',
#                 optimizer='adam', 
#                 verbose = args['v'])
#    
#
#    # Fit transformer
#    model.fit(X_train, y_train, k=2, mu=args['mu'], maxEpoch=50,#args['ne'], 
#              batch_size=args['bs'], learning_rate=args['lr'], 
#              val_set=[X_test, y_test], snapshot=5)
    
    model = lmnn_new()
    model.add(InputLayer(input_shape=(28, 28, 1)))
    model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    
    model.build(k=2, optimizer='adam', learning_rate = 1e-4,
                mu=0.5, margin=1)
    
    model.fit(X_train, y_train, maxEpoch=50, batch_size=100,
              val_set=[X_test, y_test], snapshot=5)
    