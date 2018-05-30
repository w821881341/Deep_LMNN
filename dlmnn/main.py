#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:36:27 2018

@author: nsde
"""
#%%
from dlmnn.helper.argparser import lmnn_argparser
from dlmnn.data.get_img_data import get_mnist
from dlmnn.helper.tf_funcs import KerasTransformer

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, \
                                            InputLayer, Flatten
from tensorflow.python.keras.utils import to_categorical

#%% 
if __name__ == '__main__':
    # Get input arguments
    args = lmnn_argparser()
    
    # Get some data
    X_train, y_train, X_test, y_test = get_mnist()
    X_train = X_train[:30000]
    y_train = to_categorical(y_train[:30000], 10)

    # Construct keras feature extractor
#    kc = KerasTransformer(input_shape=(64, 64, 1))
#    kc.add(Conv2D(32, (5,5), activation='relu', padding='same'))
#    kc.add(MaxPool2D((2,2), strides=(2,2)))
#    kc.add(Conv2D(64, (5,5), activation='relu', padding='same'))
#    kc.add(Dense(1024, activation='relu'))
#    kc.add(Dropout(0.4))
#    kc.add(Dense(10))
#    trans_func = kc.get_function()
    
    # Construct normal neural network
    nn = Sequential()
    nn.add(InputLayer(input_shape=(28, 28, 1)))
    nn.add(Conv2D(32, (5,5), activation='relu', padding='same'))
    nn.add(MaxPool2D((2,2), strides=(2,2)))
    nn.add(Conv2D(64, (5,5), activation='relu', padding='same'))
    nn.add(Flatten())
    nn.add(Dense(1024, activation='relu'))
    nn.add(Dropout(0.4))
    nn.add(Dense(10, activation='softmax'))
    nn.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    nn.fit(x=X_train, y=y_train, epochs=args['ne'])
#
#    
#    # Define class
#    model = LMNN(tf_transformer = keras_mahalanobisTransformer,
#                 margin = args['m'], 
#                 dir_loc = 'results',
#                 optimizer=args['o'], 
#                 verbose = args['v'])
#    
#    # Fit transformer
#    model.fit(X_train, y_train, k=2, mu=args['mu'], maxEpoch=args['ne'], 
#             batch_size=args['bs'], learning_rate=args['lr'], 
#             val_set=[X_test, y_test], snapshot=args['ss'])