#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:36:27 2018

@author: nsde
"""
#%%
from dlmnn.model.LMNN import lmnn
from dlmnn.data.get_img_data import get_mnist

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, InputLayer, Flatten
from tensorflow.python.keras.utils import to_categorical

#%%
def argparser( ):
    import argparse 
    parser = argparse.ArgumentParser(description='''Something''') 
    parser.add_argument('-t', action="store", dest='mtype', type=str, 
                        default='conv', help='''Dataset to use''')
    args = parser.parse_args() 
    args = vars(args) 
    return args['mtype']

#%% 
if __name__ == '__main__':
    mtype = argparser()
    
    # Get some data
    X_train, y_train, X_test, y_test = get_mnist()
    
    import tensorflow as tf
    with tf.device("/cpu:0"):
        # Construct and train normal conv net
        if mtype == 'conv':
            model = Sequential()
            model.add(InputLayer(input_shape=(28, 28, 1)))
            model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
            model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
            model.add(MaxPool2D(pool_size=(2,2)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer=Adam(lr=1e-4), 
                          loss='categorical_crossentropy', 
                          metrics=['accuracy'])
            model.fit(X_train, to_categorical(y_train, 10), 
                      epochs=20,
                      validation_data=(X_test, to_categorical(y_test, 10)))
        # Construct and train lmnn net
        elif mtype == 'lmnn':
            model = lmnn()
            model.add(InputLayer(input_shape=(28, 28, 1)))
            model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
            model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
            model.add(MaxPool2D(pool_size=(2,2)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.build(k=2, optimizer='adam', learning_rate=1e-4, 
                        mu=0.5, margin=1)
            model.fit(X_train, y_train, 
                      maxEpoch=50, batch_size=100,
                      val_set=[X_test, y_test], snapshot=5)
        