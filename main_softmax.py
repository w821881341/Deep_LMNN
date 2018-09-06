#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 10:33:14 2018

@author: nsde
"""

#%%
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical
from dlmnn.helper.layers import InputLayer, Flatten, Conv2D, MaxPool2D, \
                                Dense, LeakyReLU                         
from dlmnn.data.get_img_data import get_dataset


#%%
if __name__ == '__main__':
    # Get some data 
    X_train, y_train, X_test, y_test = get_dataset('cifar10') 
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    input_shape=X_train.shape[1:]
    
    # Construct model
    model = Sequential()
    
    # Add layers
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
    model.add(Dense(10, activation='softmax'))
    
    # Compile model
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Fit model
    model.fit(X_train, y_train, 128, epochs=150,
              validation_data=[X_test, y_test])