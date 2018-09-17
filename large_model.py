# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 08:54:06 2018

@author: nsde
"""
#%%
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, BatchNormalization, InputLayer
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, ELU
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import utils

from dlmnn.data.get_img_data import get_dataset
from dlmnn.LMNN import lmnn
from dlmnn.LMNNSequential import sequental_lmnn
from dlmnn.helper.argparser import lmnn_argparser

import numpy as np

#%%
if __name__ == '__main__':    
    # Input arguments
    args = lmnn_argparser()
    print(args)
    
    # Get some data
    x_train, y_train, x_test, y_test = get_dataset(args.dataset)
    input_shape=x_train.shape[1:]
    
    # Constants
    baseMapNum = 32
    weight_decay = 1e-4
    num_classes = 10
    
    # Normalize data
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)

    if args.model_type == 'softmax':
        y_train = utils.to_categorical(y_train,num_classes)
        y_test = utils.to_categorical(y_test,num_classes)
        
        model = Sequential()
        
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
    
        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))
    
        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))
    
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        model.fit(x_train, y_train,
                  epochs=100,
                  validation_data = [x_test, y_test],
                  batch_size=200)
        
    elif args.model_type == 'lmnn':
        model = lmnn()
        
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
    
        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))
    
        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))
    
        model.add(Flatten())
        
        model.compile(k=args.k, optimizer='adam', learning_rate=args.lr,
                      mu=args.mu, margin=args.margin)
        
        model.fit(x_train, y_train,
                  maxEpoch=args.n_epochs, 
                  batch_size=args.batch_size,
                  val_set=[x_test, y_test])
        
    elif args.model_type == 'sequential_lmnn':
        model = sequental_lmnn()
        # Input layer
        model.add(InputLayer, input_shape=x_train.shape[1:])
        # Conv block 1
        model.add(Conv2D, filters=baseMapNum, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))
        model.add(ELU)
        model.add(BatchNormalization)
        model.add(Conv2D, filters=baseMapNum, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))
        model.add(ELU)
        model.add(BatchNormalization)
        model.add(MaxPooling2D, pool_size=(2,2))
        model.add(Dropout, rate=0.2)
        # Conv block 2
        model.add(Conv2D, filters=2*baseMapNum, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))
        model.add(ELU)
        model.add(BatchNormalization)
        model.add(Conv2D, filters=2*baseMapNum, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))
        model.add(ELU)
        model.add(BatchNormalization)
        model.add(MaxPooling2D, pool_size=(2,2))
        model.add(Dropout, rate=0.3)
        # Conv_block 3
        model.add(Conv2D, filters=4*baseMapNum, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))
        model.add(ELU)
        model.add(BatchNormalization)
        model.add(Conv2D, filters=4*baseMapNum, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))
        model.add(ELU)
        model.add(BatchNormalization)
        model.add(MaxPooling2D, pool_size=(2,2))
        model.add(Dropout, rate=0.4)
        # Flatten
        model.add(Flatten)
        
        # Set model list
        model.set_model_list([8, 16], [25,])
        
        # Fit models in sequential order
        model.fit_sequential(x_train, y_train, 
                             epochs_pr_model=args.n_epochs, 
                             batch_size=args.batch_size, 
                             val_set=[x_test, y_test], 
                             k=args.k, 
                             optimizer='adam', 
                             learning_rate = args.lr, 
                             mu=args.mu, 
                             margin=args.margin)
    else:
        raise ValueError(args.model_type + ' is an uknown option for this script')