# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 08:54:06 2018

@author: nsde
"""
#%%
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras import utils
from dlmnn.data.get_img_data import get_dataset

from dlmnn.model.LMNN import lmnn

import numpy as np

#%%
def argparser( ):
    import argparse 
    parser = argparse.ArgumentParser(description='''Something''') 
    parser.add_argument('-t', action="store", dest='mtype', type=str,
                        default='conv', help='''model to use''')
    args = parser.parse_args() 
    args = vars(args) 
    return args

#%%
if __name__ == '__main__':
    args = argparser()
    
    # Get some data
    x_train, y_train, x_test, y_test = get_dataset('cifar10')
    input_shape=x_train.shape[1:]
    
    baseMapNum=32
    weight_decay = 1e-4
    num_classes = 10
    
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)
    y_train = utils.to_categorical(y_train,num_classes)
    y_test = utils.to_categorical(y_test,num_classes)
    
    if args['mtype'] == 'conv':
        model = Sequential()
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
    
        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))
    
        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))
    
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        
        optim = RMSprop(lr=0.001,decay=1e-6)
        model.compile(loss='categorical_crossentropy',
            optimizer=optim,
            metrics=['accuracy'])
        
        model.fit(x_train, y_train,
                  epochs=100,
                  validation_data = [x_test, y_test])
        
    elif args['mtype'] == 'lmnn':
         model = lmnn()
         
         model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
         model.add(Activation('relu'))
         model.add(BatchNormalization())
         model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
         model.add(Activation('relu'))
         model.add(BatchNormalization())
         model.add(MaxPooling2D(pool_size=(2,2)))
         model.add(Dropout(0.2))
         
         model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
         model.add(Activation('relu'))
         model.add(BatchNormalization())
         model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
         model.add(Activation('relu'))
         model.add(BatchNormalization())
         model.add(MaxPooling2D(pool_size=(2,2)))
         model.add(Dropout(0.3))
         
         model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
         model.add(Activation('relu'))
         model.add(BatchNormalization())
         model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
         model.add(Activation('relu'))
         model.add(BatchNormalization())
         model.add(MaxPooling2D(pool_size=(2,2)))
         model.add(Dropout(0.4))
        
         model.add(Flatten())
         
         model.compile(k=3, normalize=True, margin=1)
         
         model.fit(x_train, y_train, maxEpoch=100, val_set=[x_test, y_test],
                   snapshot=5)
         
        