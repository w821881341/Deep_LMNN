# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 08:54:06 2018

@author: nsde
"""
#%%
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, ELU
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras import utils
from dlmnn.data.get_img_data import get_dataset

from dlmnn.model.LMNN import lmnn
from dlmnn.model.LMNNredo import lmnnredo

import numpy as np

#%%
def argparser( ):
    import argparse 
    parser = argparse.ArgumentParser(description='''Something''') 
    parser.add_argument('-t', action="store", dest='mtype', type=str,
                        default='lmnnredo', help='''model to use''')
    parser.add_argument('-k', action="store", dest='k', type=int,
                        default=1, help='''Number of neighbours''')
    parser.add_argument('-e', action="store", dest='e', type=int,
                        default=10, help='''epochs''')
    parser.add_argument('-b', action="store", dest='b', type=int,
                        default=100, help='''batch size''')
    parser.add_argument('-m', action="store", dest='m', type=float,
                        default=1.0, help='''margin''')
    parser.add_argument('-l', action="store", dest='l', type=float,
                        default=1e-4, help='''learning rate''')
    parser.add_argument('-w', action="store", dest='w', type=float,
                        default=0.5, help='''mu''')
    parser.add_argument('-r', action="store", dest='r', type=str,
                        default='res', help='''where to store final results''')
    parser.add_argument('-n', action="store", dest='n', type=bool,
                        default=True, help='''L2 normalize features''')
    args = parser.parse_args() 
    args = vars(args) 
    return args

#%%
if __name__ == '__main__':
    args = argparser()
    
    # Get some data
    x_train, y_train, x_test, y_test = get_dataset('cifar10')
    input_shape=x_train.shape[1:]
    
    baseMapNum = 32
    weight_decay = 1e-4
    num_classes = 10
    
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)

    
    if args['mtype'] == 'conv':
        y_train = utils.to_categorical(y_train,num_classes)
        y_test = utils.to_categorical(y_test,num_classes)
        
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
                  epochs=args['e'],
                  validation_data = [x_test, y_test],
                  batch_size=args['b'])
        
    elif args['mtype'] == 'lmnn':
         tN, tN_val=np.load('targetNeighbours.npy')
        
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
         model.add(Activation('relu'))
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
         
         model.compile(k=args['k'], normalize=args['n'], margin=args['m'])
         
         model.fit(x_train, y_train, 
                   maxEpoch=args['e'], 
                   val_set=[x_test, y_test],
                   batch_size=args['b'],
                   snapshot=5,
                   tN=tN, tN_val=tN_val)
         
    elif args['mtype'] == 'lmnnredo':
        
         model = lmnnredo()
         
         model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
         model.add(ELU())
         model.add(BatchNormalization())
         model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
         model.add(ELU())
         model.add(BatchNormalization())
         model.add(MaxPooling2D(pool_size=(2,2)))
         model.add(Dropout(0.2))
         
         model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
         model.add(Activation('relu'))
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
         
         model.compile(k=args['k'], normalize=args['n'], margin=args['m'])
         
         model.fit(x_train, y_train, 
                   maxEpoch=args['e'], 
                   val_set=[x_test, y_test],
                   batch_size=args['b'],
                   snapshot=5,
                   redo_step=5)