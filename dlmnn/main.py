#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:36:27 2018

@author: nsde
"""
#%%
from dlmnn.helper.layers import  InputLayer, Flatten, Conv2D, MaxPool2D, Dense 
from dlmnn.model.LMNN import lmnn
from dlmnn.data.get_img_data import get_dataset

#%%
def argparser( ):
    import argparse 
    parser = argparse.ArgumentParser(description='''Something''') 
    parser.add_argument('-k', action="store", dest='k', type=int,
                        default=1, help='''Number of neighbours''')
    parser.add_argument('-e', action="store", dest='e', type=int,
                        default=10, help='''epochs''')
    parser.add_argument('-b', action="store", dest='b', type=int,
                        default=100, help='''batch size''')
    parser.add_argument('-m', action="store", dest='m', type=float,
                        default=1.0, help='''margin''')
    parser.add_argument('-w', action="store", dest='w', type=float,
                        default=0.5, help='''mu''')
    parser.add_argument('-r', action="store", dest='r', type=str,
                        default='res', help='''where to store final results''')
    parser.add_argument('-n', action="store", dest='n', type=bool,
                        default=False, help='''L2 normalize features''')
    args = parser.parse_args() 
    args = vars(args) 
    return args

#%% 
if __name__ == '__main__':
    # Get input arguments
    args = argparser()
    print(args)
    
    # Get some data
    X_train, y_train, X_test, y_test = get_dataset('olivetti')
    input_shape=X_train.shape[1:]
    
    # Make model
    model = lmnn()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(16, kernel_size=(3,3), activation='linear', padding='same'))
    model.add(Conv2D(32, kernel_size=(3,3), activation='linear', padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    
    # Compile model
    model.compile(k=args['k'], optimizer='adam', learning_rate=1e-4, 
                  mu=args['w'], margin=args['m'], normalize=args['n'])
    
    X_trans1 = model.transform(X_test)
    
    # Fit model and save result
    model.fit(X_train, y_train, 
              maxEpoch=args['e'], batch_size=args['b'],
              val_set=[X_test, y_test], snapshot=5,
              verbose=2)
    model.save_embeddings(X_test, labels=y_test)