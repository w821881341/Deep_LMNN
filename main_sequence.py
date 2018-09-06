#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 08:37:00 2018

@author: nsde
"""

#%%
import tensorflow as tf
from dlmnn import lmnn
from dlmnn.helper.argparser import lmnn_argparser
from dlmnn.helper.layers import InputLayer, Flatten, Conv2D, MaxPool2D, \
                                Dense, LeakyReLU, layerlist
from dlmnn.data.get_img_data import get_dataset
from dlmnn.helper.neighbor_funcs import findTargetNeighbours, compare_tN, similarity_tN

#%% 
if __name__ == '__main__':
    # Get input arguments
    args = lmnn_argparser()
    print(args)
    
    # Get some data 
    X_train, y_train, X_test, y_test = get_dataset('cifar10') 
    input_shape=X_train.shape[1:]
          
    # Get a layer list we can retrive layers from
    layers = layerlist()
    layers.add(InputLayer, input_shape=input_shape)
    layers.add(Conv2D, filters=16, kernel_size=(3,3), padding='same')
    layers.add(LeakyReLU, alpha=0.3)
    layers.add(MaxPool2D, pool_size=(2,2))
    layers.add(Conv2D, filters=32, kernel_size=(3,3), padding='same')
    layers.add(LeakyReLU, alpha=0.3)
    layers.add(MaxPool2D, pool_size=(2,2))
    layers.add(Conv2D, filters=64, kernel_size=(3,3), padding='same')
    layers.add(LeakyReLU, alpha=0.3)
    layers.add(MaxPool2D, pool_size=(2,2))
    layers.add(Flatten)
    layers.add(Dense, units=128)
    layers.add(LeakyReLU, alpha=0.3)
    
    # Sequence of models
    l1 = [0,1,2,3,10,11,12]
    l2 = [0,1,2,3,4,5,6,10,11,12]
    l3 = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    all_l = [l1, l2, l3]
    
    # Do sequence of model fittings
    tN = [findTargetNeighbours(X_train, y_train, args.k)]
    tN_val = [findTargetNeighbours(X_test, y_test, args.k)]
    for layer_list in all_l:        
        # Add layers to model
        model = lmnn()
        for l in layer_list: 
            model.add(layers.get_layer(l))
        
        # Compile model
        model.compile(k=args.k, optimizer='adam', learning_rate=args.lr,  
                       mu=args.weight, margin=args.margin)
     
        # Fit model
        model.fit(X_train, y_train,  
                  maxEpoch=args.n_epochs, 
                  batch_size=args.batch_size, 
                  val_set=[X_test, y_test], 
                  tN=tN[-1], tN_val=tN_val[-1],
                  snapshot=5)
            
        # Transform into new feature space
        X_train_trans = model.transform(X_train)
        X_test_trans = model.transform(X_test)
    
        # Compute target neighbours in feature space
        tN.append(findTargetNeighbours(X_train_trans, y_train, args.k))
        tN_val.append(findTargetNeighbours(X_test_trans, y_test, args.k))
        
        # Save embeddings
        model.save_embeddings(data = X_test)
        
        # Reset graph
        tf.reset_default_graph()
        
    # Compare tNs
    for i in range(1,len(all_l)+1):
        print('tN similarity between 0 and ' + 
              str(i) + ': ', compare_tN(tN[0], tN[i])[0])