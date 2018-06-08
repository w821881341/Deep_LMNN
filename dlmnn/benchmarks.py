#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 09:59:59 2018

@author: nsde
"""

#%%
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, InputLayer, Flatten
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.backend import function
import datetime
import numpy as np

from dlmnn.model.LMNN import lmnn
from dlmnn.data.get_img_data import get_dataset
from dlmnn.helper.embeddings import embedding_projector
from dlmnn.helper.utility import create_dir
from dlmnn.helper.neighbor_funcs import knnClassifier



#%%
if __name__ == '__main__':
    datasets = ['mnist', 'mnist_distorted', 'mnist_fashion', 
                'devanagari', 'olivetti', 'cifar10', 'cifar100']
    
    performance = np.zeros((4, len(datasets)), dtype=np.float32)
    pred1, pred2, pred3, pred4 = [ ], [ ], [ ], [ ]
    
    for c, d in enumerate(datasets):
        X_train, y_train, X_test, y_test = get_dataset(d)
        input_shape = X_train.shape[1:]
        n_class = len(np.unique(y_train))
        
        ############################## MODEL 1 ################################
        ############################## KNN classifier #########################
        pred = knnClassifier(X_test, X_train, y_train, k=3)
        acc = np.mean(pred==y_test)
        
        # Save results
        pred1.append(pred)
        performance[0, c] = acc
        
        ############################## MODEL 2 ################################
        ############################## Conv net ###############################
        run_id = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        create_dir('logs')
        create_dir('logs/'+run_id)
        cb = TensorBoard(log_dir='logs/'+run_id)
        
        # Create model
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer=Adam(lr=1e-4), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        # Fit model
        model.fit(X_train, to_categorical(y_train, 10), 
                  epochs=50, callbacks=[cb], batch_size=200,
                  validation_data=(X_test, to_categorical(y_test, n_class)))
        
        # Evaluate model
        pred = model.predict_classes(X_test)
        acc = model.evaluate(X_test, to_categorical(y_test, n_class))[1]
        
        # Save model
        model.save('logs/'+run_id+'/model.h5')
        
        # Save embeddings
        extractor = function([model.input], [model.layers[-2].output])
        X_trans = extractor([X_test])[0]
        embedding_projector(X_trans, 'logs/'+run_id, imgs=X_test, labels=y_test)
        
        # Save results
        pred2.append(pred)
        performance[1, c] = acc
        
        ############################## MODEL 3 ################################
        ############################## KNN classifier on conv features ########
        batch_size = 100
        n_batch_train = int(np.ceil(X_train.shape[0]/batch_size))
        n_batch_test = int(np.ceil(X_test.shape[0]/batch_size))
        X_train_trans = np.zeros((X_train.shape[0], 128))
        X_test_trans = np.zeros((X_test.shape[0], 128))
        
        # Feature extraction
        for b in range(n_batch_train):
            X_train_trans[batch_size*b:batch_size*(b+1)] = extractor(
                    [X_train[batch_size*b:batch_size*(b+1)]])[0]
        for b in range(n_batch_test):
            X_test_trans[batch_size*b:batch_size*(b+1)] = extractor(
                    [X_test[batch_size*b:batch_size*(b+1)]])[0]

        # Fit model        
        pred = knnClassifier(X_test, X_train, y_train, k=3)
        acc = np.mean(pred==y_test)
        
        # Save results
        pred3.append(pred)
        performance[2, c] = acc
        
        ############################## MODEL 4 ################################
        ############################## Deep lmnn ##############################
        # Create model
        model = lmnn()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.compile(k=3, optimizer='adam', learning_rate=1e-4, 
                      mu=0.5, margin=1)
        
        # Fit model
        model.fit(X_train, y_train, 
                  maxEpoch=50, batch_size=200,
                  val_set=[X_test, y_test], snapshot=5,
                  verbose=2)
        
        # Save embeddings
        model.save_embeddings(X_test, labels=y_test)
        
        # Evaluate model
        acc = model.evaluate(X_test, y_test, X_train, y_train)
        pred = model.predict(X_test, X_train, y_train)
        
        # Save results
        pred4.append(pred)
        performance[3, c] = acc
        
    np.save('performance', performance)
    np.save('predictions', [pred1, pred2, pred3, pred4])