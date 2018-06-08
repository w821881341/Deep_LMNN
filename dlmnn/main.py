#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:36:27 2018

@author: nsde
"""
#%%
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, InputLayer, Flatten
from tensorflow.python.keras.utils import to_categorical

from dlmnn.model.LMNN import lmnn
from dlmnn.data.get_img_data import get_mnist, get_devanagari
from dlmnn.helper.embeddings import embedding_projector
from dlmnn.helper.utility import create_dir

#%%
def argparser( ):
    import argparse 
    parser = argparse.ArgumentParser(description='''Something''') 
    parser.add_argument('-t', action="store", dest='mtype', type=str, 
                        default='lmnn', help='''Dataset to use''')
    args = parser.parse_args() 
    args = vars(args) 
    return args['mtype']

#%% 
if __name__ == '__main__':
    mtype = argparser()
    
    # Get some data
    X_train, y_train, X_test, y_test = get_devanagari()
    input_shape=(32, 32, 1)
    
    # Construct and train normal conv net
    if mtype == 'conv':
        from tensorflow.python.keras.callbacks import TensorBoard
        from tensorflow.python.keras.backend import function
        import datetime
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
        model.fit(X_train, to_categorical(y_train, 10), 
                  epochs=50, callbacks=[cb],
                  validation_data=(X_test, to_categorical(y_test, 10)))
        
        # Save model
        model.save('logs/'+run_id+'/model.h5')
        
        # Save embeddings
        extractor = function([model.input], [model.layers[-2].output])
        X_trans = extractor([X_test])[0]
        embedding_projector(X_trans, 'logs/'+run_id, imgs=X_test, labels=y_test)
        
    # Construct and train lmnn net
    elif mtype == 'lmnn':
        model = lmnn()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.compile(k=3, optimizer='adam', learning_rate=1e-4, 
                      mu=0.5, margin=1)
        model.fit(X_train, y_train, 
                  maxEpoch=50, batch_size=100,
                  val_set=[X_test, y_test], snapshot=5,
                  verbose=2)
        model.save_embeddings(X_test, labels=y_test)
    
    elif mtype == 'lmnn_redo':
        from dlmnn.helper.neighbor_funcs import findTargetNeighbours
        model = lmnn()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.compile(k=3, optimizer='adam', learning_rate=1e-4, 
                      mu=0.5, margin=1)
        
        tN, tN_val = None, None
        for it in range(2):
            model.fit(X_train, y_train, 
                      maxEpoch=50, batch_size=200,
                      val_set=[X_test, y_test], snapshot=5,
                      verbose=2,
                      tN = tN, tN_val = tN_val)
            X_train_trans = model.transform(X_train)
            X_test_trans = model.transform(X_test)
            tN = findTargetNeighbours(X_train_trans, y_train, k=3)
            tN_val = findTargetNeighbours(X_test_trans, y_test, k=3)
            
            
            