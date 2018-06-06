#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 08:53:59 2018

@author: nsde
"""
#%%
from dlmnn.helper.utility import get_dir as _get_dir
from dlmnn.helper.utility import create_dir as _create_dir

import os as _os
import numpy as _np
import urllib as _urllib

#%%
def get_devanagari():
    url1 = 'https://raw.githubusercontent.com/sknepal/DHDD_CSV/master/train.csv'
    url2 = 'https://raw.githubusercontent.com/sknepal/DHDD_CSV/master/test.csv'
    direc = _get_dir(__file__)
    _create_dir(direc+'/data_files')
    for url in [url1, url2]:
        file_name = 'devanagari_'+url.split('/')[-1]
        if not _os.path.isfile(direc+'/data_files/'+file_name):
            print('Downloading the ' + file_name + ' dataset')
            _urllib.request.urlretrieve(url, direc+'/data_files/'+file_name)
    
    train = _np.genfromtxt(direc+'/data_files/'+'devanagari_train.csv', delimiter=',')
    test = _np.genfromtxt(direc+'/data_files/'+'devanagari_test.csv', delimiter=',')
    X_train = _np.reshape(train[:,1:], (-1, 32, 32, 1))
    X_test = _np.reshape(test[:,1:], (-1, 32, 32, 1))
    y_train = _np.reshape(train[:,0], (-1, ))
    y_test = _np.reshape(test[:,0], (-1, ))
    return X_train, y_train, X_test, y_test     
    
#%%
def get_mnist():
    """ Downloads mnist from internet """
    url = "https://s3.amazonaws.com/img-datasets/mnist.npz"
    direc = _get_dir(__file__)
    _create_dir(direc+'/data_files')
    file_name = url.split('/')[-1]
    if not _os.path.isfile(direc+'/data_files/'+file_name):
        print('Downloading the mnist dataset (11.5MB)')
        _urllib.request.urlretrieve(url, direc+'/data_files/'+file_name)
    
    data = _np.load(direc+'/data_files/'+file_name)
    X_train, y_train = data['x_train'], data['y_train']
    X_test, y_test = data['x_test'], data['y_test']
    X_train = _np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = _np.reshape(X_test, (X_test.shape[0], 28, 28, 1))
    return X_train, y_train, X_test, y_test

#%%
def get_mnist_distorted():
    """ Downloads distorted mnist from internet """
    url = "https://s3.amazonaws.com/lasagne/recipes/datasets/mnist_cluttered_60x60_6distortions.npz"
    direc = _get_dir(__file__)
    _create_dir(direc+'/data_files')
    file_name = url.split('/')[-1]
    if not _os.path.isfile(direc+'/data_files/'+file_name):
        print('Downloading the distorted mnist dataset (43MB)')
        _urllib.request.urlretrieve(url, direc+'/data_files/'+file_name)
    
    data = _np.load(direc+'/data_files/'+file_name)
    X_train, y_train = data['x_train'], data['y_train']
    X_test, y_test = data['x_test'], data['y_test']
    X_train = _np.reshape(X_train, (X_train.shape[0], 60, 60, 1))
    X_test = _np.reshape(X_test, (X_test.shape[0], 60, 60, 1))
    return X_train, y_train, X_test, y_test

#%%
def get_olivetti():
    from sklearn.datasets import fetch_olivetti_faces
    direc = _get_dir(__file__)
    _create_dir(direc+'/data_files')
    obj = fetch_olivetti_faces(data_home=direc+'/data_files')
    X = obj['data']
    y = obj['target']
    
    # Split data
    X_train = _np.zeros((280, 4096), dtype=_np.float32)
    X_test = _np.zeros((120, 4096), dtype=_np.float32)
    y_train = _np.zeros((280, ), dtype=_np.int64)
    y_test = _np.zeros((120, ), dtype=_np.int64)
    for i in range(40):
        rand_idx=_np.random.permutation(range(10*i,10*(i+1)))
        X_train[7*i:7*(i+1)]=X[rand_idx[:7]]
        X_test[3*i:3*(i+1)]=X[rand_idx[7:]]
        y_train[7*i:7*(i+1)]=y[rand_idx[:7]]
        y_test[3*i:3*(i+1)]=y[rand_idx[7:]]
    # Permute data
    idx = _np.random.permutation(280)
    X_train = X_train[idx]
    y_train = y_train[idx]
    idx = _np.random.permutation(120)
    X_test = X_test[idx]
    y_test = y_test[idx]
    
    # Reshape to image format
    X_train = _np.reshape(X_train, (280, 64, 64, 1))
    X_test = _np.reshape(X_test, (120, 64, 64, 1))
    
    return X_train, y_train, X_test, y_test

#%%
def get_fashion_mnist():
    import tensorflow 
    (X_train, y_train), (X_test, y_test) = \
        tensorflow.keras.datasets.fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test

#%%
if __name__ == '__main__':
    devanagari = get_devanagari()
    mnist = get_mnist()
    mnist_distorted = get_mnist_distorted()
    olivetti = get_olivetti()
    fashion = get_fashion_mnist()
