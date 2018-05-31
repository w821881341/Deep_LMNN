#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:14:21 2018

@author: nsde
"""
#%%
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from dlmnn.helper.utility import progressBar

#%%
def findTargetNeighbours(X, y, k, do_pca=True, name=''):
    ''' Numpy/sklearn implementation to find target neighbours for large 
        datasets. This function cannot use the GPU and thus runs on the CPU,
        but instead uses an advance ball-tree method.
    Arguments:
        X: N x ?, metrix or tensor with data
        y: N x 1, vector with labels
        k: scalar, number of target neighbours to find
        do_pca: bool, if true then the data will first be projected onto
            a pca-space which captures 95% of the variation in data
        name: str, name of the dataset
    Output:
        tN: (N*k) x 2 matrix, with target neighbour index. 
    '''
    print(50*'-')
    # Reshape data into 2D
    N = X.shape[0]
    X = np.reshape(X, (N, -1))
    if do_pca:
        print('Doing PCA')
        pca= PCA(n_components = 0.95)
        X = pca.fit_transform(X)
    val = np.unique(y) 
    counter = 1
    tN_count = 0
    tN = np.zeros((N*k, 2), np.int32)
    # Iterate over each class
    for c in val:
        progressBar(counter, len(val), 
                    name='Finding target neighbours for '+name)
        idx = np.where(y==c)[0]
        n_c = len(idx)
        x = X[idx]
        # Find the nearest neighbours
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute')
        nbrs.fit(x)
        _, indices = nbrs.kneighbors(x)
        for kk in range(1,k+1):
            tN[tN_count:tN_count+n_c,0] = idx[indices[:,0]]
            tN[tN_count:tN_count+n_c,1] = idx[indices[:,kk]]
            tN_count += n_c
        counter += 1
    print('')
    print(50*'-')
    return tN