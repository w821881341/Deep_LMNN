#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:14:21 2018

@author: nsde
"""
#%%
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from .utility import batchifier

#%%
def compare_tN(A, B, k):
    ''' Compares a list of target neighbours in A to a list of target neighbours
        in B, and find the procentage that is similar '''
    assert A.shape == B.shape, 'assumes A and B to have equal shape'
    n = len(A)
    A = A[np.argsort(A[:,0])]
    B = B[np.argsort(B[:,0])]
    frac_same, tN = 0, [ ]
    for i, j in enumerate(range(0,n,k)):
        same_tn = np.intersect1d(A[j:j+k, 1], B[j:j+k, 1])
        n_tn = len(same_tn)
        frac_same += n_tn
        tN.append(np.array([n_tn*[i], same_tn]).T)
    frac_same /= n
    tN = np.vstack(tN)
    return frac_same, tN
 
#%%
def similarity_tN(list_tN, k):
    ''' Takes a list of tNs arrays and computes a similarity matrix between all
        pairs '''
    n = len(list_tN)
    sim_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i < j:
                sim_mat[i,j] = compare_tN(list_tN[i], list_tN[j], k)[0]
    return sim_mat.round(3)

#%%
def findTargetNeighbours(X, y, k, do_pca=True, name=None):
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
    name = ' for ' + name if name is not None else ''
    # Reshape data into 2D
    N = X.shape[0]
    X = np.reshape(X, (N, -1))
    if do_pca:
        pca = PCA(n_components = 0.95)
        X = pca.fit_transform(X)
    val = np.unique(y) 
    counter = 1
    tN_count = 0
    tN = np.zeros((N*k, 2), np.int32)
    # Iterate over each class
    for c in tqdm(val, desc='Finding target neighbours' + name):
        # Extract class c
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
    tN = tN[np.argsort(tN[:,0])] # sort by first observation
    return tN

#%%
def findImposterNeighbours(X, y, k, do_pca=True, name=None, batch_size=64):
    name = ' for ' + name if name is not None else ''
    # Reshape data into 2D
    N = X.shape[0]
    X = np.reshape(X, (N, -1))
    
    # Do pca feature reduction if wanted
    if do_pca:
        pca = PCA(n_components = 0.95)
        X = pca.fit_transform(X)
        
    # Loop over all points (in batches), and find closest neighbours with different labels
    imp = np.zeros((N*k, 2), np.int32)
    counter = 0
    for X_batch in tqdm(batchifier(X, batch_size), desc='Finding imposters' + name):
        dist = pairwise_distances(X_batch, X)
        n = dist.shape[0]
        idx = np.argsort(dist, axis=1)
        y_idx = y[idx]
        for i in range(n):
            imp_idx = np.where(y_idx[i,0] != y_idx[i])[0]
            imp[(counter+i)*k:(counter+i+1)*k] = np.vstack((k*[counter+i], 
                                                          idx[i,imp_idx[:k]])).T
        counter += batch_size
    return imp

#%%
def _weight_func(distances):
    """ Simple weight function, that accounts for over estimatation of performance
        when a KNN classifier is used to evaluate the same set it was trained on """
    N, d = distances.shape
    if distances[0,0] != 0:
        w=1.0/d*np.ones((N,d))
    else:
        w=np.concatenate([np.zeros((N,1)), 1.0/(N-1)*np.ones((N,d-1))], axis=1)
    return w

#%%
def knnClassifier(Xtest, Xtrain, ytrain, k):
    """ Special KNN-classifier that takes care of the case when Xtest==Xtrain,
        such that performance is not overestimated in this case.
    Arguments:
        Xtest:
        Xtrain:
        ytrain:
        k:
    """
    Ntest = Xtest.shape[0]
    Ntrain = Xtrain.shape[0]
    Xtest = np.reshape(Xtest, (Ntest, -1))
    Xtrain = np.reshape(Xtrain, (Ntrain, -1))
    same = np.array_equal(Xtest, Xtrain)
    if same: # if train and test is same, account for over estimation of
             # performance by one more neighbour and zero weight to the first
        classifier = KNeighborsClassifier(n_neighbors = k+1, weights=_weight_func, 
                                          algorithm='brute')
        classifier.fit(Xtrain, ytrain)
        pred = classifier.predict(Xtest)
    else:
        classifier = KNeighborsClassifier(n_neighbors = k, algorithm='brute')
        classifier.fit(Xtrain, ytrain)
        pred = classifier.predict(Xtest)
    return pred