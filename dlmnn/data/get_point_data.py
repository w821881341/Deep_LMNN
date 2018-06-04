#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 08:56:21 2018

@author: nsde
"""
#%%
import numpy as _np
import matplotlib.pyplot as _plt
from sklearn.model_selection import train_test_split as _train_test_split

#%%
def test_set1(size = 50, plot = False):
    
    mu = [(1,100), (2,100), (3,100), (4,100)]
    std = _np.array([[0.05, 0], [0, 50]])
    
    X = [ ]
    y = [ ]
    for i in range(len(mu)):
        for _ in range(size):
            X.append(_np.random.multivariate_normal(mean = mu[i], cov = std))
            y.append(i)       
    X = _np.array(X)
    y = _np.array(y)
    
    if plot:
        col = ['red', 'blue', 'green', 'orange']
        _plt.figure()
        _plt.subplot(1,2,1)
        for i in range(X.shape[0]):
            _plt.plot(X[i,0], X[i,1], '.', color=col[y[i]])
        _plt.axis('equal')
    
        _plt.subplot(1,2,2)
        for i in range(X.shape[0]):
            _plt.plot(X[i,0], X[i,1], '.', color=col[y[i]])
    
    X_train, X_test, y_train, y_test = _train_test_split(X, y, test_size=0.1)    
    return X_train, y_train, X_test, y_test

#%%
if __name__ == '__main__':
    set1 = test_set1(size=50, plot=True)

    