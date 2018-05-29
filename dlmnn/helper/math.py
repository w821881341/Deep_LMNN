#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:03:46 2018

@author: nsde
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

#%%
def plot_normal2D(mu, sigma):
    """ Plot contour curve from a normal distribution parametrized by mean mu
        and covariance sigma """    
    eigVal, eigVec = eig(sigma)
    eigVal = np.maximum(0, np.real(np.diag(eigVal)))
    t = np.linspace(0, 2*np.pi, 100)
    xy = np.array([np.cos(t), np.sin(t)])
    Txy = eigVec.dot(np.sqrt(eigVal)).dot(xy).T + mu
    plt.plot(Txy[:,0], Txy[:,1], 'b-')

#%%
def rand_posdef_mat(d):
    ''' Generate random positive semi-definite matrix of size d x d '''
    mat = np.random.normal(size=(d,d))
    mat = np.dot(mat, mat)
    return mat

#%%
def random_not_in_sampler(s, N_range, array):
    """ Samples s different values from the range [0, N_range] that are not
        in array. Use with coution, can be stuck in a infinit loop"""
    samples = [ ]
    count = 0
    while count < s:
        i = np.random.randint(N_range)
        if i not in array and i not in samples:
            samples.append(i)
            count += 1
    return np.array(samples)

#%%
def weight_func(distances):
    """ Simple weight function, that accounts for over estimatation of performance
        when a KNN classifier is used to evaluate the same set it was trained on """
    N, d = distances.shape
    if distances[0,0] != 0:
        return 1.0/d*np.ones((N,d))
    else:
        return np.concatenate([np.zeros((N,1)), 1.0/(N-1)*np.ones((N,d-1))], axis=1)