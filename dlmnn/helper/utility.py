# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:23:37 2017

@author: nsde
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, sys

#%%
def colorise(string, color='green'):
    if color=='green': col='0;32;40m'
    elif color=='blue': col='0;34;40m'
    elif color=='red': col='0;31;40m'
    begin = '\x1b['
    end = '\033[0m'
    return begin + col + string + end

#%%
def progressBar(value, endvalue, name = 'Process', bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0}: [{1}] {2}%".format(name, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
    
#%%
def plot_normal2D(mu, sigma):
    from scipy.linalg import eig
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
def random_not_in_sampler(size, N_range, array):
    samples = [ ]
    count = 0
    while count < size:
        i = np.random.randint(N_range)
        if i not in array and i not in samples:
            samples.append(i)
            count += 1
    return np.array(samples)

#%%
def get_dir(file):
    return os.path.dirname(os.path.realpath(file))

#%%
def get_optimizer(name):
    ''' Returns a tensorflow optimizer based on name '''
    try:
        optimizer = {'adam':     tf.train.AdamOptimizer,
                     'sgd':      tf.train.GradientDescentOptimizer,
                     'momentum': tf.train.MomentumOptimizer
                     }[name]
        return optimizer
    except KeyError:
        raise Exception(name + ' is a invalid option to input optimizer')

#%%
def adjust_learning_rate(alpha, loss_new, loss_old):
    ''' Function for adjusting the learning rate while training '''
    out_alpha = alpha*1.01 if loss_new <= loss_old else alpha*0.5
    return out_alpha

#%%
if __name__ == '__main__':
    print(get_dir(__file__))