# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:23:37 2017

@author: nsde
"""

import tensorflow as tf
import os, sys
import numpy as np

#%%
class batch_builder():
    def __init__(self, tN, imp, k, batch_size):
        # Append imposters to target neighbour structure
        imp_r = imp[:,1].repeat(k) # repeat the imposters
        imp_r = imp_r.reshape((-1,k,k)).transpose((0,2,1)).reshape((-1,k))
        self.combined = np.hstack((tN, imp_r))
        
        # Shuffel
        self.combined = np.random.permutation(self.combined)
        
        # Constants
        self.batch_size = batch_size
        self.counter = 0
        self.n = tN.shape[0]
        
    def __next__(self):
        batch_idx = self.combined[self.counter:self.counter+self.batch_size]
        self.counter += self.batch_size
        idx, inv_idx = np.unique(batch_idx, return_inverse=True)
        inv_idx = np.reshape(inv_idx, (-1, 5))[:,:2]
        return idx, inv_idx
        
    def __iter__(self):
        return self
    
    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

#%%
class batchifier():
    ''' Small iterator that will cut the input data into smaller batches. Can
        then be used in a for-loop like:
            for x_batch in batchifier(X, 100):
                # x_batch.shape[0] = 100            
    '''
                
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size    
        self.counter = 0
        self.N = self.data.shape[0]
    
    def __iter__(self):
        while self.counter < self.N:
            yield self.data[self.counter:self.counter+self.batch_size]
            self.counter += self.batch_size

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

#%%
def get_dir(file):
    """ Get the folder of specified file """
    return os.path.dirname(os.path.realpath(file))

#%%
def create_dir(direc):
    """ Create a dir if it does not already exists """
    if not os.path.exists(direc):
        os.mkdir(direc)

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
def get_optimizer(name):
    ''' Returns a tensorflow optimizer based on name '''
    optimizer = {'adam':     tf.train.AdamOptimizer,
                 'sgd':      tf.train.GradientDescentOptimizer,
                 'momentum': tf.train.MomentumOptimizer}
    try:
        opt = optimizer[name]
        return opt
    except KeyError:
        raise Exception(name + ' is a invalid option to input optimizer, please'
                        + ' choose between: ' + ', '.join(optimizer.keys()))

#%%
def adjust_learning_rate(alpha, loss_new, loss_old):
    ''' Function for adjusting the learning rate while training '''
    out_alpha = alpha*1.01 if loss_new <= loss_old else alpha*0.5
    return out_alpha

#%%
if __name__ == '__main__':
    print(get_dir(__file__))