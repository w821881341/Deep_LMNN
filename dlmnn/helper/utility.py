# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:23:37 2017

@author: nsde
"""

import tensorflow as tf
import os, sys

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