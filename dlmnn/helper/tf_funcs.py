#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:40:30 2018

@author: nsde
"""

#%%
import tensorflow as tf
import numpy as np

#%%
def tf_makePairwiseFunc(trans_func):
    """ Creates a function that calculates pairwise distance between features 
        that are determined by the input trans_func 
    
    Arguments:
        trans_func: callable function that takes a single input and returns
            features extract from that output.
    
    Output:
        func: a callable function that for two inputs X1 and X2, returns a 
            pairwise distance matrix between each set of observations
    """
    
    def dist_func(X1, X2):
        with tf.name_scope('pairwise_' + trans_func.__name__):
            X1, X2 = tf.cast(X1, tf.float32), tf.cast(X2, tf.float32)
            N, M = tf.shape(X1)[0], tf.shape(X2)[0]
            X1_trans = trans_func(X1) # N x ???
            X2_trans = trans_func(X2) # M x ???
            X1_resh = tf.reshape(X1_trans, (N, -1)) # N x ?
            X2_resh = tf.reshape(X2_trans, (M, -1)) # M x ?
            term1 = tf.expand_dims(tf.pow(tf.norm(X1_resh, axis=1), 2.0), 1) # N x 1
            term2 = tf.expand_dims(tf.pow(tf.norm(X2_resh, axis=1), 2.0), 0) # 1 x M
            term3 = 2.0*tf.matmul(X1_resh, tf.transpose(X2_resh)) # N x M
            summ = term1 + term2 - term3 # N x M
            return tf.maximum(tf.cast(0.0, tf.float32), summ) # N x M
    return dist_func

#%%
def tf_mahalanobisTransformer(X, scope='mahalanobis_transformer'):
    """ Creates a transformer function that for an given input matrix X, 
        calculates the linear transformation L*X_i """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, values=[X]):
        X = tf.cast(X, tf.float32)
        L = tf.get_variable("L", initializer=np.eye(50, 50, dtype=np.float32))
        return tf.matmul(X, L)

#%%
def keras_mahalanobisTransformer(X, scope='mahalanobis_transformer'):
    from tensorflow.python.keras import Sequential
    from tensorflow.python.keras.layers import InputLayer, Dense
    X = tf.cast(X, tf.float32)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, values=[X]):
        S = Sequential()
        S.add(InputLayer(input_shape=(50,)))
        S.add(Dense(50, use_bias=False, kernel_initializer='identity'))
        return S.call(X)

#%%
def tf_convTransformer(X, scope='conv_transformer'):
    """ Creates a transformer function that for an given input tensor X,
        computes the convolution of that tensor with some weights W. """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, values=[X]):
        X = tf.cast(X, tf.float32)
        W = tf.get_variable("W", initializer=np.random.normal(size=(3,3,1,10)).astype('float32'))
        return tf.nn.conv2d(X, W, strides=[1,1,1,1], padding="VALID")

#%%
if __name__ == '__main__':
    # Make sure that variables are shared
    X=tf.cast(np.random.normal(size=(100,50)), tf.float32)
    keras_res1 = keras_mahalanobisTransformer(X)
    keras_res2 = keras_mahalanobisTransformer(X)
    if len(tf.trainable_variables())==1: print('succes!')
    