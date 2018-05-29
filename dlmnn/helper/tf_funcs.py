#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:40:30 2018

@author: nsde
"""

#%%
import tensorflow as tf

#%%
def tf_makePairwiseFunc(trans_func):
    def func(X1, X2):
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
    return func

#%%
def tf_pairwiseMahalanobisDistance2(X1, X2, L):
    '''
    For a given mahalanobis distance parametrized by L, find the pairwise
    squared distance between all observations in matrix X1 and matrix X2. 
    Input
        X1: N x d matrix, each row being an observation
        X2: M x d matrix, each row being an observation
        L: d x d matrix
    Output
        D: N x M matrix, with squared distances
    '''
    with tf.name_scope('pairwiseMahalanobisDistance2'):
        X1, X2 = tf.cast(X1, tf.float32), tf.cast(X2, tf.float32)  
        X1L = tf.matmul(X1, L)
        X2L = tf.matmul(X2, L)
        term1 = tf.pow(tf.norm(X1L, axis=1),2.0)
        term2 = tf.pow(tf.norm(X2L, axis=1),2.0)
        term3 = 2.0*tf.matmul(X1L, tf.transpose(X2L))
        return tf.transpose(tf.maximum(tf.cast(0.0, dtype=X1L.dtype),
                            term1 + tf.transpose(term2 - term3)))

#%%
def tf_mahalanobisTransformer(X, L):
    ''' Transformer for the mahalanobis distance '''
    with tf.name_scope('mahalanobisTransformer'):
        X, L = tf.cast(X, tf.float32), tf.cast(L, tf.float32)
        return tf.matmul(X, L)

#%%
def tf_pairwiseConvDistance2(X1, X2, W):
    '''
    For a given set of convolutional weights W, calculate the convolution
    with tensor X1 and X2 and then calculates the pairwise squared distance
    (euclidean) between conv features
    Input
        X: N x h x w x c tensor, each slice being an image
        Y: M x h x w x c tensor, each slice being an image
        W: f1 x f2 x c x nf tensor, where f1, f2 are the filter sizes and
           nf is the number of filters
    Output
        D: N x M matrix, with squared conv distances
    '''
    with tf.name_scope('pairwiseConvDistance2'):
        X1, X2 = tf.cast(X1, tf.float32), tf.cast(X2, tf.float32)
        N, M = tf.shape(X1)[0], tf.shape(X2)[0]
        n_filt = tf.shape(W)[3]
        convX1 = tf.nn.conv2d(X1, W, strides=[1,1,1,1], padding='SAME') # N x height x width x n_filt
        convX2 = tf.nn.conv2d(X2, W, strides=[1,1,1,1], padding='SAME') # M x height x width x n_filt
        convX1_perm = tf.transpose(convX1, perm=[3,0,1,2]) # n_filt x N x height x width
        convX2_perm = tf.transpose(convX2, perm=[3,0,1,2]) # n_filt x M x height x width
        convX1_resh = tf.reshape(convX1_perm, (n_filt, N, -1)) # n_filt x N x (height*width)
        convX2_resh = tf.reshape(convX2_perm, (n_filt, M, -1)) # n_filt x M x (height*width)
        term1 = tf.expand_dims(tf.pow(tf.norm(convX1_resh, axis=2), 2.0), 2) # n_filt x N x 1
        term2 = tf.expand_dims(tf.pow(tf.norm(convX2_resh, axis=2), 2.0), 1) # n_filt x 1 x M
        term3 = 2.0*tf.matmul(convX1_resh, tf.transpose(convX2_resh, perm=[0,2,1])) # n_filt x N x M
        summ = term1 + term2 - term3 # n_filt x N x M
        return tf.maximum(tf.cast(0.0, tf.float32), tf.reduce_sum(summ, axis=0)) # N x M

#%%
def tf_convTransformer(X, W):
    ''' Transformer for the conv distance '''
    with tf.name_scope('convTransformer'):
        X, W = tf.cast(X, tf.float32), tf.cast(W, tf.float32)
        return tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')
    
#%%
def tf_nonlin_pairwiseConvDistance2(X1, X2, W):
    '''
    For a given set of convolutional weights W, calculate the convolution
    with tensor X1 and X2 and then calculates the pairwise squared distance
    (euclidean) between conv features
    Input
        X: N x h x w x c tensor, each slice being an image
        Y: M x h x w x c tensor, each slice being an image
        W: f1 x f2 x c x nf tensor, where f1, f2 are the filter sizes and
           nf is the number of filters
    Output
        D: N x M matrix, with squared conv distances
    '''
    with tf.name_scope('pairwiseConvDistance2'):
        X1, X2 = tf.cast(X1, tf.float32), tf.cast(X2, tf.float32)
        N, M = tf.shape(X1)[0], tf.shape(X2)[0]
        n_filt = tf.shape(W)[3]
        convX1 = tf.nn.conv2d(X1, W, strides=[1,1,1,1], padding='SAME') # N x height x width x n_filt
        convX2 = tf.nn.conv2d(X2, W, strides=[1,1,1,1], padding='SAME') # M x height x width x n_filt
        convX1 = tf.nn.relu(convX1)
        convX2 = tf.nn.relu(convX2)
        convX1_perm = tf.transpose(convX1, perm=[3,0,1,2]) # n_filt x N x height x width
        convX2_perm = tf.transpose(convX2, perm=[3,0,1,2]) # n_filt x M x height x width
        convX1_resh = tf.reshape(convX1_perm, (n_filt, N, -1)) # n_filt x N x (height*width)
        convX2_resh = tf.reshape(convX2_perm, (n_filt, M, -1)) # n_filt x M x (height*width)
        term1 = tf.expand_dims(tf.pow(tf.norm(convX1_resh, axis=2), 2.0), 2) # n_filt x N x 1
        term2 = tf.expand_dims(tf.pow(tf.norm(convX2_resh, axis=2), 2.0), 1) # n_filt x 1 x M
        term3 = 2.0*tf.matmul(convX1_resh, tf.transpose(convX2_resh, perm=[0,2,1])) # n_filt x N x M
        summ = term1 + term2 - term3 # n_filt x N x M
        return tf.maximum(tf.cast(0.0, tf.float32), tf.reduce_sum(summ, axis=0)) # N x M

#%%
def tf_nonlin_convTransformer(X, W):
    ''' Transformer for the conv distance '''
    with tf.name_scope('convTransformer'):
        X, W = tf.cast(X, tf.float32), tf.cast(W, tf.float32)
        return tf.nn.relu(tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME'))
    
#%%    
def tf_mode(array):
    ''' Find the mode of the input array. Expects 1D array '''
    with tf.name_scope('mode'):
        unique, _, count = tf.unique_with_counts(array)
        max_idx = tf.argmax(count, axis=0)
        return unique[max_idx]