#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:40:30 2018

@author: nsde
"""

#%%
import tensorflow as tf

#%%
def tf_featureExtractor(keras_sequential, scope='extractor'):
    ''' Feature extractor function build from a keras sequential model '''
    def extractor(X):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, values=[X]):
            return keras_sequential.call(X)
    
    return extractor

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
def tf_findImposters(D, y, tN, margin=1):
    ''' Function for finding imposters in LMNN
        For a set of observations X and that sets target neighbours in tN, 
        find all points that violate the following two equations
                D(i, imposter) <= D(i, target_neighbour) + 1,
                y(imposter) == y(target_neibour)
        for a given distance measure
        
    Arguments:
        D: 
        
        y: N x 1 vector, with class labels
        
        L: d x d matrix, mahalanobis parametrization where M = L^T*L
        
        tN: (N*k) x 2 matrix, where the first column in each row is the
            observation index and the second column is the index of one
            of the k target neighbours
    Output:
        tup: M x 3, where M is the number of triplets that where found to
             fullfill the imposter equation. First column in each row is the 
             observation index, second column is the target neighbour index
             and the third column is the imposter index
    '''
    with tf.name_scope('findImposters'):
        N = tf.shape(D)[0]
        n_tN = tf.shape(tN)[0]
               
        # Create all combination of [points, targetneighbours, imposters]
        possible_imp_array =  tf.expand_dims(tf.reshape(
            tf.ones((n_tN, N), tf.int32)*tf.range(N), (-1, )), 1)
        tN_tiled = tf.reshape(tf.tile(tN, [1, N]), (-1, 2))
        full_idx = tf.concat([tN_tiled, possible_imp_array], axis=1)
        
        # Find distances for all combinations
        tn_index = full_idx[:,:2]
        im_index = full_idx[:,::2]
        D_tn = tf.gather_nd(D, tn_index)
        D_im = tf.gather_nd(D, im_index)
        
        # Find actually imposter by evaluating equation
        y = tf.cast(y, tf.float32) # tf.gather do not support first input.dtype=int32 on GPU
        cond = tf.logical_and(D_im <= margin + D_tn, tf.logical_not(tf.equal(
                              tf.gather(y,tn_index[:,1]),tf.gather(y,im_index[:,1]))))
        full_idx = tf.cast(full_idx, tf.float32) # tf.gather do not support first input.dtype=int32 on GPU
        tup = tf.boolean_mask(full_idx, cond)
        tup = tf.cast(tup, tf.int32) # tf.gather do not support first input.dtype=int32 on GPU
        return tup

#%%    
def tf_LMNN_loss(D, tN, tup, mu, margin=1):
    ''' Calculates the LMNN loss (eq. 13 in paper)
    
    Arguments:
        D: 
    
        tN: (N*k) x 2 matrix, with targetmetric,  neighbour index
        
        tup: ? x 3, where M is the number of triplets that where found to
             fullfill the imposter equation. First column in each row is the 
             observation index, second column is the target neighbour index
             and the third column is the imposter index
             
        mu: scalar, weighting coefficient between the push and pull term
        
        margin: scalar, margin for the algorithm
    
    Output:
        loss: scalar, the LMNN loss
        D_pull: ? x 1 vector, with pull distance terms
        D_tN: ? x 1 vector, with the first push distance terms
        D_im: ? x 1 vector, with the second push distance terms
    '''
    with tf.name_scope('LMNN_loss'):
        # Gather relevant distances
        D_pull = tf.gather_nd(D, tN)
        D_tn = tf.gather_nd(D, tup[:,:2])
        D_im = tf.gather_nd(D, tup[:,::2])
        
        # Calculate pull and push loss
        pull_loss = tf.reduce_sum(D_pull)
        push_loss = tf.reduce_sum(margin + D_tn - D_im)            
        
        # Total loss
        loss = (1-mu) * pull_loss + mu * push_loss
        return loss, D_pull, D_tn, D_im