#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 09:58:39 2018

@author: nsde
"""
#%%
#%%
def tf_mahalanobisTransformer(X, scope='mahalanobis_transformer'):
    """ Creates a transformer function that for an given input matrix X, 
        calculates the linear transformation L*X_i """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, values=[X]):
        X = tf.cast(X, tf.float32)
        L = tf.get_variable("L", initializer=np.eye(50, 50, dtype=np.float32))
        return tf.matmul(X, L)

#%%
def tf_convTransformer(X, scope='conv_transformer'):
    """ Creates a transformer function that for an given input tensor X,
        computes the convolution of that tensor with some weights W. """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, values=[X]):
        X = tf.cast(X, tf.float32)
        W = tf.get_variable("W", initializer=np.random.normal(size=(3,3,1,10)).astype('float32'))
        return tf.nn.conv2d(X, W, strides=[1,1,1,1], padding="VALID")

#%%
def keras_mahalanobisTransformer(X, scope='mahalanobis_transformer'):
    X = tf.cast(X, tf.float32)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, values=[X]):
        S = Sequential()
        S.add(InputLayer(input_shape=(50,)))
        S.add(Dense(50, use_bias=False, kernel_initializer='identity'))
        return S.call(X)
        
#%%
if __name__ == '__main__':
    # Make sure that variables are shared
    X=tf.cast(np.random.normal(size=(100,50)), tf.float32)
    
    tr = KerasTransformer(input_shape=(50,))
    tr.add(Dense(50, use_bias=False, kernel_initializer='identity'))
    tr.add(Dense(20, use_bias=False, kernel_initializer='identity'))
    tr.add(Dense(10, use_bias=False, kernel_initializer='identity'))
    trans_func1 = tr.get_function()
    
    res1=trans_func1(X)
    res2=trans_func1(X)
    
    # Check that we only have created three variables
    for w in tf.trainable_variables():
        print(w)
    
    
    


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