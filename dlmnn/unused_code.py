#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:15:02 2017

@author: nsde
"""
#%%
import tensorflow as tf
import numpy as np

 #%%
    def plot_metric(self, X, y, metric):
        fig = plt.gcf()
        fig.clear()
        X_trans = self.transform(X, metric)
        plt.subplot(1,3,1)
        plt.scatter(X[:,0], X[:,1], c=y)
        plt.axis('equal')
        plt.subplot(1,3,2)
        plt.scatter(X_trans[:,0], X_trans[:,1], c=y)
        plt.axis('equal')
        plt.subplot(1,3,3)
        Linv = np.linalg.inv(metric)
        plot_normal2D(np.mean(X_trans,axis=0), Linv.dot(Linv.T))
        plt.axis('equal')
        plt.show()
        plt.pause(0.1)
        

    #%%
    def predict(self, Xtest, Xtrain, ytrain, k, metric=None):
        '''
        Predict the class labels of the test set using KNN classifier
        '''
        metric = self.metric if metric is None else metric
        y = self.tf_KNN_classifier(Xtest, Xtrain, ytrain, k, metric)
        ypred = self.session.run(y)
        return ypred
        
    #%%
    def evaluate(self, Xtest, ytest, Xtrain, ytrain, k, metric=None):
        '''
        Evaluate the performance of metric L, by calculating the classification
        performance on the test set based on the training set using a KNN classifier
        Input
            Xtest: M x d matrix, with test observations
            ytest: M x 1 vector, with test class labels
            Xtrain: N x d matrix, with training observations
            ytrian: N x 1 vector, with training class labels
            L: d x d matrix, mahalanobis parametrization where M = L^T*L
            k: scalar, number of neigbours for the KNN classifierdataset = "training"
        Output
            acc: scalar, prediction accuracy on the test set
        '''
        metric = self.metric if metric is None else metric
        
        # Predict class labels on test set
        ypred = self.predict(Xtest, Xtrain, ytrain, k, metric=metric)
        
        # Calcualte accuracy
        acc = 1/ytest.size * np.sum(ytest == ypred)
        return acc



    #%%
    def tf_LMNN_loss(self, X, y, metric, tN, tup, mu, margin=None):
        '''
        Calculates the LMNN loss (eq. 13 in paper)
        Input:
            X: N x d matrix, each row being a_batchn observation
            y: N x 1 vector, with class labels
            L: d x d matrix, mahalanobis parametrization where M = L^T*L
            tN: (N*k) x 2 matrix, where the first column in each row is the
                observation index and the second column is the index of one
                of the k target neighbours
            tup: M x 3, where M is the number of triplets that where found to
                 fullfill the imposter equation. First column in each row is the 
                 observation index, second column is the target neighbour index
                 and the third column is the imposter index
            mu: scalar, weighting coefficient between the push and pull term
        Output
            loss: scalar, the LMNN loss
        '''
        with tf.name_scope('LMNN_loss'):
            margin = self.margin if margin is None else margin
            # Calculate distance
            D = self.metric_func(X, X, metric) 
            
            # Gather relevant distances
            D_pull = tf.gather_nd(D, tN)
            D_push_1 = tf.gather_nd(D, tup[:,:2]) # first and second column
            D_push_2 = tf.gather_nd(D, tup[:,::2]) # first and third column
            
            # Calculate pull and push loss
            pull_loss = tf.reduce_sum(D_pull)
            push_loss = tf.reduce_sum(margin + D_push_1 - D_push_2)
            
            # Total loss
            loss = (1-mu) * pull_loss + mu * push_loss
            return loss

    #%%
    def findTargetNeighbours(self, X, y, k, bs=50):
        '''
        Similar to the function above, but with the difference that:
            * This function returns a numpy array instead of a tensorflow tensor
            * The function computes the distances in batches, meaning that it
              is a lot slower to compute however, it should fit into memory
              as long the batch size is kept resonble.
            * This function expects that we can find atleast k target neighbours
              for each observation in X
        '''
        assert np.min(np.bincount(y_train))-1 >= k, 'parameter k is too high'
        N = X.shape[0] # Number of observations
        tN = np.empty((N*k,2))
        for i in range(N):
            progressBar(i+1, N, name='Finding target neighbours')
            c = y[i] # Class of observation i
            D = [ ]
            n_batch = int(np.ceil(N/bs))
            for j in range(n_batch):
                # Calculate distance between i and a batch of points
                x = np.concatenate((X[i][np.newaxis], X[bs*j:bs*(j+1)]), axis=0)
                dist = self.metric_func(x, x, self.identity_metric)
                D.append(self.session.run(dist)[0,1:])
            # Find the k closest points to observation i
            D = np.hstack(tuple(D))
            idx = np.argsort(D)
            idx_sort = idx[y[idx]==c][1:k+1]
            row1, row2 = np.meshgrid(i, idx_sort)
            row_i = np.concatenate((row1, row2), axis=1)
            # Save result
            tN[k*i:k*(i+1),:]=row_i
        return tN


    #%%
    def train_new(self, Xtrain, ytrain, k, mu=0.5, maxEpoch=100, metricInit=None,
                  tN_init = None, learning_rate=1e-4, batch_size=50, sample_size=100,
                  val_set=None, run_id = None):
        '''
        Function for training the LMNN algorithm
        TODO: More description
        '''
        # Tensorboard file writers
        run_id = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') if run_id \
                 is None else run_id
        loc = self.dir_loc + '/' + run_id
        self.train_writer = tf.summary.FileWriter(loc + '/train')
        if val_set:
            validation = True
            self.val_writer = tf.summary.FileWriter(loc + '/val')
            Xval, yval = val_set
        else:
            validation = False
            
        # Training parameters
        N_train = Xtrain.shape[0]
        assert N_train > 2*k*batch_size + sample_size, '''Batch size or sample size
                are too high, please reduce one of them'''
        
        # Metric to train
        metricInit = self.identity_metric if metricInit is None else metricInit
        metricInit = metricInit + np.random.normal(scale=0.1, size=metricInit.shape) # add some noise
        metric = tf.Variable(metricInit, dtype=tf.float32, name='Metric')
        
        # Pre-calculate targetneighbours
        if not tN_init:
            if N_train < 5000: # use fast version
                tN = self.session.run(self.tf_findTargetNeighbours(Xtrain, ytrain, k))
            else: # use slow but memory efficient version
                tN = self.findTargetNeighbours(Xtrain, ytrain, k, bs=batch_size)
        else:
            assert tN.shape[1] == 2, 'Target neighbours not formatted correctly'
        N_tN = tN.shape[0]
        tN_perm = np.random.permutation(tN)
        n_batch_tN = int(np.ceil(int(N_tN/k)/batch_size))
        
        # Placeholders for data to feed
        Xp = tf.placeholder(tf.float32, shape=(None, *Xtrain.shape[1:]), name='In_features')
        yp = tf.placeholder(tf.int32, shape=(None,), name='In_targets')
        tNp = tf.placeholder(tf.int32, shape=(None,2), name='In_targetNeighbours')
        global_step = tf.Variable(0, trainable=False)
        
        # Imposter tuples
        tup = self.tf_findImposters(Xp, yp, metric, tNp)
        
        # LMNN loss
        LMNN_loss = self.tf_LMNN_loss_adjusted(Xp, yp, metric, tup, mu)
        
        # Optimizer
        optimizer = self.optimizer(learning_rate = learning_rate)
        trainer = optimizer.minimize(LMNN_loss, global_step=global_step)
        
        # Summaries
        metricNorm = tf.norm(metric)
        summ_norm = tf.summary.scalar('Metric_Norm', metricNorm)
        n_tup = tf.shape(tup)[0]
        summ_imp = tf.summary.scalar('NumberOfImposters', n_tup)
        summ_loss = tf.summary.scalar('Loss', LMNN_loss)
        merged = tf.summary.merge([summ_norm, summ_loss, summ_imp])
        
        # Initilize
        init = tf.global_variables_initializer()
        self.session.run(init)
        self.train_writer.add_graph(self.session.graph)
        
        # Training
        stats = stat_logger(maxEpoch, n_batch_tN)
        stats.on_train_begin()
        for e in range(maxEpoch):
            stats.on_epoch_begin()
            for b in range(n_batch_tN):
                stats.on_batch_begin()
                # Sampling scheme
                tN_idx = tN_perm[batch_size*k*b:batch_size*k*(b+1)] # Get random set of target neighbours
                idx, rev = np.unique(tN_idx, return_inverse=True) # Find uniq ones
                tN_batch = np.reshape(rev, (-1, 2)) # Map target neighbour index to batch format
                # Draw possible random imposters
                batch_idx = np.concatenate((idx, random_not_in_sampler(sample_size, N_train, idx))) 
                
                # Get data
                X_batch = Xtrain[batch_idx]
                y_batch = ytrain[batch_idx]
                feed_data = {Xp: X_batch, yp: y_batch, tNp: tN_batch}
                
                # Evaluate graph
                _, val1, val2, val3, summ = self.session.run(
                    [trainer, LMNN_loss, metricNorm, n_tup, merged], 
                    feed_dict = feed_data)
                
                # Save stats
                stats.add_stat(e, 'loss', val1)
                stats.add_stat(e, 'norm', val2)
                stats.add_stat(e, '#imp', val3)
                
                # Save to tensorboard
                self.train_writer.add_summary(summ, global_step=b+n_batch_tN*e)
                stats.on_batch_end()
                
            stats.on_epoch_end()
            
            # Output stats 
            stats.write_stats(e)
            
        stats.on_train_end()




    #%%
    def tf_getNeighbours(self, X, k, metric=None):
        ''' For a matrix X, get the index of the k nearest neighbours '''
        with tf.name_scope('getNeighbours'):
            metric = self.metric if metric is None else metric
            
            # Calculate distance
            D = self.metric_func(X, X, metric)
            
            # Find k nearest neigbours
            _, idx = tf.nn.top_k(tf.reduce_max(D) - D, k=k+1)    
            return idx[:,1:]

#%%
            # Do some plotting if asked for (only every 10 iteration)
            if plot and e % 10 == 0 and self.metric_parameters == 2:
                fig.clear()
                X_trans = self.transform(Xtrain, metricOut)
                plt.subplot(1,3,1)
                plt.scatter(X_trans[:,0], X_trans[:,1], c=ytrain)
                plt.axis('equal')
                plt.subplot(1,3,2)
                Linv = np.linalg.inv(metricOut)
                plot_normal2D(np.mean(X_trans,axis=0), Linv.dot(Linv.T))
                plt.axis('equal')
                plt.subplot(1,3,3)
                plt.semilogy(loss_e, 'b-')
                plt.pause(0.05)
                plt.show()
            if plot and e % 10 == 0 and self.metric_type == 'conv':
                fig.clear()
                Xtrans = self.transform(X_batch, metric=metricOut)
                plt.subplot(1,3,1)
                plt.imshow(metricOut[:,:,0,0], cmap = plt.get_cmap('gray'))
                plt.axis('off')
                plt.subplot(1,3,2)
                plt.imshow(Xtrans[0,:,:,0], cmap=plt.get_cmap('gray'))
                plt.axis('off')
                plt.subplot(1,3,3)
                plt.semilogy(loss_e, 'b-')
                plt.show()


#%%
def pairwiseMahalanobisDistance_M(X1, X2, M):
    '''
    For a given mahalanobis distance parametrized by M, find the pairwise
    distance between all observations in matrix X and matrix Y. 
    Input
        X: N x d matrix, each row being an observation
        Y: M x d matrix, each row being an observation
        M: d x d matrix
    Output
        D: N x M matrix
    '''
    term1 = np.diagonal(np.dot(X1,np.dot(M,X1.T))) # N x 1
    term2 = np.diagonal(np.dot(X2,np.dot(M,X2.T))) # M x 1
    term3 = 2*X1.dot(M.dot(X2.T))
    return (term1 + (term2 - term3).T).T # N x M 

#%%    
def tf_pairwiseMahalanobisDistance_M(X1, X2, M):
    '''
    For a given mahalanobis distance parametrized by M, find the pairwise
    distance between all observations in matrix X and matrix Y. 
    Input
        X: N x d matrix, each row being an observation
        Y: M x d matrix, each row being an observation
        M: d x d matrix
    Output
        D: N x M matrix
    '''
    term1 = tf.diag_part(tf.matmul(X1,tf.matmul(M,tf.transpose(X1)))) # N x 1
    term2 = tf.diag_part(tf.matmul(X2,tf.matmul(M,tf.transpose(X2)))) # M x 1
    term3 = 2*tf.matmul(X1,tf.matmul(M,tf.transpose(X2)))
    return tf.sqrt(tf.transpose(term1 + tf.transpose(term2 - term3))) # N x M 

#%%    
def _LMMN_loss_grad(op, grad):
    '''
    Calculates the gradient of the LMNN loss (the function above)
    Input
        op: tensorflow operator class
        grad: incoming gradient to be backpropegated
    Output
        gradient: a list, where each entry is the gradient w.r.t. the input
                  of the LMNN loss function
    '''
    # Grap input
    X, _, _, tN, tup, mu = op.inputs
    
    # Gather relevant x's 
    x_pull_i = tf.expand_dims(tf.gather(X, tN[:,0]), 2)
    x_pull_j = tf.expand_dims(tf.gather(X, tN[:,1]), 1)
    x_push_i = tf.expand_dims(tf.gather(X, tup[:,0]), 2)
    x_push_j = tf.expand_dims(tf.gather(X, tup[:,1]), 1)
    x_push_l = tf.expand_dims(tf.gather(X, tup[:,2]), 1)
    
    # Calculate outer products
    Cij_pull = tf.matmul(x_pull_i, x_pull_j)
    Cij_push = tf.matmul(x_push_i, x_push_j)
    Cil_push = tf.matmul(x_push_i, x_push_l)
    
    # Calculate pull and push grad
    pull_grad = tf.reduce_sum(Cij_pull, axis=0)
    push_grad = tf.reduce_sum(Cij_push - Cil_push, axis=0)
    
    # Total grad
    gradient = (1-mu) * pull_grad + mu * push_grad
    return [None, None, gradient*grad, None, None, None]

#%%
@function.Defun( tf.float32, tf.int32, tf.float32, tf.int32, tf.int32, tf.float32, 
                 func_name = 'tf_LMNN_loss', python_grad_func = _LMMN_loss_grad)     
def tf_LMNN_loss(X, y, L, tN, tup, mu):
    ''' Simple wrapper function to combine the LMNN loss with its gradient function'''
    return _LMNN_loss(X, y, L, tN, tup, mu)

#%%
def targetNeighbours_sparse_to_dense(self, tN_sparse, N, k):
    '''
    Converts a target neighbour matrix from the sparse format ((N*k) x 2)to 
    the dense format (N x k)
    '''
    tN_dense = np.zeros((N, k), dtype = np.int32)
    count = 0
    for i in range(N*k):
        tN_dense[tN_sparse[i,0],count] = tN_sparse[i,1]
        count += 1
        if count == k:
            count = 0
    return tN_dense

#%%
def targetNeighbours_dense_to_sparse(self, tN_dense, N, k):
    '''
    Converts a target neighbour matrix from the dense format (N x k) to 
    the sparse format ((N*k) x 2)
    '''
    tN_sparse = np.zeros((N*k, 2), dtype = np.int32)
    count = 0
    for i in range(N):
        for j in range(k):
            tN_sparse[count] = [i, tN_dense[i,j]]
            count += 1
    return tN_sparse

#%%
def pairwiseMahalanobisDistance2(self, X1, X2, L):
    '''
    For a given mahalanobis distance parametrized by L, find the pairwise
    squared distance between all observations in matrix X1 and matrix X2. 
    Input
        X1: N x d matrix, each row being an observation
        X2: M x d matrix, each row being an observation
        L: d x d matrix
    Output
        D2: N x M matrix, with squared distances
    '''
    X1L = X1.dot(L)
    X2L = X2.dot(L)
    term1 = np.linalg.norm(X1L, axis=1)**2
    term2 = np.linalg.norm(X2L, axis=1)**2
    term3 = 2*(X1L.dot(X2L.T))
    return np.maximum(0,term1 + (term2 - term3).T).T

#%%     
def findTargetNeighbours(self, X, y, k):
    '''
    For a set of observations in X, for each observation find the k closest 
    points by euclidian distance that have the same label as the observation.
    These are called the target neighbours.
    Input
        X: N x d matrix, each row being an observation
        y: N x 1 vector, with class labels
        k: parameter, number of target neighbours to find
    Output
        tN: (N*k) x 2 matrix, where the first column in each row is the
            observation index and the second column is the index of one
            of the k target neighbours
    '''
    N, d = X.shape
    
    # Secure that their is k neighbours in each class
    counts = np.bincount(y)
    assert np.min(counts[counts.nonzero()[0]]) >= k+1, '''Number of target 
        neigbours are too high, could not find enough neighbours in all classes'''
    
    # Calculate distance
    D = self.pairwiseMahalanobisDistance2(X, X, np.eye(d))
    idx = np.argsort(D, axis=1)
    neighbourIndex = y[idx]
    
    # Iterate over all observations
    tN = np.zeros((N*k, 2), np.int32)
    for i in range(N):
        count = 0
        for j in range(1, N):

            if neighbourIndex[i,0] == neighbourIndex[i,j] and count < k:
                tN[k*i + count] = [i, idx[i,j]]
                count += 1
                
    return tN