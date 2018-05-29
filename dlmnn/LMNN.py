# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:06:31 2017

@author: nsde
"""
#%%
from __future__ import print_function
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from tf_funcs import get_transformer_struct
from helper import get_optimizer, lmnn_argparser, progressBar
from logger import stat_logger

plt.ion()

#%%
class LMNN(object):
    def __init__(self,  metric_struct,
                        margin = 1, 
                        session=None, 
                        dir_loc=None, 
                        optimizer='adam',
                        verbose = 1):
        """ LMNN class used for training the LMNN algorithm
        Parameters
        -----------
        metric_struc: ???
        margin: scalar, optional (default = 1)
            margin threshold for the algorithm. Determines the scaling of the
            feature space
        session: tf.Session, optional (default = None)
            tensorflow session which the computations are runned within. If None
            then a new is opened
        dir_loc: str, optional (default = None)
            directory to store tensorboard files for plotting. If None, a folder
            will be created named lmnn_'metric_type'
        optimizer: str, optional (default = 'adam')
            choose which optimizer to use. Choose between 'sgd', 'adam' or 'momentum'
        verbose: scalar, optional (default = 1)
            choose the level of output. If verbose=0, nothing will be printed. 
            If verbose=1, training progress will be printed to the screen. If
            verbose=2, also do some plotting
        """
        # Initilize session and tensorboard dirs 
        self.session = tf.Session() if session is None else session
        self.dir_loc = (dir_loc + '/lmnn_' + str(metric_struct['name'])) if dir_loc \
                        is not None else ('lmnn_' + str(metric_struct['name']))
        self.train_writer = None
        self.val_writer = None
        self.optimizer = get_optimizer(optimizer)
        self.verbose = verbose
        
        # Set margin for algorithm (determine scaling in feature space)
        self.margin = margin
        self.dim = 1 # dimensionality of data, set by the train func
        
        # Extract relevant function from the metric struct
        self.metric_type = metric_struct['name']
        self.transformer = metric_struct['transformer']
        self.metric_func = metric_struct['metric_func']
        self.metric = metric_struct['initial_metric']
        self.metric_parameters = metric_struct['params']

    #%%
    def tf_findTargetNeighbours(self, X, y, k):
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
        with tf.name_scope('findTargetNeighbours'):
            # Get shapes
            N = tf.shape(X)[0]
            
            # Calculate distance
            D = self.metric_func(X, X, self.metric)
            
            # Find closest points
            _, idx = tf.nn.top_k(tf.reduce_max(D)-D, k=N)
            y_idx = tf.gather(y, idx)
            
            # Find points with same class label
            cond = tf.cast(tf.equal(tf.expand_dims(y_idx[:,0], 1), y_idx[:,1:]), tf.int32)
            val_same, idx_same = tf.nn.top_k(cond, k=k)
            
            # Get the index of those points
            init_idx = tf.reshape(tf.transpose(tf.ones((k, N), dtype=tf.int32)*tf.range(N)), (-1,1))
            map_idx = tf.concat([init_idx, tf.reshape(idx_same, (-1,1))], axis=1)
            final_idx = tf.expand_dims(tf.gather_nd(idx[:,1:], map_idx), axis=1)
            final_idx = tf.concat([init_idx, final_idx], axis=1)
            
            # Filter out some of the observations if k is too high for some of the
            # classes
            final_idx = tf.boolean_mask(final_idx, tf.equal(
                    tf.gather(y, final_idx[:,0]), tf.gather(y, final_idx[:,1])))
            
            return final_idx
        
    #%%  
    def tf_findImposters(self, X, y, metric, tN, margin = None):
        '''
        For a set of observations X and that sets target neighbours in tN, find
        all points that violate the following two equations
            D_L(i, imposter) <= D_L(i, target_neighbour) + 1,
            y(imposter) == y(target_neibour)
        for a given mahalanobis distance M = L^T*L
        Input
            X: N x d matrix, each row being an observation
            y: N x 1 vector, with class labels
            L: d x d matrix, mahalanobis parametrization where M = L^T*L
            tN: (N*k) x 2 matrix, where the first column in each row is the
                observation index and the second column is the index of one
                of the k target neighbours
        Output
            tup: M x 3, where M is the number of triplets that where found to
                 fullfill the imposter equation. First column in each row is the 
                 observation index, second column is the target neighbour index
                 and the third column is the imposter index
        '''
        with tf.name_scope('findImposters'):
            margin = self.margin if margin is None else margin
            N = tf.shape(X)[0]
            n_tN = tf.shape(tN)[0]
            
            # Calculate distance
            D = self.metric_func(X, X, metric) # d x d
            
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
    def tf_LMNN_loss(self, X, y, metric, tN, tup, mu, margin=None):
        '''
        Calculated an adjusted version of the LMNN loss (eq. 13 in paper). In
        this version we sum over all triplets in both the pull and push term,
        such that no imposters implies that we do not pull or push:
            sum_i(sum_j(pull(j->i)) + sum_j(sum_k(push(k != j->i))))
        Input
            X: N x d matrix, each row being a_batchn observation
            y: N x 1 vector, with class labels
            metric: d x d matrix, mahalanobis parametrization where M = L^T*L
            tN: 
            tup: M x 3, where M is the number of triplets that where found to
                 fullfill the imposter equation. First column in each row is the 
                 observation index, second column is the target neighbour index
                 and the third column is the imposter index
            mu: scalar, weighting coefficient between the push and pull term
            margin: scalar, margin for the algorithm
        Output
            loss: scalar, the LMNN loss
        '''
        with tf.name_scope('LMNN_loss'):
            margin = self.margin if margin is None else margin
            
            # Calculate distance
            D = self.metric_func(X, X, metric) # N x N
            
            # Gather relevant distances
            D_pull = tf.gather_nd(D, tN)
            D_tn = tf.gather_nd(D, tup[:,:2])
            D_im = tf.gather_nd(D, tup[:,::2])
            
            # Calculate pull and push loss
            pull_loss = tf.reduce_sum(1.0/self.dim * D_pull)
            push_loss = tf.reduce_sum(margin + 1.0/self.dim*D_tn - 1.0/self.dim*D_im)            
            
            # Total loss
            loss = (1-mu) * pull_loss + mu * push_loss
            return loss, D_pull, D_tn, D_im
    
    #%%
    def tf_KNN_classifier(self, Xtest, Xtrain, ytrain, k, metric=None):
        '''
        Standard KNN classifier based on the distance measure parametrized by 
        some metric        
        Input
            Xtest: M x d matrix, with test observations            
            Xtrain: N x d matrix, with training observations            
            ytrian: N x 1 vector, with training class labels            
            L: d x d matrix, mahalanobis parametrization where M = L^T*L
            k: scalar, number of neigbours to consider
        Output
            ytest: M x 1 vector, with predicted class labels for the test set
        '''
        with tf.name_scope('KNN_classifier'):
            metric = self.metric if metric is None else metric
            # Calculate distance
            D = self.metric_func(Xtest, Xtrain, metric)
        
            # Find k nearest neigbours
            _, idx = tf.nn.top_k(tf.reduce_max(D) - D, k)
            
            # Get their labels
            yidx = tf.gather(ytrain, idx)
            
            # Find the most common label
            ytest = tf.map_fn(tf_mode, yidx)
            return ytest

    #%%
    def train(self, Xtrain, ytrain, k, mu=0.5, maxEpoch=100, metricInit = None, 
                 learning_rate=1e-4, batch_size=50, val_set=None, run_id = None,
                 tN=None, snapshot=10):
        '''
        Function for training the LMNN algorithm
        Input
            Xtrain: N x ?, Input training data, can be a 4D tensor of images
                    or a matrix with data in standard format
            ytrain: N x 1 vector, with training labels
            k: scalar, number of target neighbours
            mu: scalar, weighting coefficient between the push and pull term
            maxEpoch: scalar, maximum number of epochs to run the algorithm
            metricInit: tensor or matrix, initial metric
            learning_rate: scalar, initial learning rate for the optimizer
            batch_size: integer, number of samples to process in each iteration
            val_set: 2-element list, a list [Xval, yval] where Xval should have
                     same format as Xtrain and yval should have equal number of
                     labels as samples in Xval
            run_id: string, custom name to use for the tensorboard loggin
            tN: matrix, target neighbours for the learning algorithm. If none,
                these will be calculated by the program
            snapshot: integer, determines how often to calculate accuracy on
                      train set and evaluate on validation set, since this is
                      very costly to do
        Output
            stats: a instance of the stat_logger class which contains all
                   information gather during training
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
        n_batch_train = int(np.ceil(N_train / batch_size))
        print(50*'-')
        print('Number of training samples:    ', N_train)
        if validation:
            N_val = Xval.shape[0]
            n_batch_val = int(np.ceil(N_val / batch_size))
            print('Number of validation samples:  ', N_val)
        print(50*'-')
        
        # Target neighbours
        if tN is None:
            tN = self.findTargetNeighbours(Xtrain, ytrain, k, name='Training')
        if validation:
            tN_val = self.findTargetNeighbours(Xval, yval, k, name='Validation')
        
        # Metric to train
        self.dim = np.product(Xtrain.shape[1:])
        metricInit = self.metric if metricInit is None else metricInit
        metricInit = 1 + metricInit + np.random.normal(scale=0.1, size=metricInit.shape) # add some noise
        metric = tf.Variable(metricInit, dtype=tf.float32, name='Metric')
        
        # Placeholders for data
        global_step = tf.Variable(0, trainable=False)
        Xp = tf.placeholder(tf.float32, shape=(None, *Xtrain.shape[1:]), name='In_features')
        yp = tf.placeholder(tf.int32, shape=(None,), name='In_targets')
        tNp = tf.placeholder(tf.int32, shape=(None, 2), name='In_targetNeighbours')
        
        # Imposters
        tup = self.tf_findImposters(Xp, yp, metric, tNp)
        
        # Loss func
        LMNN_loss, D_1, D_2, D_3 = self.tf_LMNN_loss(Xp, yp, metric, tNp, tup, mu)
        
        # Optimizer
        optimizer = self.optimizer(learning_rate = learning_rate)
        trainer = optimizer.minimize(LMNN_loss, global_step=global_step)
        
        # Summaries
        metricNorm = tf.norm(metric)
        n_tup = tf.shape(tup)[0]
        true_imp = tf.cast(tf.less(D_3, D_2), tf.float32)
        tf.summary.scalar('Loss', LMNN_loss) 
        tf.summary.scalar('Metric_Norm', metricNorm)
        tf.summary.histogram('Metric', metric)
        tf.summary.scalar('Num_imp', n_tup)
        tf.summary.scalar('Loss_pull', tf.reduce_sum(D_1))
        tf.summary.scalar('Loss_push', tf.reduce_sum(self.margin + D_2 - D_3))
        tf.summary.histogram('Rel_push_dist', D_3 / (D_2 + self.margin))
        tf.summary.scalar('True_imp', tf.reduce_sum(true_imp))
        tf.summary.scalar('Frac_true_imp', tf.reduce_mean(true_imp))
        merged = tf.summary.merge_all()
        
        # Initilize
        init = tf.global_variables_initializer()
        self.session.run(init)
        self.train_writer.add_graph(self.session.graph)
        
        # Training
        stats = stat_logger(maxEpoch, n_batch_train)
        stats.on_train_begin() # Start training
        for e in range(maxEpoch):
            stats.on_epoch_begin() # Start epoch
            
            tN = np.random.permutation(tN)
            # Do backpropagation
            for b in range(n_batch_train):
                stats.on_batch_begin() # Start batch

                tN_batch = tN[k*batch_size*b:k*batch_size*(b+1)]
                idx, inv_idx = np.unique(tN_batch, return_inverse=True)
                inv_idx = np.reshape(inv_idx, (-1, 2))
                X_batch = Xtrain[idx]
                y_batch = ytrain[idx]
                feed_data = {Xp: X_batch, yp: y_batch, tNp: inv_idx}
            
                # Evaluate graph
                _, loss_out, norm_out, ntup_out, summ = self.session.run(
                    [trainer, LMNN_loss, metricNorm, n_tup, merged], 
                    feed_dict=feed_data)
                
                # Save stats
                stats.add_stat('loss', loss_out)
                stats.add_stat('norm', norm_out)
                stats.add_stat('#imp', ntup_out)                   
                
                # Save to tensorboard
                self.train_writer.add_summary(summ, global_step=b+n_batch_train*e)
                stats.on_batch_end() # End batch
            
            # Evaluate accuracy every 'snapshot' epoch (expensive to do) and
            # on the last epoch
            if e % snapshot == 0 or e == maxEpoch-1:
                y_pred = self.KNN_classifier(Xtrain, Xtrain, ytrain, k=k, 
                                             metric=metric, batch_size=batch_size)
                acc = np.mean(y_pred == y_train)
                stats.add_stat('acc', acc)
                summ = tf.Summary(value=[tf.Summary.Value(tag='Accuracy', simple_value=acc)])
                self.train_writer.add_summary(summ, global_step=b+n_batch_train*e)
                                
            # Do validation if val_set is given and we are in the snapshot epoch
            # or at the last epoch
            if validation and (e % snapshot == 0 or e == maxEpoch-1):
                # Evaluate loss and tuples on val data
                tN_val = np.random.permutation(tN_val)
                for b in range(n_batch_val):
                    tN_batch = tN_val[k*batch_size*b:k*batch_size*(b+1)]
                    idx, inv_idx = np.unique(tN_batch, return_inverse=True)
                    inv_idx = np.reshape(inv_idx, (-1, 2))
                    X_batch = Xval[idx]
                    y_batch = yval[idx]
                    feed_data = {Xp: X_batch, yp: y_batch, tNp: inv_idx}
                    loss_out, ntup_out = self.session.run([LMNN_loss, n_tup], feed_dict=feed_data)
                    stats.add_stat('loss_val', loss_out)
                    stats.add_stat('#imp_val', ntup_out)
                
                # Compute accuracy
                y_pred = self.KNN_classifier(Xval, Xtrain, ytrain, k=k, 
                                             metric=metric, batch_size=batch_size)
                acc = np.mean(y_pred == yval)
                stats.add_stat('acc_val', acc)
                
                # Write stats to summary protocol buffer
                summ = tf.Summary(value=[
                    tf.Summary.Value(tag='Loss', simple_value=np.mean(stats.get_stat('loss_val'))),
                    tf.Summary.Value(tag='NumberOfImposters', simple_value=np.mean(stats.get_stat('#imp_val'))),
                    tf.Summary.Value(tag='Accuracy', simple_value=np.mean(stats.get_stat('acc_val')))])
             
                # Save to tensorboard
                self.val_writer.add_summary(summ, global_step=n_batch_train*e)
    
            stats.on_epoch_end() # End epoch
            
            # Check if we should terminate
            if stats.terminate: break
            
            # Write stats to console
            if self.verbose: stats.write_stats()
        
        stats.on_train_end() # End training
        
        # Save the metric and output the training stats
        trained_metric = self.session.run(metric)
        self.metric = trained_metric
        self.save_metric(run_id+'/trained_metric.npy', metric=trained_metric)
        stats.save(loc + '/training_stats')
        return stats
    
    #%%
    def transform(self, X, metric=None, batch_size = 100):
        ''' Transform the data in X according to a given metric
        Input:
            X: N x ?, matrix or tensor of data
            metric: matrix or tensor with the metric to use
            batch_size: scalar, number of samples to transform in parallel
        Output:
            X_trans: N x ?, matrix or tensor with the transformed data
        '''
        # Check metric and metric parameters
        metric = self.metric if metric is None else metric
        metric_parameters = [self.metric_parameters] if type(self.metric_parameters) \
                            is not list else self.metric_parameters
        # Create structure to hold new observations
        N = X.shape[0]
        n_batch = int(np.ceil(N / batch_size))
        X_trans = np.zeros((*X.shape[:-1], metric_parameters[-1]), X.dtype)
        
        # Transform data in batches
        for b in range(n_batch):
            X_batch = X[batch_size*b:batch_size*(b+1)]
            X_batch_trans = self.transformer(X_batch, metric)
            X_trans[batch_size*b:batch_size*(b+1)] = self.session.run(X_batch_trans)
        return X_trans
   
    #%%
    def findTargetNeighbours(self, X, y, k, do_pca=True, permute=False, name=''):
        ''' Numpy/sklearn implementation to find target neighbours for large datasets.
        This cannot use the GPU, but instead uses an advance ball-tree method.
        Input:
            X: N x ?, metrix or tensor with data
            y: N x 1, vector with labels
            k: scalar, number of target neighbours to find
            reshape: bool, if the input should be reshaped into a (N, -1) matrix
            do_pca: bool, if true then the data will first be projected onto
                    a pca-space which captures 95% of the variation in data
            permute: bool, if true then the target neighbour matrix is permuted
                     before it is returned
            name: str, name of the dataset
        '''
        print(50*'-')
        N = X.shape[0]
        X = np.reshape(X, (N, -1))
        if do_pca:
            print('Doing PCA')
            pca= PCA(n_components = 0.95)
            X = pca.fit_transform(X)
        val = np.unique(y) 
        counter = 1
        tN_count = 0
        tN = np.zeros((N*k, 2), np.int32)
        for c in val:
            progressBar(counter, len(val), name='Finding target neighbours for ' + name)
            idx = np.where(y==c)[0]
            n_c = len(idx)
            x = X[idx]
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
            nbrs.fit(x)
            _, indices = nbrs.kneighbors(x)
            for kk in range(1,k+1):
                tN[tN_count:tN_count+n_c,0] = idx[indices[:,0]]
                tN[tN_count:tN_count+n_c,1] = idx[indices[:,kk]]
                tN_count += n_c
            counter += 1
        print(' ')
        if permute:
            print('Permuting target neighbours')
            tN = np.random.permutation(tN)
        print(50*'-')
        return tN
    
    #%%
    def KNN_classifier(self, Xtest, Xtrain, ytrain, k, metric, batch_size=50):
        '''
        KNN classifier using sklearns library. This scales well for alot of data
        Input:
            Xtest: M x ? metrix or tensor with test data for which we want to
                   predict its classes for
            Xtrain: N x ? matrix or tensor with training data
            ytrain: N x 1 vector with class labels for the training data
            k: scalar, number of neighbours to look at
            metric: determines the space we measure distance in
            batch_size: integer, number of samples to transform in parallel
        Output:
            pred: M x 1 vector with predicted class labels for the test set
        '''
        Ntest = Xtest.shape[0]
        Ntrain = Xtrain.shape[0]
        Xtest_t = self.transform(Xtest, metric, batch_size=batch_size)
        Xtrain_t = self.transform(Xtrain, metric, batch_size=batch_size)
        Xtest_t = np.reshape(Xtest_t, (Ntest, -1))
        Xtrain_t = np.reshape(Xtrain_t, (Ntrain, -1))
    
        classifier = KNeighborsClassifier(n_neighbors = k, n_jobs=-1)
        classifier.fit(Xtrain_t, ytrain)
        pred = classifier.predict(Xtest_t)
        return pred
    
    #%%
    def Energy_classifier(self, Xtest, Xtrain, ytrain, k, metric, batch_size=50):
        ''' Not completed''' 
        Ntest = Xtest.shape[0]
        Ntrain = Xtrain.shape[0]
        
        # Transform according to metric
        Xtest_t = self.transform(Xtest, metric, batch_size=batch_size)
        Xtrain_t = self.transform(Xtrain, metric, batch_size=batch_size)
        Xtest_t = np.reshape(Xtest_t, (Ntest, -1))
        Xtrain_t = np.reshape(Xtrain_t, (Ntrain, -1))
        
        # Observations in euclidian space
        Xtest_euclid = np.reshape(Xtest, (Ntest, -1))
        Xtrain_euclid = np.reshape(Xtrain, (Ntrain, -1))
        
        # Loop over all possible classes
        val = np.unique(ytrain) 
        Dist = np.zeros((Ntest, len(val)))
        count = 0
        for c in val:
            idx = np.where(c==ytrain)[0]
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
            nbrs.fit(Xtrain_euclid[idx])
            _, indices = nbrs.kneighbors(Xtest_euclid)
            
            # Each test points target neighbour
            idx_x = idx[indices] # Ntest x k
            
            # Get all distance between test and train
            D = cdist(Xtest_t, Xtrain_t) # Ntest x Ntrain
            
            # Use linear index to get pull distances
            idx_lin = idx_x.flatten() + [i*Ntrain for i in range(Ntest) for _ in range(k)]
            D_tn = np.reshape(D.flatten()[idx_lin], idx_x.shape)
            Dpull = np.sum(D_tn, axis=1)
            
            # Iterate over all 
            Dpush = np.zeros((Ntest,))
            for kk in range(k):
                # Find all imposters
                cond = np.logical_and((D.T<=1+D_tn[:,kk]).T, ytrain != c)
                
                # Sum all imposters
                Dpush+=np.sum(np.where(cond, D, np.zeros_like(D)), axis=1)
            mu = 0.5
            Dist[count, :] = (1-mu)*Dpull + mu*Dpush
        
        min_c = np.argmin(Dist, axis=1)
        return val[min_c]

    #%%
    def save_metric(self, filename, metric=None):
        ''' Save the current metric to a file '''
        metric = self.metric if metric is None else metric
        np.save(self.dir_loc + '/' + filename, metric)
    
    #%%
    def load_metric(self, filename):
        ''' Load a metric from a file, and set It as the current metric '''
        metric = np.load(filename)
        self.metric = metric

#%%
if __name__ == '__main__':
    from helper import read_olivetti, read_mnist, test_set1
    tf.InteractiveSession()
    
    # Get arguments
    args = lmnn_argparser()
    
    # Prepare data
    if args['ds'] == 'testset':
        X_train, y_train, X_test, y_test = test_set1(size=50)
        dim = 2
    elif args['ds'] == 'olivetti':
        X_train, y_train, X_test, y_test = read_olivetti(permute=True)
        X_train = np.reshape(X_train, (280, -1))
        X_test = np.reshape(X_test, (120, -1))
        dim = 105
        pca = PCA(n_components=dim)
        X_train = pca.fit_transform(X_train)    
        X_test = pca.transform(X_test)
    elif args['ds'] == 'mnist2000':
        X_train, y_train = read_mnist(dataset='training')
        X_test, y_test = read_mnist(dataset='testing')
        n1, n2 = 2000, 200
        X_train = np.reshape(X_train[:n1], (n1, -1))
        X_test = np.reshape(X_test[:n2], (n2, -1))
        y_train = y_train[:n1]
        y_test = y_train[:n2]
        dim = 105
        pca = PCA(n_components=dim)
        X_train = pca.fit_transform(X_train)    
        X_test = pca.transform(X_test)
    elif args['ds'] == 'mnist5000':
        X_train, y_train = read_mnist(dataset='training')
        X_test, y_test = read_mnist(dataset='testing')   
        n1, n2 = 5000, 500
        X_train = np.reshape(X_train[:n1], (n1, -1))
        X_test = np.reshape(X_test[:n2], (n2, -1))
        y_train = y_train[:n1]
        y_test = y_train[:n2]
        dim = 105
        pca = PCA(n_components=dim)
        X_train = pca.fit_transform(X_train)    
        X_test = pca.transform(X_test)
    elif args['ds'] == 'mnist':
        X_train, y_train = read_mnist(dataset='training')
        X_test, y_test = read_mnist(dataset='testing')   
        X_train = np.reshape(X_train, (60000, -1))
        X_test = np.reshape(X_test, (10000, -1))
        dim = 105
        pca = PCA(n_components=dim)
        X_train = pca.fit_transform(X_train)    
        X_test = pca.transform(X_test)
    elif args['ds'] == 'olivetti_conv':
        X_train, y_train, X_test, y_test = read_olivetti(permute=False)
        X_train = np.reshape(X_train, (-1, 64, 64, 1))
        X_test = np.reshape(X_test, (-1, 64, 64, 1))
    elif args['ds'] == 'mnist_conv':
        X_train, y_train = read_mnist(dataset='training')
        X_test, y_test = read_mnist(dataset='testing')
        X_train = X_train[:,:,:,np.newaxis]
        X_test = X_test[:,:,:,np.newaxis]
    
    ms = get_transformer_struct(transformer=args['mt'], params=args['p'])
    
    lmnn = LMNN(metric_struct=ms, 
                margin=args['m'], 
                optimizer=args['o'],
                verbose=args['v'])

    train_stats = lmnn.train(X_train, y_train, k=args['k'], mu=args['mu'], 
                             maxEpoch=args['ne'], batch_size=args['bs'], 
                             learning_rate=args['lr'], val_set=[X_test, y_test], 
                             snapshot=args['ss'])
    
