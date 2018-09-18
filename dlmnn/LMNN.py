#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:11:15 2018

@author: nsde
"""

#%%
from .helper.neighbor_funcs import findTargetNeighbours, findImposterNeighbours, knnClassifier
from .helper.tf_funcs import tf_makePairwiseFunc, tf_findImposters
from .helper.tf_funcs import tf_LMNN_loss, tf_featureExtractor
from .helper.logger import stat_logger
from .helper.utility import get_optimizer, lmnn_batch_builder
from .helper.embeddings import embedding_projector

import tensorflow as tf
from tensorflow.python.keras import Sequential
import numpy as np
import datetime, os

#%%
class lmnn(object):
    """  Large margin nearest neighbour model class  """
    def __init__(self, session=None, dir_loc=None):
        # Initilize session and tensorboard dirs
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config) if session is None else session
        self.dir_loc = './logs' if dir_loc is None else dir_loc
        self._writer = None
        
        # Initialize feature extractor
        self.extractor = Sequential()
        
        # Set idication for when model is build
        self.built = False
    
    #%%        
    def add(self, layer):
        """ Add layer to extractor """
        self.extractor.add(layer)
    
    #%%    
    def compile(self, k=1, optimizer='adam', learning_rate = 1e-4, 
              mu=0.5, margin=1):
        """ Builds the tensorflow graph that is evaluated in the fit method 
        
        Arguments:
            k: integer, number of target neighbours
            optimizer: string, name of optimizer to use. See dlmnn.helper.utility
                for which optimizers that are supported
            learning_rate: scalar, learning rate for optimizer
            mu: scalar, weighting of the pull and push term. Should be between 0
                and 1. High values put weight on the push term and vice verse for
                the pull term
            margin: scalar, size of margin inforcing between similar pairs and
                imposters. Should be higher than 0.
        """
        assert k > 0 and isinstance(k, int), ''' k need to be a positive integer '''   
        assert learning_rate > 0, ''' learning rate needs to be a positive number '''
        assert 0 <= mu and mu <= 1, ''' mu needs to be between 0 and 1 '''
        assert margin > 0, ''' margin needs to be a positive number '''
        assert len(self.extractor.layers)!=0, '''Layers must be added with the 
                lmnn.add() method before this function is called '''
        
        self.built = True
        
        # Set number of neighbours
        self.k = k        

        # Shapes
        self.input_shape = self.extractor.input_shape
        self.output_shape = self.extractor.output_shape
        
        # Placeholders for data
        self.global_step = tf.Variable(0, trainable=False)
        self.Xp = tf.placeholder(tf.float32, shape=self.input_shape, name='In_features')
        self.yp = tf.placeholder(tf.int32, shape=(None,), name='In_targets')
        self.tNp = tf.placeholder(tf.int32, shape=(None, 2), name='In_targetNeighbours')
        
        # Feature extraction function and pairwise distance function
        self.extractor_func = tf_featureExtractor(self.extractor)
        self.dist_func = tf_makePairwiseFunc(self.extractor_func)
        
        # Build graph
        #D = self.dist_func(self.Xp, self.Xp)
        D = self.dist_func(self.Xp)
        tup = tf_findImposters(D, self.yp, self.tNp, margin=margin)
        self._LMNN_loss, D_1, D_2, D_3 = tf_LMNN_loss(D, self.tNp, tup, mu, margin=margin)
        
        # Construct training operation
        self._optimizer = get_optimizer(optimizer)(learning_rate=learning_rate)
        self._trainer = self._optimizer.minimize(self._LMNN_loss, 
                                                 global_step=self.global_step)
        
        # Summaries
        self._n_tup = tf.shape(tup)[0]
        self._true_imp = tf.cast(tf.less(D_3, D_2), tf.float32)
        features = self.extractor_func(self.Xp)
        tf.summary.scalar('Loss', self._LMNN_loss) 
        tf.summary.scalar('Num_imp', self._n_tup)
        tf.summary.scalar('Loss_pull', tf.reduce_sum(D_1))
        tf.summary.scalar('Loss_push', tf.reduce_sum(margin + D_2 - D_3))
        tf.summary.scalar('True_imp', tf.reduce_sum(self._true_imp))
        tf.summary.scalar('Frac_true_imp', tf.reduce_mean(self._true_imp))
        tf.summary.scalar('Sparsity_tanh', tf.reduce_mean(tf.reduce_sum(
                tf.tanh(tf.pow(features, 2.0)), axis=1)))
        tf.summary.scalar('Sparsity_l0', tf.reduce_mean(tf.reduce_sum(
                tf.cast(tf.equal(features, 0), tf.int32), axis=1)))
        self._summary = tf.summary.merge_all()
               
        # Initilize session
        init = tf.global_variables_initializer()
        self.session.run(init)
        
        # Create callable functions
        self._transformer = self.session.make_callable(
                self.extractor_func(self.Xp), [self.Xp])
        self._distances = self.session.make_callable(
                self.dist_func(self.Xp, self.Xp), [self.Xp])
        
    #%%
    def reintialize(self):
        """ Reintialize all weights in the network """
        self._assert_if_build()
        init = tf.global_variables_initializer()
        self.session.run(init)
    
    #%%
    def fit(self, Xtrain, ytrain, maxEpoch=100, batch_size=50, tN=None, imp=None,
            run_id=None, verbose=2, snapshot=10, val_set=None, tN_val=None, imp_val=None):
        """ Fit a deep neural network with lmnn loss
        
        Arguments:
            Xtrain: N x ?, tensor with training data
            ytrain: N x 1, vector with labels (not one-hot encoded)
            maxEpoch: integer, number of epochs to run algoritm
            batch_size: number of samples to feed through network. Not that the
                actual number of samples that are feed in each iteration is
                somewhere between 
                1 + self.k * batch_size < actual < 2 * self.k * batch_size
            val_set: 2-element list, where the first element is a tensor with
                similar shape to Xtrain, and the second element is a vector
                with labels.
            tN: (N*k) x 2, matrix with k target neighbours for each observation
                in the training set. If None, the tNs are computed by using the
                nearest neighbours in a euclidian space
            tN_val: (M*k) x 2, matrix with k target neighbours for each 
                observation in the validation set. If None, the tNs are computed 
                by using the nearest neighbours in a euclidian space
            run_id: str, name of the folder to save results to (will be created). 
                If None, will create a folder with the current time
            verbose: level of output. 0 = no output. 1 = output to consol.
                2 = output to colsol + results saved to tensorboard
            snapshot: integer, how often to evaluate on the validation set.
                Note this is expensive to do, due to accuracy evaluation
            
        Output:
            stats: object containing training stats. See dlmnn.helper.logger
                for more info
        
        """
        self._assert_if_build()
        # Tensorboard file writers
        run_id = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') if run_id \
                 is None else run_id
        self.current_loc = self.dir_loc + '/' + run_id
        if not os.path.exists(self.dir_loc): os.makedirs(self.dir_loc)
        if verbose == 2: 
            self._writer = tf.summary.FileWriter(self.current_loc)
            self._writer.add_graph(self.session.graph)
        
        # Check for validation set
        validation = False
        if val_set:
            validation = True
            Xval, yval = val_set
        
        # Training parameters
        Xtrain = Xtrain.astype('float32')
        ytrain = ytrain.astype('int32')
        N_train = Xtrain.shape[0]
        n_batch_train = int(np.ceil(N_train / batch_size))
        print(70*'-')
        print('Number of training samples:    ', N_train)
        print('Number of training batches:    ', n_batch_train) 
        if validation:
            Xval = Xval.astype('float32')
            yval = yval.astype('int32')
            N_val = Xval.shape[0]
            n_batch_val = int(np.ceil(N_val / batch_size))
            print('Number of validation samples:  ', N_val)
            print('Number of validation batches:  ', n_batch_val)
        print(70*'-' + '\n')
        
        # Target neighbours and imposters
        if tN is None:
            tN = findTargetNeighbours(Xtrain, ytrain, self.k, name='training')
        if imp is None:
            imp = findImposterNeighbours(Xtrain, ytrain, self.k, name='training')
        if validation and tN_val is None:
            tN_val = findTargetNeighbours(Xval, yval, self.k, name='validation')
        if validation and imp_val is None:
            imp_val = findImposterNeighbours(Xval, yval, self.k, name='validation')
            
        # Training loop
        stats = stat_logger(maxEpoch, self.k*n_batch_train, verbose=verbose)
        stats.on_train_begin() # Start training
        for e in range(maxEpoch):
            stats.on_epoch_begin() # Start epoch
            
            # Do backpropagation
            for b, (X_batch, y_batch, tN_batch) in enumerate(
                    lmnn_batch_builder(Xtrain, ytrain, tN, imp, self.k, batch_size)):
                stats.on_batch_begin() # Start batch
                
                # Construct feed dict
                feed_dict = {self.Xp: X_batch, self.yp: y_batch, self.tNp: tN_batch}
                
                # Evaluate graph
                _, loss_out, ntup_out, ntup_true_out, summ = self.session.run(
                        [self._trainer, self._LMNN_loss, self._n_tup,
                         self._true_imp, self._summary], 
                         feed_dict=feed_dict)
                
                # Save stats
                stats.add_stat('loss', loss_out)
                stats.add_stat('#imp', ntup_out)
                if ntup_true_out.size: # find frac of true imposters
                    stats.add_stat('Frac_true_imp', np.mean(ntup_true_out))
                
                # Save to tensorboard
                if verbose==2: 
                    self._writer.add_summary(summ, global_step=b+n_batch_train*e)
                stats.on_batch_end() # End batch
                       
            # If we are at an snapshot epoch and are doing validation
            if validation and ((e+1) % snapshot == 0 or (e+1) == maxEpoch or e==0):
                # Evaluate loss and tuples on val data
                for X_batch, y_batch, tN_batch in lmnn_batch_builder(
                        Xval, yval, tN_val, imp_val, self.k, batch_size):
                    # Construct feed dict
                    feed_dict = {self.Xp: X_batch, self.yp: y_batch, self.tNp: tN_batch}
                
                    # Compute loss
                    loss_out= self.session.run(self._LMNN_loss, feed_dict=feed_dict)
                    stats.add_stat('loss_val', loss_out)
                
                # Compute accuracy
                acc = self.evaluate(Xval, yval, Xtrain, ytrain, batch_size=batch_size)
                stats.add_stat('acc_val', acc)
                
                if verbose==2:
                    # Write stats to summary protocol buffer
                    summ = tf.Summary(value=[
                        tf.Summary.Value(tag='Loss_val', simple_value=
                                         np.mean(stats.get_stat('loss_val'))),
                        tf.Summary.Value(tag='Accuracy_val', simple_value=
                                         np.mean(stats.get_stat('acc_val')))])
             
                    # Save to tensorboard
                    self._writer.add_summary(summ, global_step=n_batch_train*e)
            
            # If we are at an snapshot epoch, then recompute the imposters
            if (e+1) % snapshot == 0:
                Xtrain_trans = self.transform(Xtrain)
                Xval_trans = self.transform(Xval)
                imp = findImposterNeighbours(Xtrain_trans, ytrain, self.k, name='training')
                imp_val = findImposterNeighbours(Xval_trans, yval, self.k, name='validation')
                
            stats.on_epoch_end() # End epoch
            
            # Check if we should terminate
            if stats.terminate: break
            
            # Write stats to console (if verbose>0)
            stats.write_stats()
            
        stats.on_train_end() # End training
        
        # Save variables and training stats
        self.save_weights(run_id + '/trained_metric')
        stats.save(self.current_loc + '/training_stats')
        return stats
                
    #%%
    def transform(self, X, batch_size=64):
        ''' Transform the data in X
        Arguments:
            X: N x ?, matrix or tensor of data
            batch_size: scalar, number of samples to transform in parallel
        Output:
            X_trans: N x ?, matrix or tensor with the transformed data
        '''
        self._assert_if_build()
        # Parameters for transformer
        N = X.shape[0]
        n_batch = int(np.ceil(N / batch_size))
        X_trans = np.zeros((N, *self.output_shape[1:]))
        
        # Transform data in batches
        for b in range(n_batch):
            X_batch = X[batch_size*b:batch_size*(b+1)]
            X_trans[batch_size*b:batch_size*(b+1)] = self._transformer(X_batch)
        return X_trans
    
    #%%    
    def predict(self, Xtest, Xtrain, ytrain, batch_size=64):
        ''' Predicts the labels of Xtest using KNN
        Arguments: 
            Xtest: N x ?, tensor with data to predict labels for
            Xtrain: M x ?, tensor with training data
            ytrain: M x 1, vector with training labels
            batch_size: scalar, number of samples to transform in parallel
        Output:
            pred: N x 1, vector with predicted labels
        '''
        self._assert_if_build()
        Xtest = self.transform(Xtest, batch_size=batch_size)
        Xtrain = self.transform(Xtrain, batch_size=batch_size)
        pred = knnClassifier(Xtest, Xtrain, ytrain, self.k)
        return pred
    
    #%%
    def evaluate(self, Xtest, ytest, Xtrain, ytrain, batch_size=64):
        ''' Evaluates the current metric
        
        Arguments:
            Xtest: M x ? metrix or tensor with test data for which we want to
                   predict its classes for
            Xtrain: N x ? matrix or tensor with training data
            ytrain: N x 1 vector with class labels for the training data
            k: scalar, number of neighbours to look at
            batch_size: integer, number of samples to transform in parallel
        
        Output:
            accuracy: scalar, accuracy of the prediction for the current metric
        '''
        self._assert_if_build()
        pred = self.predict(Xtest, Xtrain, ytrain, batch_size=batch_size)
        accuracy = np.mean(pred == ytest)
        return accuracy
    
    #%%
    def save_weights(self, filename, step=None):
        ''' Save all weights/variables in the current session to a file 
        
        Arguments:
            filename: str, name of the file to write to
            step: integer, appended to the filename to distingues different saved
                files from each other
        '''
        self._assert_if_build()
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        saver.save(self.session, self.dir_loc+'/'+filename, global_step = step)
    
    #%%
    def save_embeddings(self, data, labels=None, direc=None):
        """ Embed some data with the current network, and save these to
            tensorboard for vizualization
            
        Arguments:
            data: data to embed, shape must be equal to model.input_shape
            direc: directory to save data to
            labels: if data has labels, supply these for vizualization
        
        """
        self._assert_if_build()
        direc = self.current_loc if direc is None else '.'
        embeddings = self.transform(data)
        imgs = data if (data.ndim==4 and (data.shape[-1]==1 or data.shape[-1]==3)) else None
        embedding_projector(embeddings, direc, name='embedding',
                            imgs=imgs, labels=labels, writer=self._writer)
    
    #%%
    def get_weights(self):
        """ Returns a list of weights in the current graph """
        self._assert_if_build()
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        return self.session.run(weights)

    #%%
    def summary(self):
        """ Gets a summary of the feature extraction model """
        print('Model: lmnn')
        print('='*65)
        print('Input shape:               ', self.extractor.input_shape)
        self.extractor.summary()
    
    #%%
    def _get_feed_dict(self, idx_start, idx_end, X, y, tN):
        """ Utility function for getting a batch of data """
        tN_batch = tN[idx_start:idx_end]
        idx, inv_idx = np.unique(tN_batch, return_inverse=True)
        inv_idx = np.reshape(inv_idx, (-1, 2))
        X_batch = X[idx]
        y_batch = y[idx]
        feed_dict = {self.Xp: X_batch, self.yp: y_batch, self.tNp: inv_idx}
        return feed_dict
   
    #%%
    def _assert_if_build(self):
        """ Utility function for checking if model is compiled """
        assert self.built, '''Model is not build, call model.compile() 
                before this method is called. '''
    
#%%
if __name__ == '__main__':
    model = lmnn()
        
            
            