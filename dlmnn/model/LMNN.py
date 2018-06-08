#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:11:15 2018

@author: nsde
"""

#%%
from dlmnn.helper.neighbor_funcs import findTargetNeighbours, knnClassifier
from dlmnn.helper.tf_funcs import tf_makePairwiseFunc, tf_findImposters
from dlmnn.helper.tf_funcs import tf_LMNN_loss, tf_featureExtractor
from dlmnn.helper.logger import stat_logger
from dlmnn.helper.utility import get_optimizer
from dlmnn.helper.embeddings import embedding_projector

import tensorflow as tf
from tensorflow.python.keras import Sequential
import numpy as np
import datetime, os

#%%
class lmnn(object):
    """   """
    def __init__(self, session=None, dir_loc=None):
        # Initilize session and tensorboard dirs 
        self.session = tf.Session() if session is None else session
        self.dir_loc = './logs' if dir_loc is None else dir_loc
        
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
        """       """
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
        D = self.dist_func(self.Xp, self.Xp)
        tup = tf_findImposters(D, self.yp, self.tNp, margin=margin)
        self._LMNN_loss, D_1, D_2, D_3 = tf_LMNN_loss(D, self.tNp, tup, mu, margin=1)
        
        # Construct training operation
        self.optimizer = get_optimizer(optimizer)(learning_rate=learning_rate)
        self._trainer = self.optimizer.minimize(self._LMNN_loss, 
                                                global_step=self.global_step)
        
        # Summaries
        self._n_tup = tf.shape(tup)[0]
        true_imp = tf.cast(tf.less(D_3, D_2), tf.float32)
        tf.summary.scalar('Loss', self._LMNN_loss) 
        tf.summary.scalar('Num_imp', self._n_tup)
        tf.summary.scalar('Loss_pull', tf.reduce_sum(D_1))
        tf.summary.scalar('Loss_push', tf.reduce_sum(margin + D_2 - D_3))
        tf.summary.histogram('Rel_push_dist', D_3 / (D_2 + margin))
        tf.summary.scalar('True_imp', tf.reduce_sum(true_imp))
        tf.summary.scalar('Frac_true_imp', tf.reduce_mean(true_imp))
        self._summary = tf.summary.merge_all()
               
        # Initilize session
        init = tf.global_variables_initializer()
        self.session.run(init)
        
        # Create callable functions
        self.transformer = self.session.make_callable(
                self.extractor_func(self.Xp), [self.Xp])
        
    #%%
    def fit(self, Xtrain, ytrain, maxEpoch=100, batch_size=50, tN=None, 
            run_id=None, verbose=1, snapshot=10, val_set=None, tN_val=None):
        """
        
        """
        self._assert_if_build()
        # Tensorboard file writers
        run_id = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') if run_id \
                 is None else run_id
        self.current_loc = self.dir_loc + '/' + run_id
        if not os.path.exists(self.dir_loc): os.makedirs(self.dir_loc)
        if verbose == 2: 
            self.train_writer = tf.summary.FileWriter(self.current_loc + '/train')
            self.train_writer.add_graph(self.session.graph)
            if val_set:
                self.val_writer = tf.summary.FileWriter(self.current_loc + '/val')
        
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
        if validation:
            Xval = Xval.astype('float32')
            yval = yval.astype('int32')
            N_val = Xval.shape[0]
            n_batch_val = int(np.ceil(N_val / batch_size))
            print('Number of validation samples:  ', N_val)
        print(70*'-')
        
        # Target neighbours
        if tN is None:
            tN = findTargetNeighbours(Xtrain, ytrain, self.k, name='Training')
        if validation and tN_val is None:
            tN_val = findTargetNeighbours(Xval, yval, self.k, name='Validation')
    
        # Training loop
        stats = stat_logger(maxEpoch, n_batch_train, verbose=verbose)
        stats.on_train_begin() # Start training
        for e in range(maxEpoch):
            stats.on_epoch_begin() # Start epoch
            
            # Permute target neighbours
            tN = np.random.permutation(tN)
            
            # Do backpropagation
            for b in range(n_batch_train):
                stats.on_batch_begin() # Start batch
                
                # Get data
                feed_dict = self._get_feed_dict(self.k*batch_size*b, 
                                                self.k*batch_size*(b+1),
                                                Xtrain, ytrain, tN)
    
                # Evaluate graph
                _, loss_out, ntup_out, summ = self.session.run(
                        [self._trainer, self._LMNN_loss, 
                         self._n_tup, self._summary], 
                         feed_dict=feed_dict)
                
                # Save stats
                stats.add_stat('loss', loss_out)
                stats.add_stat('#imp', ntup_out)
                
                # Save to tensorboard
                if verbose==2: 
                    self.train_writer.add_summary(summ, global_step=b+n_batch_train*e)
                stats.on_batch_end() # End batch
                       
            # Evaluation of validation data
            if validation and ((e+1) % snapshot == 0 or (e+1) == maxEpoch):
                # Evaluate loss and tuples on val data
                tN_val = np.random.permutation(tN_val)
                for b in range(n_batch_val):
                    feed_dict = self._get_feed_dict(self.k*batch_size*b, 
                                                    self.k*batch_size*(b+1),
                                                    Xval, yval, tN_val)
                    loss_out= self.session.run(self._LMNN_loss, feed_dict=feed_dict)
                    stats.add_stat('loss_val', loss_out)
                
                # Compute accuracy
                acc = self.evaluate(Xval, yval, Xtrain, ytrain, batch_size=batch_size)
                stats.add_stat('acc_val', acc)
                
                if verbose==2:
                    # Write stats to summary protocol buffer
                    summ = tf.Summary(value=[
                        tf.Summary.Value(tag='Loss', simple_value=np.mean(stats.get_stat('loss_val'))),
                        tf.Summary.Value(tag='Accuracy', simple_value=np.mean(stats.get_stat('acc_val')))])
             
                    # Save to tensorboard
                    self.val_writer.add_summary(summ, global_step=n_batch_train*e)
            
            stats.on_epoch_end() # End epoch
            
            # Check if we should terminate
            if stats.terminate: break
            
            # Write stats to console (if verbose=True)
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
            X_trans[batch_size*b:batch_size*(b+1)] = self.transformer(X_batch)
        return X_trans
    
    #%%    
    def predict(self, Xtest, Xtrain, ytrain, batch_size=64):
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
        
        Output
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
    def save_embeddings(self, data, direc=None, labels=None):
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
                            imgs=imgs, labels=labels)
    
    #%%
    def get_weights(self):
        """ Returns a list of weights in the current graph """
        self._assert_if_build()
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        return self.session.run(weights)

    #%%
    def summary(self):
        self._assert_if_build()
        print('Model: lmnn')
        print('='*65)
        print('Input shape:               ', self.extractor.input_shape)
        self.extractor.summary()
    
    #%%
    def _get_feed_dict(self, idx_start, idx_end, X, y, tN):
        tN_batch = tN[idx_start:idx_end]
        idx, inv_idx = np.unique(tN_batch, return_inverse=True)
        inv_idx = np.reshape(inv_idx, (-1, 2))
        X_batch = X[idx]
        y_batch = y[idx]
        feed_dict = {self.Xp: X_batch, self.yp: y_batch, self.tNp: inv_idx}
        return feed_dict
   
    #%%
    def _assert_if_build(self):
        assert self.built, '''Model is not build, call lmnn.compile() 
                before this function is called '''
    
#%%
if __name__ == '__main__':
    # Construct model
    model = lmnn()   
    