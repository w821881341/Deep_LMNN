#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:11:15 2018

@author: nsde
"""

#%%
from dlmnn.helper.neighbor_funcs import findTargetNeighbours
from dlmnn.helper.tf_funcs import tf_makePairwiseFunc, tf_findImposters, \
                                    tf_LMNN_loss, tf_featureExtractor
from dlmnn.helper.logger import stat_logger
from dlmnn.helper.utility import get_optimizer

import tensorflow as tf
from tensorflow.python.keras import Sequential
import numpy as np
import datetime, os

class lmnn(object):
    #%%
    def __init__(self, session=None, dir_loc=None):
        # Initilize session and tensorboard dirs 
        self.session = tf.Session() if session is None else session
        self.dir_loc = './logs' if dir_loc is None else dir_loc
        self.train_writer = None
        self.val_writer = None
        
        # Initialize feature extractor
        self.extractor = Sequential()
    
    #%%        
    def add(self, layer):
        """ Add layer to extractor """
        self.extractor.add(layer)
    
    #%%    
    def build(self, optimizer='adam', learning_rate = 1e-4, mu=0.5, margin=1):
        """       """
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
        
        # Initilize
        init = tf.global_variables_initializer()
        self.session.run(init)
        
    #%%
    def fit(self, Xtrain, ytrain, k=3, maxEpoch=100, 
            batch_size=50, val_set=None, run_id=None):
        """
        
        """
        # Tensorboard file writers
        run_id = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') if run_id \
                 is None else run_id
        loc = self.dir_loc + '/' + run_id
        if not os.path.exists(self.dir_loc): os.makedirs(self.dir_loc)
        if self.verbose == 2: 
            self.train_writer = tf.summary.FileWriter(loc + '/train')
            self.train_writer.add_graph(self.session.graph)
            if val_set:
                self.val_writer = tf.summary.FileWriter(loc + '/val')
        
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
        print(50*'-')
        print('Number of training samples:    ', N_train)
        if validation:
            Xval = Xval.astype('float32')
            yval = yval.astype('int32')
            N_val = Xval.shape[0]
            n_batch_val = int(np.ceil(N_val / batch_size))
            print('Number of validation samples:  ', N_val)
        print(50*'-')
        
        # Target neighbours
        tN = findTargetNeighbours(Xtrain, ytrain, k, name='Training')
        if validation:
            tN_val = findTargetNeighbours(Xval, yval, k, name='Validation')
    
        # Training loop
        stats = stat_logger(maxEpoch, n_batch_train, verbose=self.verbose)
        stats.on_train_begin() # Start training
        for e in range(maxEpoch):
            stats.on_epoch_begin() # Start epoch
            
            # Permute target neighbours
            tN = np.random.permutation(tN)
            
            # Do backpropagation
            for b in range(n_batch_train):
                stats.on_batch_begin() # Start batch
                
                # Get data
                feed_dict = self._get_feed_dict(k*batch_size*b, k*batch_size*(b+1),
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
                if self.verbose==2: 
                    self.train_writer.add_summary(summ, global_step=b+n_batch_train*e)
                stats.on_batch_end() # End batch
            
            stats.on_epoch_end() # End epoch
            
            # Check if we should terminate
            if stats.terminate: break
            
            # Write stats to console (if verbose=True)
            stats.write_stats()
            
        stats.on_train_end() # End training
        
        # Save variables and training stats
        self.save_weights(run_id + '/trained_metric')
        stats.save(loc + '/training_stats')
        return stats
                
                
    #%%
    def transform(self, ):
        pass
    
    #%%    
    def predict(self, ):
        pass
    
    #%%
    def evaluate(self, ):
        pass
    
    #%%
    def save_weights(self, filename, step=None):
        ''' Save all weights/variables in the current session to a file 
        Arguments:
            filename: str, name of the file to write to
            step: integer, appended to the filename to distingues different saved
                files from each other
        '''
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        saver.save(self.session, self.dir_loc+'/'+filename, global_step = step)
    
    #%%
    def get_weights(self):
        """ Returns a list of weights in the current graph """
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        return self.session.run(weights)
    
    #%%
    def _get_feed_dict(self, idx_start, idx_end, tN, X, y):
        tN_batch = tN[idx_start:idx_end]
        idx, inv_idx = np.unique(tN_batch, return_inverse=True)
        inv_idx = np.reshape(inv_idx, (-1, 2))
        X_batch = X[idx]
        y_batch = y[idx]
        feed_dict = {self.Xp: X_batch, self.yp: y_batch, self.tNp: inv_idx}
        return feed_dict
    
#%%
if __name__ == '__main__':
    from tensorflow.python.keras.layers import Dense, InputLayer
    # Construct model
    model = lmnn()
    
    # Add feature extraction layers
    model.add(InputLayer(input_shape=(50,)))
    model.add(Dense(20, use_bias=False, kernel_initializer='identity'))
    
    # Build graph
    model.build(optimizer='adam')