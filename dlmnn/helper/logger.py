#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:24:01 2018

@author: nsde
"""
#%%
import numpy as np
import time, sys, pickle

#%%
class stat_logger(object):
    '''
    This class can be used during an training process to keep track of different
    training parameters, training time ect.
    Input:
        n_epoch: scalar, number of epochs
        n_batch: scalar, number of batches
        n_move: scalar, number of batches to calculate moving average over
        verbose: int, determines if training stats are printed
        terminate_tol: scalar, if difference in moving average is smaller than
                       this, then the process is terminated
    '''
    def __init__(self, n_epoch, n_batch, verbose=1, 
                 n_move = 20, terminate_tol = 1e-8):
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.verbose = verbose
        
        # Stat variables
        self.data = [ ]
        self.data_show = { }
    
        # Timing variables
        self.start_train, self.end_train, self.total_train_time = 0,0,0
        self.start_epoch, self.end_epoch, self.epoch_train_time = 0,0,0
        self.start_batch, self.end_batch, self.batch_train_time = 0,0,0
        self.eta = [ ]
        
        # Counters
        self.epoch, self.batch = 0, 0
        
        # Early stopping variables
        self.n_move = n_move
        self.loss_means, self.old_loss, self.new_loss = [], 0, 0
        self.tolerance, self.terminate = terminate_tol, False
        
        # Printing stuff
        self.output_e = self.output_b = None
        self.output_0 = self.output_1 = None
        
    def on_train_begin(self):
        ''' Call on beginning of training '''
        self.start_train = time.time()
        self.epoch = 0
    
    def on_train_end(self):
        ''' Call on the end of training '''
        self.end_train = time.time()
        self.total_train_time = self.end_train - self.start_train
        self._writer(70*'=' + '\n')
        self._writer('Total train time: {0:0.1f}s \n'.format(self.total_train_time))
    
    def on_epoch_begin(self):
        ''' Call on the beginning of an epoch '''
        self.start_epoch = time.time()
        self.data.append({})
        self.batch = 0
        
        l = len(str(self.n_epoch))
        self._writer(70*'-' + '\n')
        self.output_e = ('Epoch: {0:'+str(l)+'}/{1}, ').format(self.epoch+1, self.n_epoch)
        self._writer(self.output_e)
                          
    def on_epoch_end(self):
        ''' Call on the end of an epoch '''
        self.end_epoch = time.time()
        self.epoch_train_time = self.end_epoch - self.start_epoch
        self.epoch += 1
        
        # Compute moving average over the last n loss means
        self.loss_means.append(np.mean(self.data[self.epoch-1]['loss']))
        self.new_loss = np.mean(self.loss_means[-self.n_move:])
        if np.abs(self.new_loss - self.old_loss) < self.tolerance:
            print('No significant change in loss, terminating!')
            self.terminate = True
        self.old_loss = self.new_loss
        self._writer(('\r' + self.output_e + self.output_b + 'Time: {0:3.2f}s' +
                     '                   ').format(self.epoch_train_time))
    
    def on_batch_begin(self):
        ''' Call in the beginning of a batch '''
        self.start_batch = time.time()
        l = len(str(self.n_batch))
        self.output_b = ('Batch: {0:'+str(l)+'}/{1}, ').format(self.batch+1, self.n_batch)
        self._writer('\r' + self.output_e + self.output_b)
        
    def on_batch_end(self):
        ''' Call in the end of a batch '''
        self.end_batch = time.time()
        self.batch_train_time = self.end_batch - self.start_batch
        self.batch += 1
        
        # Check for nan values in loss
        if np.isnan(self.data[self.epoch]['loss'][-1]):
            print('Encountered nan in loss function, terminating!')
            self.terminate = True
            sys.exit()
            
        self.eta.append(self.batch_train_time) 
        self.output_0 = 'ETA: ' + str(np.round(np.mean(self.eta)*
                    (self.n_batch - self.batch + 1), 2)) + 's, '
        self.output_1 = 'loss: '+str(np.round(self.data[self.epoch]['loss'][-1],3))
        self._writer('\r' + self.output_e + self.output_b 
                         + self.output_0 + self.output_1)
    
    def add_stat(self, name, value, epoch=None, verbose=True):
        ''' 
        Store some training value, for printing or later inspection 
        Input:
            name: str, name of value to store
            value: scalar-vector-matrix ect, with the values to store
            epoch: scalar, what epoch to store the values under. If none it
                   will store it under the current epoch
            verbose: bool, determines if the variable should be printed by the
                     write_stats() function
        '''
        epoch = self.epoch if epoch is None else epoch
        if name in self.data[epoch]:
            self.data[epoch][name].append(value)
        else:
            self.data[epoch][name] = [value]
        self.data_show[name] = verbose
   
    
    def get_stat(self, name, epoch=None):
        ''' Return the save values of a given stat '''
        epoch = self.epoch if epoch is None else epoch
        return self.data[epoch][name]
  
    def save(self, filename):
        file = open(filename + '.pkl', 'wb')
        pickle.dump(self.__dict__, file)
        file.close()
        
    def load(self, filename):
        self.__dict__ = pickle.load(open(filename + '.pkl', 'rb'))
    
    def write_stats(self, epoch=None):
        ''' Write saved stats to the console '''
        epoch = self.epoch-1 if epoch is None else epoch        
        # Format
        out = '   '
        for name, val in self.data[epoch].items():
            if self.data_show[name]:
                if name == 'loss' or name == 'loss_val': # known format
                    out += (name + ': {0:8.3f}, ').format(np.mean(val))
                elif name == 'norm': # known format
                    out += 'norm: {0:.3f}, '.format(np.mean(val)) 
                elif name == '#imp' or name == '#imp_val': # known format
                    out += (name + ': {0:5d}, ').format(np.sum(val))
                elif name == 'acc' or name == 'acc_val': # known format
                    out += (name + ': {0:.3f}, ').format(np.mean(val))
                else: # default output mean of value
                    out += (name + ': {0:.3f}, ').format(np.mean(val)) 

        # Print
        self._writer('\n' + out + '\n')
    
    def _writer(self, string):
        if self.verbose:
            sys.stdout.write(string)
            sys.stdout.flush()
        

    
#%%
if __name__ == '__main__':
    n_epoch = 10
    n_batch = 5
    
    stats = stat_logger(n_epoch, n_batch)
    
    stats.on_train_begin() # start training
    for e in range(n_epoch):
        stats.on_epoch_begin() # start epoch
        for b in range(n_batch):
            stats.on_batch_begin() # start batch
            
            # Do computations
            time.sleep(np.random.uniform(0.5, 1))
            
            # Add stats
            stats.add_stat('loss', np.random.normal(10)/(1+e*b)) # common to use                          
            stats.add_stat('acc', np.random.uniform(0,1)) # common to use
            stats.add_stat('hest', np.random.normal()) # not common to use 
            
            stats.on_batch_end() # end batch
            
        # Do validation
        if e % 3 == 0:
            time.sleep(np.random.uniform(1, 2))
            stats.add_stat('loss_val', np.random.normal(10)/(1+e*b))
            stats.add_stat('acc_val', np.random.uniform(0,1))

        stats.on_epoch_end() # end epoch
        stats.write_stats()
    
    stats.on_train_end() # end training

    

    
    
    
        