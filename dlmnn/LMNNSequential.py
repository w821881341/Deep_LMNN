#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:15:34 2018

@author: nsde
"""
#%%
from .LMNN import lmnn
from .helper.layers import layerlist, InputLayer, MaxPool2D, Flatten, LeakyReLU, \
                            Dense, Conv2D
from .helper.neighbor_funcs import findTargetNeighbours
from tensorflow.python.keras import Sequential
import tensorflow as tf

#%%
class sequental_lmnn(lmnn):
    ''' Sequential large margin nearest neighbour model class. This is a
        subclass of the lmnn class, the implements some additional methods 
        for easier with a sequence of lmnn models'''
    def __init__(self, session=None, dir_loc=None):
        super().__init__(session, dir_loc)
        self._layerlist = layerlist()
        
        # These list may need to be expanded
        self._parametric_layers = (Dense, Conv2D)
        self._nonparametric_layers = (InputLayer, MaxPool2D, Flatten, LeakyReLU)
        
    def add(self, layer, **arguments):
        ''' Add a layer and the arguments for the layer to layer list '''
        self._layerlist.add(layer, **arguments)
    
    def determine_models(self):
        ''' Function for determing the set of models that we should train. The
            paradigm is the following:
                * First layer is always added
                * The three last layers are always added
                    - Flatten() layer
                    - Dense() layer
                    - Activation() layer
                * The intermedian layers are added in such a way that model[i+1]
                  contains one more parametric layer than model[i]
        Output:
            models: list, each element is a list that contain the index of the
                layers in that particular model. Note len(models[i]) < len(models[i+1])            
        '''
        # Number of layers
        n_layers = self._layerlist.n_layers
        assert n_layers > 0, 'Add layers to the model before trying to fit'        
        
        # Initial structures
        models = [ ]
        para_layers_found, models_found, i = 0, 0, 0
        
        # Loop until we reach the final 3 layers
        while i < n_layers-4:
            # Try to find a model
            current_list = [ ]
            for i in range(n_layers-3):
                if self._layerlist.layers[i] in self._parametric_layers:
                    # Make sure that next time we encounter a parametric layer
                    # we break the loop
                    if para_layers_found <= models_found:
                        current_list.append(i)
                        para_layers_found += 1
                    else:
                        models_found += 1
                        break
                elif self._layerlist.layers[i] in self._nonparametric_layers:
                    # We can add as many non parametric layers as we want
                    current_list.append(i)
                else:
                    raise Exception(str(self._layerlist.layers[i]) + '''not found
                        found in the list of allowed layers. Please add the layer
                        to model._parametric_layers or model._nonparametric_layers
                        dependent on if the layers has trainable parameters''')

            # Always add the last three layers (Flatten(), Dense(), Activation())
            current_list.append(n_layers-3)
            current_list.append(n_layers-2)
            current_list.append(n_layers-1)
            
            # Save and reset
            para_layers_found = 0
            models.append(current_list)
            
        return models
        
    def fit_sequential(self, Xtrain, ytrain, epochs_pr_model=50, batch_size=50, 
                       run_id=None, verbose=2, snapshot=10, val_set=None, k=1, 
                       optimizer='adam', learning_rate = 1e-4, mu=0.5, margin=1):
        ''' Main method of this class. Fits a sequence of lmnn models'''
        # Check for validation set
        validation = False
        if val_set:
            validation = True
            Xval, yval = val_set
        
        # Find valid models
        model_list = self.determine_models()
        n_models = len(model_list)
        
        # Find initial tNs
        tN = [findTargetNeighbours(Xtrain, ytrain, k)]
        if validation:
            tN_val = [findTargetNeighbours(Xval, yval, k)]
        
        # Loop through models
        all_stats = [ ]
        for i in range(n_models):
            # Construct the extractor
            self.extractor = Sequential()
            for l in model_list[i]: self.extractor(self._layerlist.get_layer[l])
            
            # Compile model
            self.compile(k=k, optimizer=optimizer, learning_rate=learning_rate,
                         mu=mu, margin=margin)
            
            # fit
            stats = self.fit(Xtrain, ytrain,
                             val_set=val_set,
                             maxEpoch=epochs_pr_model, 
                             batch_size=batch_size, 
                             tN=tN[-1], tN_val=tN_val[-1], 
                             run_id=run_id, 
                             verbose=verbose, 
                             snapshot=snapshot)
            all_stats.append(stats)
            
            # Transform data and compute target neighbours
            Xtrain_trans = self.transform(Xtrain)
            tN.append(findTargetNeighbours(Xtrain_trans, ytrain, k))
            if validation:
                Xval_trans = self.transform(Xval)
                tN_val.append(findTargetNeighbours(Xval_trans, yval, k))
            
            # reset graph
            tf.reset_default_graph()
        
        return all_stats
            