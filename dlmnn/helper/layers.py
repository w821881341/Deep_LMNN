# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 02:40:18 2018

@author: nsde
"""

#%% easy access to keras layers
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Dense 
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import LeakyReLU

from tensorflow.python.keras.layers.core import Layer as _Layer
from tensorflow.python.keras import backend as _K

#%%
class layerlist(object):
    ''' Utility structure for holding layers. Layers are added to the structure
        using the add method, and then using the get_layer method will always
        return a new copy of the layer '''
    def __init__(self):
        self.layers = [ ]
        self.layers_args = [ ]
        
    def add(self, layer, **kwargs):
        self.layers.append(layer)
        self.layers_args.append(kwargs)
        
    def get_layer(self, index):
        return self.layers[index](**self.layers_args[index])
    
    @property
    def n_layers(self):
        return len(self.layers)


#%%
class L2normalize(_Layer):
    def __init__(self, **kwargs):
        super(L2normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(L2normalize, self).build(input_shape)

    def call(self, X):
        return _K.l2_normalize(X, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (None, input_shape[1])
        
#%%
if __name__ == '__main__':
    from tensorflow.python.keras import Sequential
    s = Sequential()
    s.add(InputLayer(input_shape=(30, 30, 3)))
    s.add(Conv2D(16, 3))
    s.add(MaxPool2D((2,2)))
    s.add(Flatten())
    s.add(Dense(32))
    s.add(LeakyReLU(alpha=0.3))
    s.add(L2normalize())