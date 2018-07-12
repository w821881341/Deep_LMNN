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

from tensorflow.python.keras._impl.keras.layers.core import Layer as _Layer
from tensorflow.python.keras import backend as _K

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
    s.add(Flatten())
    s.add(Dense(32))
    s.add(L2normalize())