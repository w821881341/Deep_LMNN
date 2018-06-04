#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 08:26:14 2018

@author: nsde
"""
#%%
o = ' -o adam '
lr = ' -lr 1e-3 '
ne = ' -ne 20000 '
k = ' -k 2 '
mu = ' -mu 0.5 '
ss = ' -ss 100'
m = ' -m 1 '
p = ' -p 105 '

#%%
import os
os.system('python LMNN.py -ds olivetti -mt mahalanobis -bs 280' + 
          o + lr + ne + k + mu + ss + m + p)
os.system('python LMNN.py -ds olivetti -mt mahalanobis -bs 140' + 
          o + lr + ne + k + mu + ss + m + p)
os.system('python LMNN.py -ds olivetti -mt mahalanobis -bs 70' + 
          o + lr + ne + k + mu + ss + m + p)
os.system('python LMNN.py -ds olivetti -mt mahalanobis -bs 35' + 
          o + lr + ne + k + mu + ss + m + p)

