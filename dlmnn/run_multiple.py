#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 08:26:14 2018

@author: nsde
"""

#%%
import os
os.system('python LMNN.py -ds olivetti_conv -mt conv -o adam -lr 1e-3 -ne 20000 -bs 280 -k 2 -mu 0.5 -ss 100 -m 1 -p 3 3 1 10')
os.system('python LMNN.py -ds olivetti_conv -mt conv -o adam -lr 1e-3 -ne 20000 -bs 140 -k 2 -mu 0.5 -ss 100 -m 1 -p 3 3 1 10')
os.system('python LMNN.py -ds olivetti_conv -mt conv -o adam -lr 1e-3 -ne 20000 -bs 70 -k 2 -mu 0.5 -ss 100 -m 1 -p 3 3 1 10')
os.system('python LMNN.py -ds olivetti_conv -mt conv -o adam -lr 1e-3 -ne 20000 -bs 35 -k 2 -mu 0.5 -ss 100 -m 1 -p 3 3 1 10')
os.system('python LMNN.py -ds olivetti -mt mahalanobis -o adam -lr 1e-3 -ne 20000 -bs 280 -k 2 -mu 0.5 -ss 100 -m 1 -p 105')
os.system('python LMNN.py -ds olivetti -mt mahalanobis -o adam -lr 1e-3 -ne 20000 -bs 140 -k 2 -mu 0.5 -ss 100 -m 1 -p 105')
os.system('python LMNN.py -ds olivetti -mt mahalanobis -o adam -lr 1e-3 -ne 20000 -bs 70 -k 2 -mu 0.5 -ss 100 -m 1 -p 105')
os.system('python LMNN.py -ds olivetti -mt mahalanobis -o adam -lr 1e-3 -ne 20000 -bs 35 -k 2 -mu 0.5 -ss 100 -m 1 -p 105')


