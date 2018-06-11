#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 08:26:14 2018

@author: nsde
"""

#%%
import os
os.system("PYTHONPATH='/home/nsde/Documents/Deep_LMNN' python benchmarks.py -d mnist")
os.system("PYTHONPATH='/home/nsde/Documents/Deep_LMNN' python benchmarks.py -d mnist_distorted")
os.system("PYTHONPATH='/home/nsde/Documents/Deep_LMNN' python benchmarks.py -d mnist_fashion")
os.system("PYTHONPATH='/home/nsde/Documents/Deep_LMNN' python benchmarks.py -d devanagari")
os.system("PYTHONPATH='/home/nsde/Documents/Deep_LMNN' python benchmarks.py -d olivetti")
os.system("PYTHONPATH='/home/nsde/Documents/Deep_LMNN' python benchmarks.py -d cifar10")
os.system("PYTHONPATH='/home/nsde/Documents/Deep_LMNN' python benchmarks.py -d cifar100")
