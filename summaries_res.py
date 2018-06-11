#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:45:45 2018

@author: nsde
"""

#%%
import numpy as np
from prettytable import PrettyTable
import os

#%%
def load_res(folder):
    results = dict()
    for f in os.listdir(folder):
        if 'performance' in f:
            name = '_'.join(f.split('_')[1:])[:-4]
            res=np.load(folder+'/'+f)
            results[name] = np.round(res, 3)
    
    return results
#%%
if __name__ == '__main__':
    folder='dlmnn'
    results = load_res(folder)
    
    t = PrettyTable(['Dataset', 'KNN', 'CONV', 'CONV-KNN', 'LMNN'])
    for key, value in results.items():
        t.add_row([key, *value])
    print(t)
    