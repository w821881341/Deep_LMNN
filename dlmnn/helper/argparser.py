#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 08:52:47 2018

@author: nsde
"""

#%%
import argparse

#%%
def lmnn_argparser( ):
    parser = argparse.ArgumentParser(description='''This program will run the
        LMNN algorithm on a selected dataset''')
    parser.add_argument('-ds', action="store", dest='ds', type=str,
                        default='olivetti', help='''Dataset to use''')
    parser.add_argument('-mt', action="store", dest='mt', type=str,
                        default='mahalanobis', help='''Transformation to learn''')
    parser.add_argument('-p', action="store", dest='p', nargs='+', type=int, 
        default = [2], help = 'Parameters for metric')
    parser.add_argument('-k', action="store", dest="k", type=int,
        default = 3, help = '''Number of target neighbours''')
    parser.add_argument('-o', action="store", dest='o', type=str,
        default = 'sgd', help = 'Optimizer to use')
    parser.add_argument('-lr', action="store", dest="lr", type=float, 
        default = 0.0001, help = '''Learning rate for optimizer''')
    parser.add_argument('-ne', action="store", dest="ne", type=int, 
        default = 10, help = '''Number of epochs''')
    parser.add_argument('-bs', action="store", dest="bs", type=int, 
        default = 100, help = '''Batch size''')
    parser.add_argument('-mu', action="store", dest="mu", type=float,
        default = 0.5, help = '''Weighting parameter in loss function''')
    parser.add_argument('-m', action="store", dest='m', type=float,
        default = 1, help = 'Margin')
    parser.add_argument('-ss', action="store", dest='ss', type=int,
        default = 10, help = 'Snapshot epoch')
    parser.add_argument('-v', action="store", dest='v', type=int,
        default = 1, help = 'Verbose level')
    args = parser.parse_args()
    args = vars(args)
    
    print(50*'-')
    print('Running script with arguments:')
    print('  dataset:            ', args['ds'])
    print('  metric type:        ', args['mt'])
    print('  metric parameters   ', args['p'])
    print('  margin:             ', args['m'])
    print('  optimizer:          ', args['o'])
    print('  target neighbours:  ', args['k'])
    print('  mu parameter:       ', args['mu'])
    print('  batch size:         ', args['bs'])
    print('  number of epochs:   ', args['ne'])
    print('  learning rate:      ', args['lr'])
    print('  verbose level:      ', args['v'])
    print('  snapshot epoch:     ', args['ss'])
    print(50*'-')
    return args