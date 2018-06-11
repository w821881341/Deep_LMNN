#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 08:26:14 2018

@author: nsde
"""
import os

#%%
def argparser( ):
    import argparse 
    parser = argparse.ArgumentParser(description='''Something''') 
    parser.add_argument('-k', action="store", dest='k', type=int,
                        default=1, help='''Number of neighbours''')
    parser.add_argument('-e', action="store", dest='e', type=int,
                        default=10, help='''epochs''')
    parser.add_argument('-n', action="store", dest='n', type=str,
                        default='res', help='''where to store final results''')
    args = parser.parse_args() 
    args = vars(args) 
    return args

#%%
if __name__ == '__main__':
    # Get arguments
    args = argparser()
    k = '-k ' + str(args['k']) + ' '
    e = '-e ' + str(args['e']) + ' '
    n = '-n ' + args['n'] + ' '
    
    # Run benchmarks
    os.system("python benchmarks.py "+k+e+n+"-d mnist")
    os.system("python benchmarks.py "+k+e+n+"-d mnist_distorted")
    os.system("python benchmarks.py "+k+e+n+"-d mnist_fashion")
    os.system("python benchmarks.py "+k+e+n+"-d devanagari")
    os.system("python benchmarks.py "+k+e+n+"-d olivetti")
    os.system("python benchmarks.py "+k+e+n+"-d cifar10")
    os.system("python benchmarks.py "+k+e+n+"-d cifar100")
