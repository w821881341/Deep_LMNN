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
    parser.add_argument('-m', action="store", dest='m', type=float,
                        default=1.0, help='''margin''')
    parser.add_argument('-mu', action="store", dest='mu', type=float,
                        default=0.5, help='''mu''')
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
    m = '-m ' + str(args['m']) + ' '
    mu = '-mu' + str(args['mu']) + ' '
    
    # Run benchmarks
    os.system("python dlmnn/benchmarks.py "+k+e+n+m+mu+"-d mnist")
    os.system("python dlmnn/benchmarks.py "+k+e+n+m+mu+"-d mnist_distorted")
    os.system("python dlmnn/benchmarks.py "+k+e+n+m+mu+"-d mnist_fashion")
    os.system("python dlmnn/benchmarks.py "+k+e+n+m+mu+"-d devanagari")
    os.system("python dlmnn/benchmarks.py "+k+e+n+m+mu+"-d olivetti")
    os.system("python dlmnn/benchmarks.py "+k+e+n+m+mu+"-d cifar10")
    os.system("python dlmnn/benchmarks.py "+k+e+n+m+mu+"-d cifar100")
