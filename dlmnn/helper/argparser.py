#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:10:48 2018

@author: nsde
"""

#%%
class _Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

#%%
def lmnn_argparser( ):
    import argparse 
    parser = argparse.ArgumentParser(description='''Something''') 
    parser.add_argument('-k', action="store", dest='k', type=int,
                        default=1, help='''Number of neighbours''')
    parser.add_argument('-e', action="store", dest='n_epochs', type=int,
                        default=10, help='''epochs''')
    parser.add_argument('-b', action="store", dest='batch_size', type=int,
                        default=100, help='''batch size''')
    parser.add_argument('-m', action="store", dest='margin', type=float,
                        default=1.0, help='''margin''')
    parser.add_argument('-l', action="store", dest='lr', type=float,
                        default=1e-4, help='''learning rate''')
    parser.add_argument('-w', action="store", dest='weight', type=float,
                        default=0.5, help='''mu''')
    parser.add_argument('-r', action="store", dest='log_folder', type=str,
                        default='res', help='''where to store final results''')
    parser.add_argument('-n', action="store", dest='normalization', type=bool,
                        default=True, help='''L2 normalize features''')
    args = parser.parse_args() 
    args = vars(args) 
    args = _Struct(**args)
    return args

#%%
if __name__ == '__main__':
    args = lmnn_argparser()
