#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:10:48 2018

@author: nsde
"""

#%%
class ArgsStruct:
    def __init__(self, **entries):
        self._entries = entries
        self.__dict__.update(entries)
    
    def __repr__(self):
        return "Argument structure"
    
    def __str__(self):
        s = 70*'-' + '\n'
        s += 'Running script with input arguments \n'
        for item, val in self._entries.items():
            s += str(item).ljust(15) + ': ' + str(val) + '\n'
        s += 70*'-' + '\n'
        return s

#%%
def lmnn_argparser( ):
    import argparse 
    parser = argparse.ArgumentParser(description='''Argument parser for the 
                lmnn model class. Contains the most important parameters for the class''',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument('-k', action="store", dest='k', type=int,
                        default=3, help='''Number of neighbours''')
    parser.add_argument('-e', action="store", dest='n_epochs', type=int,
                        default=50, help='''epochs''')
    parser.add_argument('-b', action="store", dest='batch_size', type=int,
                        default=200, help='''batch size''')
    parser.add_argument('-m', action="store", dest='margin', type=float,
                        default=1.0, help='''margin''')
    parser.add_argument('-l', action="store", dest='lr', type=float,
                        default=1e-4, help='''learning rate''')
    parser.add_argument('-w', action="store", dest='mu', type=float,
                        default=0.5, help='''mu''')
    parser.add_argument('-s', action="store", dest='snapshot', type=int,
                        default=10, help='''snapshot epoch''')
    parser.add_argument('-r', action="store", dest='log_folder', type=str,
                        default='res', help='''where to store final results''')
    args = parser.parse_args() 
    args = vars(args) 
    args = ArgsStruct(**args)
    return args

#%%
if __name__ == '__main__':
    args = lmnn_argparser()
