# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:23:37 2017

@author: nsde
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, struct, argparse, sys

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

#%%
def colorise(string, color='green'):
    if color=='green': col='0;32;40m'
    elif color=='blue': col='0;34;40m'
    elif color=='red': col='0;31;40m'
    begin = '\x1b['
    end = '\033[0m'
    return begin + col + string + end

#%%
def progressBar(value, endvalue, name = 'Process', bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0}: [{1}] {2}%".format(name, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
    
#%%
def plot_normal2D(mu, sigma):
    from scipy.linalg import eig
    eigVal, eigVec = eig(sigma)
    eigVal = np.maximum(0, np.real(np.diag(eigVal)))
    t = np.linspace(0, 2*np.pi, 100)
    xy = np.array([np.cos(t), np.sin(t)])
    Txy = eigVec.dot(np.sqrt(eigVal)).dot(xy).T + mu
    plt.plot(Txy[:,0], Txy[:,1], 'b-')

#%%
def rand_posdef_mat(d):
    ''' Generate random positive semi-definite matrix of size d x d '''
    mat = np.random.normal(size=(d,d))
    mat = np.dot(mat, mat)
    return mat

#%%
def random_not_in_sampler(size, N_range, array):
    samples = [ ]
    count = 0
    while count < size:
        i = np.random.randint(N_range)
        if i not in array and i not in samples:
            samples.append(i)
            count += 1
    return np.array(samples)

#%%
def get_dir(file):
    return os.path.dirname(os.path.realpath(file))

#%%
def get_optimizer(name):
    ''' Returns a tensorflow optimizer based on name '''
    try:
        optimizer = {'adam':     tf.train.AdamOptimizer,
                     'sgd':      tf.train.GradientDescentOptimizer,
                     'momentum': tf.train.MomentumOptimizer
                     }[name]
        return optimizer
    except KeyError:
        raise Exception(name + ' is a invalid option to input optimizer')

#%%
def adjust_learning_rate(alpha, loss_new, loss_old):
    ''' Function for adjusting the learning rate while training '''
    out_alpha = alpha*1.01 if loss_new <= loss_old else alpha*0.5
    return out_alpha

#%%
def read_mnist(dataset = "training", path = "MNIST/"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return img, lbl

#%%
def read_olivetti(path = "OLIVETTI/", split=True, permute = True):
    from sklearn.datasets import fetch_olivetti_faces
    obj = fetch_olivetti_faces(data_home=path, )
    X = obj['data']
    y = obj['target']
    if split:
        X_train = np.zeros((280, 4096), dtype=np.float32)
        X_test = np.zeros((120, 4096), dtype=np.float32)
        y_train = np.zeros((280, ), dtype=np.int64)
        y_test = np.zeros((120, ), dtype=np.int64)
        for i in range(40):
            rand_idx=np.random.permutation(range(10*i,10*(i+1)))
            X_train[7*i:7*(i+1)]=X[rand_idx[:7]]
            X_test[3*i:3*(i+1)]=X[rand_idx[7:]]
            y_train[7*i:7*(i+1)]=y[rand_idx[:7]]
            y_test[3*i:3*(i+1)]=y[rand_idx[7:]]
        if permute:
            idx = np.random.permutation(280)
            X_train = X_train[idx]
            y_train = y_train[idx]
            idx = np.random.permutation(120)
            X_test = X_test[idx]
            y_test = y_test[idx]
        return X_train, y_train, X_test, y_test
    else:
        if permute:
            idx = np.random.permutation(400)
            X = X[idx]
            y = y[idx]
        return X, y

#%%
def test_set1(size = 50, plot = False):
    from sklearn.model_selection import train_test_split
    mu = [(1,100), (2,100), (3,100), (4,100)]
    std = np.array([[0.05, 0], [0, 50]])
    
    X = [ ]
    y = [ ]
    for i in range(len(mu)):
        for _ in range(size):
            X.append(np.random.multivariate_normal(mean = mu[i], cov = std))
            y.append(i)       
    X = np.array(X)
    y = np.array(y)
    
    if plot:
        col = ['red', 'blue', 'green', 'orange']
        plt.figure()
        plt.subplot(1,2,1)
        for i in range(X.shape[0]):
            plt.plot(X[i,0], X[i,1], '.', color=col[y[i]])
        plt.axis('equal')
    
        plt.subplot(1,2,2)
        for i in range(X.shape[0]):
            plt.plot(X[i,0], X[i,1], '.', color=col[y[i]])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)    
    return X_train, y_train, X_test, y_test

#%%
def test_set2(size = 50, plot = False):
    mu = [(1,100), (3,100), (5,100), (7,100)]
    std = np.array([[0.05, 0], [0, 50]])
    
    X = [ ]
    y = [ ]
    for i in range(len(mu)):
        for _ in range(size):
            X.append(np.random.multivariate_normal(mean = mu[i], cov = std))
            y.append(i)       
    X = np.array(X)
    y = np.array(y)
    return X, y



#%%
if __name__ == '__main__':
    print(get_dir(__file__))