#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 08:01:09 2018

@author: nsde
"""
#%%
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import matplotlib.pyplot as plt

from dlmnn.data.get_img_data import get_olivetti
from dlmnn.helper.utility import create_dir

#%%
def create_sprite(imgs):
    """ From a tensor of images, create a single sprite image """
    N, height, width, nchannel = imgs.shape
    rows = cols = int(np.ceil(np.sqrt(N)))
    sprite = np.zeros((height*rows, width*cols, nchannel), dtype=imgs.dtype)
    for i in range(rows):
        for j in range(cols):
            if i*cols+j < N:
                sprite[i*height:(i+1)*height, j*width:(j+1)*width] = imgs[i*cols+j]
    return sprite, height, width, nchannel

#%%
def write_metadata(filename, labels):
    """ Write a tensorboard embedding metadata file with label information """
    with open(filename,'w') as f:
        f.write("Index\tLabel\n")
        for index,label in enumerate(labels):
            f.write("%d\t%d\n" % (index,label))

#%%
def embedding_projector(embedding_var, path, imgs=None, labels=None):
    """ Tensorboard embedding projector
    Arguments:
        embedding_var: tensorboard variable to embed
        path: string, where to save the embedding files
        imgs (optional): if data comes from images, these can be given as a
            numpy array, such that they can be visualized with the embeddings
        labels (optional): if data is labeled, these can be given, such that
            embeddings can be colored
    """
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name 
    
    # If images are supplied, create sprite image, and save it for the embeddings
    if imgs is not None:
        s, h, w, nc = create_sprite(imgs)
        s = s[:,:,0] if nc == 1 else s
        plt.imsave(path + '/sprite.png', s, cmap='gray' if nc==1 else 'color')
        embedding.sprite.image_path = 'sprite.png'
        embedding.sprite.single_image_dim.extend([w, h])
    
    # labels are supplied, create metadata        
    if labels is not None:
        write_metadata(path + '/metadata.tsv', labels)
        embedding.metadata_path = 'metadata.tsv'
    
    # Connect summary writer with projector
    summary_writer = tf.summary.FileWriter(path)
    projector.visualize_embeddings(summary_writer, config)
    
    # Close
    summary_writer.close()
    
    
#%%
if __name__ == '__main__':
    # Set folder where results will be saved
    folder = 'embedding_example'
    create_dir(folder)
    
    Xtrain, Ytrain, Xtest, Ytest = get_olivetti()
    X = np.reshape(Xtrain, (280, -1))
    Y = Ytrain
    
    # Embed training set
    embedding_var = tf.Variable(tf.cast(X, tf.float32), name='embedding')

    # Run session where variable is initialized, then save session        
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver([embedding_var])
        saver.save(sess, folder + '/model.ckpt')

    embedding_projector(embedding_var, 
                        path=folder, 
                        imgs=Xtrain, 
                        labels=Y)
    
    
    
    
    


    