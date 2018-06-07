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
def embedding_projector(np_tensor, folder, name='embedding', imgs=None, labels=None):
    """ Tensorboard embedding projector
    Arguments:
        embedding_var: tensorboard variable to embed
        path: string, where to save the embedding files
        imgs (optional): if data comes from images, these can be given as a
            numpy array, such that they can be visualized with the embeddings
        labels (optional): if data is labeled, these can be given, such that
            embeddings can be colored
    """
    tf_tensor = tf.cast(np_tensor, tf.float32)
    embedding_var = tf.Variable(tf_tensor, name=name, trainable=False)
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer()) 
        saver = tf.train.Saver([embedding_var]) 
        saver.save(sess, folder + '/' + name + '.ckpt') 
    
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name 
    
    # If images are supplied, create sprite image, and save it for the embeddings
    if imgs is not None:
        s, h, w, nc = create_sprite(imgs)
        s = s[:,:,0] if nc == 1 else s
        plt.imsave(folder + '/'+name+'_sprite.png', s, cmap='gray' if nc==1 else 'color')
        embedding.sprite.image_path = name+'_sprite.png'
        embedding.sprite.single_image_dim.extend([w, h])
    
    # labels are supplied, create metadata        
    if labels is not None:
        write_metadata(folder + '/'+name+'_metadata.tsv', labels)
        embedding.metadata_path = name+'_metadata.tsv'
    
    # Connect summary writer with projector
    summary_writer = tf.summary.FileWriter(folder, filename_suffix=name)
    projector.visualize_embeddings(summary_writer, config)
    
    # Close
    summary_writer.close()