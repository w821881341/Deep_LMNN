#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 08:53:45 2018

@author: nsde
"""

from . import get_img_data
from . import get_point_data

def download_all():
    _ = get_img_data.get_mnist()
    _ = get_img_data.get_mnist_distorted()
    _ = get_img_data.get_mnist_fashion()
    _ = get_img_data.get_devanagari()
    _ = get_img_data.get_olivetti()
    _ = get_img_data.get_cifar10()
    _ = get_img_data.get_cifar100()
    
    

    