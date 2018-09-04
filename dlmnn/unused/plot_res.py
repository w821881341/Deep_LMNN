# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 09:59:31 2018

@author: nsde
"""


import numpy as np
import matplotlib.pyplot as plt

# Transform and rotate
X_trans2 = model.transform(X_test)
u,s,v = np.linalg.svd(np.random.normal(size=(128, 128)))
X_rot = np.dot(X_trans2, u)

fig, ax = plt.subplots(1,7)
for i in range(6):
    plot=ax[i].imshow(X_trans1[500*i:500*(i+1)]>0.3)
    ax[i].axis('off')
ax[6].axis('off')
fig.colorbar(plot, ax=ax[6])
fig.suptitle('Before learning', fontsize=20)
plt.show()

fig, ax = plt.subplots(1,7)
for i in range(6):
    plot=ax[i].imshow(X_trans2[500*i:500*(i+1)])
    ax[i].axis('off')
ax[6].axis('off')
fig.colorbar(plot, ax=ax[6])
fig.suptitle('After learning', fontsize=20)
plt.show()


fig, ax = plt.subplots(1,7)
for i in range(6):
    ax[i].imshow(X_rot[500*i:500*(i+1)])
    ax[i].axis('off')
ax[6].axis('off')
fig.colorbar(plot, ax=ax[6])
fig.suptitle('After learning - rotated', fontsize=20)
plt.show()

def sparsity_index_tanh(X):
    return np.mean(np.sum(np.tanh(X**2), axis=1))

def sparsity_index_l0(X):
    return np.mean(np.sum(X==0, axis=1))

print('Before learning:         ', 
      sparsity_index_tanh(X_trans1), sparsity_index_l0(X_trans1))
print('After learning:          ', 
      sparsity_index_tanh(X_trans2), sparsity_index_l0(X_trans2))
print('After learning - rotated:', 
      sparsity_index_tanh(X_rot), sparsity_index_l0(X_rot))