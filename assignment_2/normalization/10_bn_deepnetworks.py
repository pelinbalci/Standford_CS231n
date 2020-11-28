# Ref: https://colab.research.google.com/drive/10Yu8mK_phAwwXu4tAEDWVrAKbmOKU1pu#scrollTo=j6d4qotx9Fa4

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from assignment_2.cs231n.data_utils import get_CIFAR10_data
from assignment_2.cs231n.layers import *
from assignment_2.cs231n.layer_utils import *
from assignment_2.cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from assignment_2.cs231n.fc_nets import *
from assignment_2.cs231n.solver import *
from assignment_2.cs231n.plot_history import *
from assignment_2.helper import *

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

###########
# Load the raw CIFAR-10 data.
###########
data = get_CIFAR10_data()
for k, v in list(data.items()):
  print(('%s: ' % k, v.shape))


def print_mean_std(x,axis=0):
    print('  means: ', x.mean(axis=axis))
    print('  stds:  ', x.std(axis=axis))
    print()


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

############
# Batch Normalization: Deep Networks
############
np.random.seed(231)
# Try training a very deep net with batchnorm
hidden_dims = [100, 100, 100, 100, 100]

num_train = 1000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

weight_scale = 2e-2
bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization='batchnorm')
model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization=None)

print('Solver with batch norm:')
bn_solver = Solver(bn_model, small_data,
                num_epochs=10, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True,print_every=20)
bn_solver.train()

print('\nSolver without batch norm:')
solver = Solver(model, small_data,
                num_epochs=10, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()

fig = plt.figure(figsize=(25, 20))
ax1 = fig.add_subplot(3,1,1)
plot_training_history('Training loss', 'Iteration', solver, [bn_solver], \
                      lambda x: x.loss_history, bl_marker='o', bn_marker='o')
ax2 = fig.add_subplot(3, 1, 2)
plot_training_history('Training accuracy', 'Epoch', solver, [bn_solver], \
                      lambda x: x.train_acc_history, bl_marker='-o', bn_marker='-o')
ax2 = fig.add_subplot(3, 1, 3)
plot_training_history('Validation accuracy', 'Epoch', solver, [bn_solver], \
                      lambda x: x.val_acc_history, bl_marker='-o', bn_marker='-o')

fig = plt.gcf()
save_fig('normalization', fig, 'deep_networks')

# plt.subplot(3, 1, 1)
# plot_training_history('Training loss', 'Iteration', solver, [bn_solver], \
#                       lambda x: x.loss_history, bl_marker='o', bn_marker='o')
# plt.subplot(3, 1, 2)
# plot_training_history('Training accuracy', 'Epoch', solver, [bn_solver], \
#                       lambda x: x.train_acc_history, bl_marker='-o', bn_marker='-o')
# plt.subplot(3, 1, 3)
# plot_training_history('Validation accuracy', 'Epoch', solver, [bn_solver], \
#                       lambda x: x.val_acc_history, bl_marker='-o', bn_marker='-o')
#
# plt.gcf().set_size_inches(15, 15)
# plt.show()