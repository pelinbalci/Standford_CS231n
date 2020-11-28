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
# BN vs LN
############


def run_batchsize_experiments():
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
    n_epochs = 10
    weight_scale = 2e-2
    b_size = 10
    lr = 10 ** (-3.5)
    normal_modes = ['layernorm', 'batchnorm']
    solver_bnorm = normal_modes[0]

    print('No normalization: batch size = ', solver_bnorm)
    model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization=None)
    solver = Solver(model, small_data,
                    num_epochs=n_epochs, batch_size=b_size,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': lr,
                    },
                    verbose=False)
    solver.train()

    bn_solvers = []
    for i in range(len(normal_modes)):
        norm_mode = normal_modes[i]
        print('Normalization: type = ', norm_mode)
        bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization=norm_mode)
        bn_solver = Solver(bn_model, small_data,
                           num_epochs=n_epochs, batch_size=b_size,
                           update_rule='adam',
                           optim_config={
                               'learning_rate': lr,
                           },
                           verbose=False)
        bn_solver.train()
        bn_solvers.append(bn_solver)

    return bn_solvers, solver, normal_modes


norm_solvers, solver_bnorm, normal_modes = run_batchsize_experiments()

fig = plt.figure(figsize=(25, 20))
ax1 = fig.add_subplot(2,1,1)
plot_training_history('Training accuracy (Layer Normalization)','Epoch', solver_bnorm, norm_solvers,\
                      lambda x: x.train_acc_history, bl_marker='-^', bn_marker='-o', labels=normal_modes)

ax2 = fig.add_subplot(2,1,2)
plot_training_history('Validation accuracy (Layer Normalization)','Epoch', solver_bnorm, norm_solvers,\
                      lambda x: x.val_acc_history, bl_marker='-^', bn_marker='-o', labels=normal_modes)

fig = plt.gcf()
save_fig('normalization', fig, 'bn_vs_ln')