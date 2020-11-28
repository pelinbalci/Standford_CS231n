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
# Batch Normalization: Initialization
############

np.random.seed(231)
# Try training a very deep net with batchnorm
hidden_dims = [50, 50, 50, 50, 50, 50, 50]
num_train = 1000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

bn_solvers_ws = {}
solvers_ws = {}
weight_scales = np.logspace(-4, 0, num=20)
for i, weight_scale in enumerate(weight_scales):
    print('Running weight scale %d / %d' % (i + 1, len(weight_scales)))
    bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization='batchnorm')
    model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization=None)

    bn_solver = Solver(bn_model, small_data,
                  num_epochs=10, batch_size=50,
                  update_rule='adam',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  verbose=False, print_every=200)
    bn_solver.train()
    bn_solvers_ws[weight_scale] = bn_solver

    solver = Solver(model, small_data,
                  num_epochs=10, batch_size=50,
                  update_rule='adam',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  verbose=False, print_every=200)
    solver.train()
    solvers_ws[weight_scale] = solver


# Plot results of weight scale experiment
best_train_accs, bn_best_train_accs = [], []
best_val_accs, bn_best_val_accs = [], []
final_train_loss, bn_final_train_loss = [], []

for ws in weight_scales:
    best_train_accs.append(max(solvers_ws[ws].train_acc_history))
    bn_best_train_accs.append(max(bn_solvers_ws[ws].train_acc_history))

    best_val_accs.append(max(solvers_ws[ws].val_acc_history))
    bn_best_val_accs.append(max(bn_solvers_ws[ws].val_acc_history))

    final_train_loss.append(np.mean(solvers_ws[ws].loss_history[-100:]))
    bn_final_train_loss.append(np.mean(bn_solvers_ws[ws].loss_history[-100:]))

fig = plt.figure(figsize=(25, 20))
ax1 = fig.add_subplot(3,1,1)
ax1.set_title('Best val accuracy vs weight initialization scale')
ax1.set_xlabel('Weight initialization scale')
ax1.set_ylabel('Best val accuracy')
ax1.semilogx(weight_scales, best_val_accs, '-o', label='baseline')
ax1.semilogx(weight_scales, bn_best_val_accs, '-o', label='batchnorm')
ax1.legend(ncol=2, loc='lower right')

ax2 = fig.add_subplot(3,1,2)
ax2.set_title('Best train accuracy vs weight initialization scale')
ax2.set_xlabel('Weight initialization scale')
ax2.set_ylabel('Best training accuracy')
ax2.semilogx(weight_scales, best_train_accs, '-o', label='baseline')
ax2.semilogx(weight_scales, bn_best_train_accs, '-o', label='batchnorm')
ax2.legend()

ax3 = fig.add_subplot(3,1,3)
ax3.set_title('Final training loss vs weight initialization scale')
ax3.set_xlabel('Weight initialization scale')
ax3.set_ylabel('Final training loss')
ax3.semilogx(weight_scales, final_train_loss, '-o', label='baseline')
ax3.semilogx(weight_scales, bn_final_train_loss, '-o', label='batchnorm')
ax3.legend()
plt.gca().set_ylim(1.0, 3.5)

fig = plt.gcf()
save_fig('normalization', fig, 'initialization')

#######

# plt.subplot(3, 1, 1)
# plt.title('Best val accuracy vs weight initialization scale')
# plt.xlabel('Weight initialization scale')
# plt.ylabel('Best val accuracy')
# plt.semilogx(weight_scales, best_val_accs, '-o', label='baseline')
# plt.semilogx(weight_scales, bn_best_val_accs, '-o', label='batchnorm')
# plt.legend(ncol=2, loc='lower right')
#
# plt.subplot(3, 1, 2)
# plt.title('Best train accuracy vs weight initialization scale')
# plt.xlabel('Weight initialization scale')
# plt.ylabel('Best training accuracy')
# plt.semilogx(weight_scales, best_train_accs, '-o', label='baseline')
# plt.semilogx(weight_scales, bn_best_train_accs, '-o', label='batchnorm')
# plt.legend()
#
# plt.subplot(3, 1, 3)
# plt.title('Final training loss vs weight initialization scale')
# plt.xlabel('Weight initialization scale')
# plt.ylabel('Final training loss')
# plt.semilogx(weight_scales, final_train_loss, '-o', label='baseline')
# plt.semilogx(weight_scales, bn_final_train_loss, '-o', label='batchnorm')
# plt.legend()
# plt.gca().set_ylim(1.0, 3.5)
#
# plt.gcf().set_size_inches(15, 15)
# plt.show()


# Explanation from: https://github.com/jariasf/CS231n/blob/master/assignment2/BatchNormalization.ipynb
#
# The second plot shows the problem of vanishing gradients (small initial weights).
# The baseline model is very sensitive to this problem (the accuracy is very low),
# therefore finding the correct weight scale is difficult.
# For this example, the baseline obtains the best result with a weight scale equal to 1e-1.
# On the other hand, we can see that the batchnorm model is less sensitive to weight initialization because
# its accuracy is around 30% for all the different weight scales.
#
# The behaviour of the first plot is very similar to that of the second plot.
# The main difference is that the first plot shows that we are overfitting our model,
# besides that we can see that with the batchnorm model we obtained better results than the baseline model
# and that occurs because batch normalization has regularization properties.
#
# The third plot depicts the problem of exploding gradients and it is very evident in the baseline model
# for weight scale values greater than 1e-1. However, the batchnorm model does not suffer from this problem.