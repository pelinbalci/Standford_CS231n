# Ref: https://colab.research.google.com/drive/1hGKmjHMWfMO3Gf3wNmbYiAYx-7T74PMR#scrollTo=iOa9jnvo-dgU

# Run some setup code for this notebook.

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

# Load the raw CIFAR-10 data.
data = get_CIFAR10_data()
for k, v in list(data.items()):
  print(('%s: ' % k, v.shape))


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

#############
# Solver
#############
model = TwoLayerNet()
solver = None

##############################################################################
# TODO: Use a Solver instance to train a TwoLayerNet that achieves at least  #
# 50% accuracy on the validation set.                                        #
##############################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

solver = Solver(model,
                data,
                update_rule='sgd',
                optim_config={'learning_rate': 1e-3,},
                lr_decay=0.95,
                num_epochs=10,
                batch_size=100,
                print_every=100)

solver.train()
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

# Run this cell to visualize training loss and train / val accuracy

fig = plt.figure(figsize=(15, 15))
ax1 = fig.add_subplot(2,1,1)
ax1.set_title('Training loss')
plt.plot(solver.loss_history, 'o')
ax1.set_xlabel('Iteration')

ax2 = fig.add_subplot(2,1,2)
ax2.set_title('Accuracy')
ax2.plot(solver.train_acc_history, '-o', label='train')
ax2.plot(solver.val_acc_history, '-o', label='val')
ax2.plot([0.5] * len(solver.val_acc_history), 'k--')
ax2.set_xlabel('Epoch')
ax2.legend(loc='lower right')
plt.show()

fig = plt.gcf()
save_fig('fc_nets', fig, 'solver')
