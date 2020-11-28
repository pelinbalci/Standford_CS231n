"""
RMSProp [1] and Adam [2] are update rules that set per-parameter learning rates by using a running average of the second
moments of gradients.

In the file cs231n/optim.py, implement the RMSProp update rule in the rmsprop function and implement the Adam update
rule in the adam function, and check your implementations using the tests below.

NOTE: Please implement the complete Adam update rule (with the bias correction mechanism), not the first simplified
version mentioned in the course notes.

[1] Tijmen Tieleman and Geoffrey Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent
magnitude." COURSERA: Neural Networks for Machine Learning 4 (2012).

[2] Diederik Kingma and Jimmy Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.
"""

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


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

#############
# Different Models
#############
num_train = 4000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

solvers = {}

# Original
# ['sgd', 'sgd_momentum', 'adam', 'rmsprop']
# learning_rates = {'sgd': 5e-3, 'sgd_momentum':  5e-3, 'rmsprop': 1e-4, 'adam': 1e-3}
# learning_rates = {'sgd': 1e-3, 'sgd_momentum':  1e-3, 'rmsprop': 1e-3, 'adam': 1e-3}

# Trials
# ['sgd', 'sgd', 'sgd', 'sgd']
# {'sgd': 5e-3, 'sgd':  3e-3, 'sgd': 1e-3, 'sgd': 1e-2}

# ['sgd_momentum', 'sgd_momentum', 'sgd_momentum', 'sgd_momentum']
# {'sgd_momentum': 5e-3, 'sgd_momentum':  3e-3, 'sgd_momentum': 1e-3, 'sgd_momentum': 1e-2}

# ['rmsprop', 'rmsprop', 'rmsprop', 'rmsprop']
# {'rmsprop': 1e-4, 'rmsprop':  5e-4, 'rmsprop': 5e-3, 'rmsprop': 1e-3}

# ['adam', 'adam', 'adam', 'adam']
# {'adam': 1e-4, 'adam':  5e-4, 'adam': 5e-3, 'adam': 1e-3}


learning_rates = {'sgd': 5e-3, 'sgd_momentum':  5e-3, 'rmsprop': 1e-4, 'adam': 1e-3}
for update_rule in ['sgd', 'sgd_momentum', 'adam', 'rmsprop']:
  print('running with ', update_rule)
  model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)

  solver = Solver(model, small_data,
                  num_epochs=5, batch_size=100,
                  update_rule=update_rule,
                  optim_config={
                    'learning_rate': learning_rates[update_rule]
                  },
                  verbose=True)
  solvers[update_rule] = solver
  solver.train()
  print()

plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')

plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')

plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')

for update_rule, solver in list(solvers.items()):
  plt.subplot(3, 1, 1)
  plt.plot(solver.loss_history, 'o', label=update_rule)

  plt.subplot(3, 1, 2)
  plt.plot(solver.train_acc_history, '-o', label=update_rule)

  plt.subplot(3, 1, 3)
  plt.plot(solver.val_acc_history, '-o', label=update_rule)

for i in [1, 2, 3]:
  plt.subplot(3, 1, i)
  plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()