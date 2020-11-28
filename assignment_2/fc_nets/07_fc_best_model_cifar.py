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

solvers = {}
best_model = None
best_val = -10000

#learning_rates = {'rmsprop': 1e-4, 'adam': 1e-3}

for update_rule in ['adam', 'rmsprop', 'sgd', 'sgd_momentum']:
    for i in range(3):
        ws = 10 ** np.random.uniform(-2, -1)  # Standard deviation (spread or “width”) of the distribution.
        lr = 10 ** np.random.uniform(-5, -2)  # learning rate
        reg = 10 ** np.random.uniform(-3, 3)  # regularization
        model = FullyConnectedNet([100, 100], weight_scale=ws, reg=reg)

        solver = Solver(model,
                        data,
                        num_epochs=5,
                        batch_size=100,
                        update_rule=update_rule,
                        optim_config={
                            'learning_rate': lr
                        },
                      verbose=False)
        solvers[update_rule] = solver
        solver.train()
        val_accuracy = solver.best_val_acc
        if best_val < val_accuracy:
          best_val = val_accuracy
          best_model = model
        # Print results
        print('update_rule %s lr %e ws %e reg %e val accuracy: %f' % (
            update_rule, lr, ws, reg, val_accuracy))

print('best validation accuracy achieved: %f' % best_val)


y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())

"""
update_rule adam lr 9.388380e-05 ws 1.516331e-02 reg 4.409413e-03 val accuracy: 0.515000
best validation accuracy achieved: 0.515000
Validation set accuracy:  0.515
Test set accuracy:  0.497

"""