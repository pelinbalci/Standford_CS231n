import torch
#assert '.'.join(torch.__version__.split('.')[:2]) == '1.4'
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F  # useful stateless functions
import torch.nn as nn

from assignment_2.helper import *

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


#######
# Load Data
######

# Path of the raw CIFAR-10 data.
cifar10_dir = 'Standford_CS231n/assignment_2/data/cifar-10-batches-py'

NUM_TRAIN = 49000

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.

# Below you can find how to calculate mean and std
# Ref: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/11
import torchvision
train_transform = T.Compose([T.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root=cifar10_dir, train=True,download=True, transform=train_transform)
print(train_set.data.shape)               # (50000, 32, 32, 3)
print(train_set.data.mean(axis=(0, 1, 2))/255)     # [0.49139968 0.48215841 0.44653091]
print(train_set.data.std(axis=(0, 1, 2))/255)      # [0.24703223 0.24348513 0.26158784]
# we are using the train data mean and std for all transformations for train, validation and test.


transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms mini batches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10(cifar10_dir, train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10(cifar10_dir, train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10(cifar10_dir, train=False, download=True,
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

###########
# Check GPU
###########

USE_GPU = True

dtype = torch.float32  # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)

#############
# Flatten the shape
#############

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

# We need to wrap `flatten` function in a module in order to stack it
# in nn.Sequential

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

#############
# Check Accuracy
#############

def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


#############
# Train the model
#############


def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    valid_acc_list = []
    for e in range(epochs):
        model.train()
        val_acc_it = 0
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()
            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            model.eval()
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                valid_acc = check_accuracy_part34(loader_val, model)
                print()

        valid_acc_list.append(valid_acc)
    return valid_acc_list

#######
# Initialization
#######


def init_weights_zeros(m):
    if type(m) == nn.Linear:
        torch.nn.init.zeros_(m.weight)
        torch.nn.init.constant_(m.bias, 0.01)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.01)


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)
        torch.nn.init.constant_(m.bias, 0.01)


def init_weights_kaiming(m):
    torch.nn.init.kaiming_uniform_(m.weight)
    torch.nn.init.constant_(m.bias, 0)

############
# Create Seq Model


def base_model(params):
    channel_1, channel_2, filter_1, filter_2, pad_1, pad_2 = params
    H_out = int((32 + 2 * pad_1 - filter_1) / 1 + 1)
    W_out = int((32 + 2 * pad_2 - filter_2) / 1 + 1)

    model = nn.Sequential(
        nn.Conv2d(3, channel_1, filter_1, padding=pad_1),
        nn.ReLU(),
        nn.Conv2d(channel_1, channel_2, filter_2, padding=pad_2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(channel_2 * H_out * W_out, 10),
    )

    #model.apply(init_weights_xavier)
    return model


params = 16, 8, 3, 3, 1, 1
learning_rate, epoch = 3e-3, 2
optimizer_name = 'SGD'
info = '16,16 layer - 3,3 filter'
plt_name = 'layer_dec'

model = base_model(params)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
valid_acc_list = train_part34(model, optimizer, epochs=epoch)
plot_loss(optimizer_name, valid_acc_list, info, plt_name, epoch)


def model_maxpool(params):
    channel_1, channel_2, filter_1, filter_2, pad_1, pad_2 = params
    H_out = int((32 + 2 * pad_1 - filter_1) / 1 + 1)
    W_out = int((32 + 2 * pad_2 - filter_2) / 1 + 1)
    H_out = int(H_out/4)
    W_out = int(W_out/4)

    model = nn.Sequential(
        nn.Conv2d(3, channel_1, filter_1, padding=pad_1),
        nn.ReLU(),
        nn.Conv2d(channel_1, channel_2, filter_2, padding=pad_2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(channel_2 * H_out * W_out, 10),
    )
    return model


# params = 64, 64, 3, 3, 1, 1
# learning_rate, epoch = 3e-3, 1
# optimizer_name = 'SGD'
# info = '64,64 layer - 3,3 filter'
# plt_name = 'inc_layer-dec_filt_size'
#
# model = model_maxpool(params)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
# valid_acc_list = train_part34(model, optimizer, epochs=epoch)
# print('max acc: ', max(valid_acc_list), ' last acc: ', valid_acc_list[-1])
# plot_loss(optimizer_name, valid_acc_list, info, plt_name, epoch)


#####
# Best Model
#####

def model_maxpool_drop_bn_three(params):
    channel_1, channel_2, channel_3, filter_1, filter_2, filter_3, pad_1, pad_2, pad_3 = params
    H_out = int((32 + 2 * pad_1 - filter_1) / 1 + 1)
    W_out = int((32 + 2 * pad_2 - filter_2) / 1 + 1)
    H_out = int(H_out / 8)
    W_out = int(W_out / 8)

    model = nn.Sequential(
        nn.Conv2d(3, channel_1, filter_1, padding=pad_1),
        nn.BatchNorm2d(channel_1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(channel_1, channel_2, filter_2, padding=pad_2),
        nn.BatchNorm2d(channel_2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(channel_2, channel_3, filter_3, padding=pad_3),
        nn.BatchNorm2d(channel_3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        Flatten(),
        nn.Dropout(0.25),
        nn.Linear(channel_3 * H_out * W_out, 10),
    )
    return model


# params = 128, 64, 32, 3, 3, 3, 1, 1, 1
# learning_rate, epoch = 1e-3, 1
# optimizer_name = 'Adam'
# info = '128,64 layer - 3,3 filter'
# plt_name = 'Increase hidden layer'
#
# model = model_maxpool_drop_bn_three(params)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
# valid_acc_list = train_part34(model, optimizer, epochs=epoch)
# print('max acc: ', max(valid_acc_list), ' last acc: ', valid_acc_list[-1])
# plot_loss(optimizer_name, valid_acc_list, info, plt_name, epoch)
