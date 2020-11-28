import os
from matplotlib import pyplot as plt
import numpy as np

folder = {'main_folder': os.path.dirname(os.path.abspath(__file__))}


############
# save_figures to folder
############
def save_fig(folder_name, fig, name):
    file_name = name + '.png'
    path = os.path.join(folder['main_folder'], folder_name, file_name)
    fig.savefig(path)


############
# helper function to un-normalize and display an image
############
def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))


def plot_loss(optimizer_name, valid_acc_list, info, plt_name, epoch):
    ######################
    # plot the loss for train and validation #
    ######################
    epochs_list = [i + 1 for i in range(epoch)]
    plt.figure(figsize=(25, 4))
    plt.plot(epochs_list, valid_acc_list)
    plt.plot(epochs_list, valid_acc_list, 'go')
    plt.xticks(np.arange(min(epochs_list), max(epochs_list), step=1))
    plt.title(str(optimizer_name) + ' for ' + str(epoch) + ' epochs ' + str(info) +
              ' validation accuracy is ' + str(valid_acc_list[-1]))
    # save the loss plot
    name = str(optimizer_name) + '_' + str(epoch) + '_' + str(plt_name)
    fig = plt.gcf()
    save_fig('output', fig, name)