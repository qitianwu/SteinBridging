import numpy as np
from scipy.stats import multivariate_normal
import math
import matplotlib.pyplot as plt

cmap = plt.get_cmap('magma')

def plot_sample_density(true_sample, true_density, XY_range, fake_sample=None, fake_density=None, noise_sample=None, method='', it=None, loss=None, save_path=None):
    """
    Show true samples and fake samples with plt.
    # true_sample: the ground-truth samples
    # true_density: the ground_truth distribution that generates true_sample
    # XY_range: X and Y locations for mesh grids
    # fake_sample: the generated samples from G
    # fake_density: the proposal distribution from E
    # noise_sample: noise samples in dataset
    # method: the adopted model
    # it: number of iterations
    # loss: the loss from model
    # save_path: location to save the picture 
    """
    
    # plot true sample
    x, y = true_sample[:, 0], true_sample[:, 1]
    X_range, Y_range = XY_range[0], XY_range[1]
    xlim = [X_range[0][0], X_range[-1][-1]]
    ylim = [Y_range[0][0], Y_range[-1][-1]]
    axis = np.concatenate([xlim, ylim], 0)
    adjust = 0.1  # adjust for fixed windows
    
    x[x < xlim[0]] = xlim[0] + adjust
    x[x > xlim[1]] = xlim[1] - adjust
    y[y < ylim[0]] = ylim[0] + adjust
    y[y > ylim[1]] = ylim[1] - adjust

    tt1 = method + " v.s. True "
    tt2 = method + " v.s. True "

    if fake_sample is None:
        fig, ax = plt.subplots(1, 2, figsize=(7, 7))
        ax[0].scatter(x, y, color='r', alpha=0.2, s=10, label="True Sample")
        ax[0].axis(axis)

        mg = ax[1].pcolormesh(X_range, Y_range, true_density, cmap=cmap)
    
    else:
        fig, ax = plt.subplots(2, 2, figsize=(7, 7))
        ax[0][0].scatter(x, y, color='m', alpha=0.2, s=10, label="True Sample")
        ax[0][0].axis(axis)

        if noise_sample is not None:
            x, y = noise_sample[:, 0], noise_sample[:, 1]
            ax[0][0].scatter(x, y, color='r', alpha=0.2, s=10, label="Noise Sample")

        mg = ax[0][1].pcolormesh(X_range, Y_range, true_density, cmap=cmap)

        x, y = fake_sample[:, 0], fake_sample[:, 1]
        show = (x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1])
        x[x < xlim[0]] = xlim[0] + adjust
        x[x > xlim[1]] = xlim[1] - adjust
        y[y < ylim[0]] = ylim[0] + adjust
        y[y > ylim[1]] = ylim[1] - adjust
        
        ax[1][0].scatter(x, y, color='g', alpha=0.2, s=10, label="Fake Sample")
        ax[1][0].axis(axis)
        ax[1][0].set_title(tt1)
    
        mg = ax[1][1].pcolormesh(X_range, Y_range, fake_density, cmap=cmap)
        ax[1][1].set_title(tt2)

    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches='tight', pad_inches=0)