# Copyright 2017 Sanket Kamthe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import matplotlib.pyplot as plt
import numpy as np


def plot_gaussian(data, label=None):
    x_mean = np.array([x.mean for x in data])
    x_sigma = np.array([np.sqrt(x.cov[0, :]) for x in data])

    upr = x_mean + 1.96 * x_sigma
    lwr = x_mean - 1.96 * x_sigma
    time = np.arange(len(data))
    plt.fill_between(time, lwr[:, 0], upr[:, 0], alpha=0.5, label=label)


def plot_gaussian_node(data, ground_truth, figsize=None, ax=None):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    ax.set_facecolor('#EBEBEB')
    ax.grid(True, color='w', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    x_mean = np.array([x.marginal.mean for x in data])
    x_sigma = np.array([np.sqrt(x.marginal.cov[0, :]) for x in data])

    upr = x_mean + 1.96 * x_sigma
    lwr = x_mean - 1.96 * x_sigma
    time = np.arange(1, len(data)+1)
    ax.plot(time, x_mean, 'C0',  linewidth=2.5, label='Prediction')
    ax.fill_between(time, lwr[:, 0], upr[:, 0], alpha=0.3)
    ax.plot(time, ground_truth, 'C3', linewidth=2.5, label='Ground truth')
    return ax


def _get_mean_and_std(data):
    mean = data.mean(axis=0)
    anomaly = data - mean[None]
    variance = (anomaly**2).mean(axis=0)
    std = np.sqrt(variance)
    return mean, std

    
def plot_1d_data(data, label=None, c=None, figsize=None, linewidth=None, ax=None):
    """ Plot mean and std of N samples of 1D data with size T
    :data: Array of size (N, T)
    :label: Plot label
    :c: Color of line
    :figsize: Figure size
    :ax: Plot axis
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.set_facecolor('#EBEBEB')
        ax.grid(True, color='w', linestyle='-', linewidth=1)
        ax.set_axisbelow(True)
    mean, std = _get_mean_and_std(data)
    # Plot mean and std
    N = mean.shape[0]
    ax.plot(np.arange(1, N+1), mean, c=c, label=label, zorder=1, linewidth=linewidth)
    if data.shape[0] > 1:
        ax.fill_between(np.arange(1, N+1), mean-std, mean+std, alpha=0.2, color=c, zorder=2)
    return ax