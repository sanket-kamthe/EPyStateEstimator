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
    # if in_data
    # data = in_data[:, 0:1]
    x_mean = np.array([x.mean for x in data])
    x_sigma = np.array([np.sqrt(x.cov[0, :]) for x in data])

    upr = x_mean + 1.96 * x_sigma
    lwr = x_mean - 1.96 * x_sigma
    time = np.arange(len(data))
    plt.fill_between(time, lwr[:, 0], upr[:, 0], alpha=0.5, label=label)


def plot_gaussian_node(data):
    # if in_data
    # data = in_data[:, 0:1]
    x_mean = np.array([x.marginal.mean for x in data])
    x_sigma = np.array([np.sqrt(x.marginal.cov[0, :]) for x in data])

    upr = x_mean + 1.96 * x_sigma
    lwr = x_mean - 1.96 * x_sigma
    time = np.arange(len(data))
    plt.fill_between(time, lwr[:, 0], upr[:, 0], alpha=0.5)


def _get_mean_and_std(data):
    mean = data.mean(axis=0)
    anomaly = data - mean[None]
    variance = (anomaly**2).mean(axis=0)
    std = np.sqrt(variance)
    return mean, std

    
def plot_1d_data(data, label, c=None, figsize=None, ax=None):
    """ Plots mean and std bars for given data
    :data: Array of size (samples, xlength)
    :label: Label for legend. Type: str
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
    ax.plot(np.arange(0, N), mean, c=c, label=label, zorder=1)
    if data.shape[0] > 1:
        ax.fill_between(np.arange(0, N), mean-std, mean+std, alpha=0.2, color=c, zorder=2)
    return ax