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


def plot_gaussian(data):
    # if in_data
    # data = in_data[:, 0:1]
    x_mean = np.array([x.mean for x in data])
    x_sigma = np.array([np.sqrt(x.cov[0, :]) for x in data])

    upr = x_mean + 2 * x_sigma
    lwr = x_mean - 2 * x_sigma
    time = np.arange(len(data))
    plt.fill_between(time, lwr[:, 0], upr[:, 0], alpha=0.5)


def plot_gaussian_node(data):
    # if in_data
    # data = in_data[:, 0:1]
    x_mean = np.array([x.marginal.mean for x in data])
    x_sigma = np.array([np.sqrt(x.marginal.cov[0, :]) for x in data])

    upr = x_mean + 2 * x_sigma
    lwr = x_mean - 2 * x_sigma
    time = np.arange(len(data))
    plt.fill_between(time, lwr[:, 0], upr[:, 0], alpha=0.5)