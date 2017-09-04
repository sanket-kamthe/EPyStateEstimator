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

import numpy as np
from scipy.stats import multivariate_normal
import itertools
from .StateModels import GaussianState

def f(x, t):
    """
    Unified Nonlinear growth model (noise not included)
    Transition function: x_(t+1) = f(x_t, t)
    x_out = x/2 + 25x/(1+x^2) + 8 cos(1.2*t)
    """
    x_out = 0.5 * x + ((25 * x) / (1 + x ** 2)) + (8 * np.cos(1.2 * t))
    return x_out


def h (x, t=None):
    """
    Unified Nonlinear growth model (noise not included)
    Measurement function: y_t = h(x_t)
    y = x^2/20
    """
    return (x ** 2)/20

class SystemModel:
    def __init__(self, **kwargs):
        for attribute, value in kwargs.items():
            self.__setattr__(attribute, value)

            print("The value of {} is {}".format(attribute, value))


class TimeSeriesModel(SystemModel):
    def __init__(self,
                 dimension_state,
                 dimension_observation,
                 transition_function,
                 measurement_function,
                 transition_noise=None,
                 measurement_noise=None,
                 init_dist=None):

        def make_multivariate_random_sampler_of_dimension(D, sigma=1):
            mean = np.zeros((D,), dtype=float)
            cov = sigma * np.eye(D, dtype=float)
            return multivariate_normal(mean=mean, cov=cov)

        if transition_noise is None:
            transition_noise = make_multivariate_random_sampler_of_dimension(dimension_state)

        if measurement_noise is None:
            measurement_noise = make_multivariate_random_sampler_of_dimension(dimension_observation)

        super().__init__(D=dimension_state,
                         E=dimension_observation,
                         transition_function=transition_function,
                         measurement_function=measurement_function,
                         Q=transition_noise,
                         R=measurement_noise,
                         init_dist=init_dist)

    def _system_sim(self, x_zero=None, t=0):

        if x_zero is None:
            x_zero = np.zeros(self.D)

        x = x_zero
        while True:
            x_true = self.transition_function(x, t)
            x_noisy = x_true + self.Q.rvs()
            y_true = self.measurement_function(x_noisy, t)
            y_noisy = y_true + self.R.rvs()
            yield x_true, x_noisy, y_true, y_noisy
            x = x_noisy
            t = t + 1

    def system_simulation(self, N, x_zero=None, t=0):
        return list(itertools.islice(self._system_sim(x_zero=x_zero, t=t), N))


class UniformNonlinearGrowthModel(TimeSeriesModel):
    """

    """
    def __init__(self):
        init_dist = GaussianState(mean_vec=np.array([0.0]), cov_matrix=np.eye(1)*0.1)
        super().__init__(1, 1, transition_function=f, measurement_function=h, init_dist=init_dist)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import os
    import sys

    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from MomentMatching.StateModels import GaussianState

    # demo = TimeSeriesModel(1, 1, transition_function=f, measurement_function=h)
    ungm = UniformNonlinearGrowthModel()
    data = ungm.system_simulation(50)
    x_true, x_noisy, y_true, y_noisy = zip(*data)

    plt.plot(x_true)

    plt.scatter(list(range(50)), x_noisy)
    plt.plot(y_noisy)
    plt.show()