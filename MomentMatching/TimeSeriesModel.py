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
from collections import namedtuple
from MomentMatching.StateModels import GaussianState

#TODO: Refactor the whole thing !!

class NoiseModel:
    def __init__(self, dimension, params):
        self.dimension = dimension
        self.params = params

    def sample(self):
        return NotImplementedError


class GaussianNoise(NoiseModel):
    def __init__(self, dimension=1, cov=np.eye(1), mean=None):
        Gaussian = namedtuple('Gaussian', ['Q'])
        params = Gaussian(Q=cov)
        self.dimension = dimension
        self.cov = cov
        if mean is None:
            self.mean = np.zeros((dimension,), dtype=float )

        super().__init__(dimension=dimension,
                         params=params)

    def sample(self, size=None):
        return np.random.multivariate_normal(mean=self.mean, cov=self.params.Q, size=size)


class DynamicSystemModel:
    def __init__(self, system_dim,
                 measurement_dim, transition,
                 measurement, system_noise,
                 measurement_noise,
                 init_distribution=None, dt=1):
        self.system_dim = system_dim
        self.measurement_dim = measurement_dim
        self.init_state = init_distribution
        self.dt = dt

        assert isinstance(system_noise, NoiseModel)
        assert isinstance(system_noise, NoiseModel)

        assert (system_noise.dimension == system_dim)
        assert (measurement_noise.dimension == measurement_dim)

        self.transition = transition
        self.measurement = measurement
        self.system_noise = system_noise
        self.measurement_noise = measurement_noise

    def transition_noise(self, x,  t=None, u=None, *args, **kwargs):
        return self.transition(x=x, u=u, t=t, *args, **kwargs) + self.system_noise.sample()

    def transition(self, x, u=None, t=None, *args, **kwargs):
        return self.transition(x, u=u, t=t, *args, **kwargs)

    def measurement_sample(self, x, *args, **kwargs):
        return self.measurement(x, *args, **kwargs) + self.measurement_noise.sample()

    def measurement(self, x, *args, **kwargs):
        return self.measurement(x, *args, **kwargs)

    def _simulate(self, N, x_zero, t_zero=0.0):

        x = x_zero
        t = t_zero

        for _ in range(N):
            x_true = self.transition(x=x, t=t)
            x_noisy = x_true + self.system_noise.sample()
            y_true = self.measurement(x=x_noisy)
            y_noisy = y_true + self.measurement_noise.sample()
            yield x_true, x_noisy, y_true, y_noisy
            x = x_noisy
            t = t + self.dt

    def simulate(self, N, x_zero=None, t_zero=0.0):

        if x_zero is None:
            x_zero = np.random.multivariate_normal(mean=self.init_state.mean,
                                                   cov=self.init_state.cov)

        return list(self._simulate(N=N, x_zero=x_zero, t_zero=t_zero))

def f(x, t, u=None):
    """
    Unified Nonlinear growth model (noise not included)
    Transition function: x_(t+1) = f(x_t, t)
    x_out = x/2 + 25x/(1+x^2) + 8 cos(1.2*t)
    """
    x_out = 0.5 * x + ((25 * x) / (1 + x ** 2)) + (8 * np.cos(1.2 * t))
    return x_out


def h(x, t=None, u=None):
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

            # print("The value of {} is {}".format(attribute, value))


class TimeSeriesModel(SystemModel):
    def __init__(self,
                 dimension_state,
                 dimension_observation,
                 transition_function,
                 measurement_function,
                 transition_noise=None,
                 measurement_noise=None,
                 init_dist=None):

        def make_multivariate_random_sampler_of_dimension(D, sigma=0.1):
            mean = np.zeros((D,), dtype=float)
            cov = sigma * np.eye(D, dtype=float)
            return multivariate_normal(mean=mean, cov=cov)

        if transition_noise is None:
            transition_noise = make_multivariate_random_sampler_of_dimension(dimension_state)

        if measurement_noise is None:
            measurement_noise = make_multivariate_random_sampler_of_dimension(dimension_observation, sigma=10.0)

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


class UniformNonlinearGrowthModel1(TimeSeriesModel):
    """

    """
    def __init__(self):
        init_dist = GaussianState(mean_vec=np.array([0.0]), cov_matrix=np.eye(1)*1.0)
        super().__init__(1, 1, transition_function=f, measurement_function=h, init_dist=init_dist)


class UniformNonlinearGrowthModel(DynamicSystemModel):
    """

    """

    def __init__(self):
        init_dist = GaussianState(mean_vec=np.array([0.0]), cov_matrix=np.eye(1) * 1)
        super().__init__(system_dim=1,
                         measurement_dim=1,
                         transition=f,
                         measurement=h,
                         system_noise=GaussianNoise(dimension=1,
                                                    cov=np.eye(1) * 1),
                         measurement_noise=GaussianNoise(dimension=1,
                                                         cov=np.eye(1) * 1),
                         init_distribution=init_dist
                         )

def dummy_sin(x, t):
    return 2*np.sin(x)

# def test_sin(x, t):
#     return np.s

class SimpleSinTest(TimeSeriesModel):
    def __init__(self):
        init_dist = GaussianState(mean_vec=np.array([0.0]), cov_matrix=np.eye(1)*1.0)
        super().__init__(1, 1, transition_function=dummy_sin, measurement_function=lambda x, t: x, init_dist=init_dist)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import os
    import sys

    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from MomentMatching.StateModels import GaussianState

    # demo = TimeSeriesModel(1, 1, transition_function=f, measurement_function=h)
    # ungm = UniformNonlinearGrowthModel()
    N = 100
    ungm = SimpleSinTest()
    data = ungm.system_simulation(N)
    x_true, x_noisy, y_true, y_noisy = zip(*data)

    plt.plot(x_true)

    plt.scatter(list(range(N)), x_noisy)
    plt.plot(y_noisy)
    plt.show()