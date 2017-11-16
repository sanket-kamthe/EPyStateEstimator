import numpy as np
import itertools
from collections import namedtuple
from abc import abstractmethod, ABCMeta

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
            self.mean = np.zeros((dimension,), dtype=float)

        super().__init__(dimension=dimension,
                         params=params)

    def sample(self, size=None):
        return np.random.multivariate_normal(mean=self.mean, cov=self.params.Q, size=size)


class DynamicSystem(metaclass=ABCMeta):

    # def transition_noise(self, x,  t=None, u=None, *args, **kwargs):
    #     return self.transition(x=x, u=u, t=t, *args, **kwargs) + self.system_noise.sample()

    @abstractmethod
    def transition(self, x, u=None, t=None, *args, **kwargs):
        pass

    @abstractmethod
    def measure(self, x, *args, **kwargs):
        pass

    def system_noise(self):
        pass

    def _simulate(self, N, x_zero, t_zero=0.0):

        x = x_zero
        t = t_zero

        for _ in range(N):
            x_true = self.transition(x=x, t=t)
            x_noisy = x_true + self.system_noise.sample()
            y_true = self.measure(x=x_noisy)
            y_noisy = y_true + self.measurement_noise.sample()
            yield x_true, x_noisy, y_true, y_noisy
            x = x_noisy
            t = t + self.dt

    def simulate(self, N, x_zero=None, t_zero=0.0):

        if x_zero is None:
            x_zero = np.random.multivariate_normal(mean=self.init_state.mean,
                                                   cov=self.init_state.cov)

        return list(self._simulate(N=N, x_zero=x_zero, t_zero=t_zero))


class DynamicSystemModel(DynamicSystem):
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

    def measure(self, x, *args, **kwargs):
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