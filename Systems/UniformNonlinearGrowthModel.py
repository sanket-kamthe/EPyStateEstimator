
from Systems import DynamicSystemModel, GaussianNoise
from StateModel import GaussianState
import numpy as np


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