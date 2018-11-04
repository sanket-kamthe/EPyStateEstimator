
from Systems import DynamicSystemModel, GaussianNoise
from StateModel import Gaussian
import numpy as np


def f(x, t, u=0):
    """
    test function
    """
    x_out = x % 10 + 0.1 * t + 0.01 * u
    return x_out


def h(x, t=None, u=None):
    """
    Unified Nonlinear growth model (noise not included)
    Measurement function: y_t = h(x_t)
    y = x^2/20
    """
    return x * 100


class TestDynamics(DynamicSystemModel):
    """

    """

    def __init__(self):
        init_dist = Gaussian(mean_vec=np.array([0.0]), cov_mat=np.eye(1) * 0.0001)
        super().__init__(system_dim=1,
                         measurement_dim=1,
                         transition=f,
                         measurement=h,
                         system_noise=GaussianNoise(dimension=1,
                                                    cov=np.eye(1) * 0),
                         measurement_noise=GaussianNoise(dimension=1,
                                                         cov=np.eye(1) * 0),
                         init_distribution=init_dist
                         )