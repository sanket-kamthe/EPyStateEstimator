
from Systems import DynamicSystemModel, GaussianNoise
from StateModel import GaussianState
from collections import namedtuple
from functools import partial
import numpy as np



Sensor = namedtuple('Sensor', ['x', 'y'])


def f(x, t=None, u=None, delta_t=0.1):
    """
    We define constant velocity model
    x_t+1 = F_t * x_t + G_t*w_t
    x_t = [x_t, y_t, \dot x_t, \dot y_t]
    """
    f_t = np.array([[1,  0,  delta_t,  0],
                  [0,  1., 0,        delta_t],
                  [0,  0,  1,        0.],
                  [0,  0,  0,         1]
                  ])

    x_out = np.dot(f_t, x)
    return x_out


def h(x, t=None, u=None, sensor_list=None):
    """

    """
    if sensor_list is None:
        sensor_list = [Sensor(x=0, y=0)]
    measurements = []
    for sensor in sensor_list:
        theta = np.arctan((x[1] - sensor.y) / (x[0] - sensor.x))
        measurements.append(theta)

    return np.array(measurements)


class BearingsOnlyTracking(DynamicSystemModel):
    """

    """

    def __init__(self, Q_sigma=0.1, R_sigma=0.05, sensor_list=None, delta_t=0.1):

        init_dist = GaussianState(mean_vec=np.array([0.0, 0.0, 1.0, 0.0]),
                                  cov_matrix=np.eye(4) * [0.1, 0.1, 10, 10])
        if sensor_list is None:
            meas_dim = 1
        else:
            meas_dim = len(sensor_list)

        transition = partial(f, delta_t=delta_t)
        measurement = partial(h, sensor_list=sensor_list)

        system_noise_cov = np.array([[(delta_t**3)/3,        0.0, (delta_t**2)/2, 0.0],
                                     [0.0,        (delta_t**3)/3, 0,   (delta_t**2)/2],
                                     [(delta_t**2)/2,          0, delta_t,          0],
                                     [0,          (delta_t**2)/2, 0,          delta_t]
                                     ]) * Q_sigma

        system_noise = GaussianNoise(dimension=4, cov=system_noise_cov)
        meas_noise = GaussianNoise(dimension=meas_dim,
                                   cov=np.eye(meas_dim) * (R_sigma**2))

        super().__init__(system_dim=4,
                         measurement_dim=meas_dim,
                         transition=transition,
                         measurement=measurement,
                         system_noise=system_noise,
                         measurement_noise=meas_noise,
                         init_distribution=init_dist
                         )

# def numpy_array(x)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import os
    import sys

    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)

    N = 50

    system = BearingsOnlyTracking()
    data = system.simulate(N)
    x_true, x_noisy, y_true, y_noisy = zip(*data)

    x_true = np.asanyarray(x_true)
    x_noisy = np.asanyarray(x_noisy)


    # plt.plot(x_true[:, 0])
    # plt.scatter(list(range(N)), x_noisy[:, 0])
    plt.scatter(list(range(N)), y_noisy)
    plt.show()

