
from Systems import DynamicSystemModel, GaussianNoise
from StateModel import Gaussian
from collections import namedtuple
from functools import partial
import autograd.numpy as np
import numpy


Sensor = namedtuple('Sensor', ['x', 'y'])
Default_Sensor_List = [Sensor(x=-10, y=10),
                       Sensor(x=-10, y=-10)]

def f(x, t=None, u=None, delta_t=0.1):
    """
    We define constant velocity model
    x_t+1 = F_t * x_t + G_t*w_t
    x_t = [x_t, y_t, \dot x_t, \dot y_t]
    """
    dt = delta_t

    f_t = np.array([[1,    dt,       0,       0],
                    [0,    1,        0,       0],
                    [0,    0,        1,      dt],
                    [0,    0,        0,       1]
                  ])

    # x_out = f_t @ x
    x_out = np.einsum('ij, kj->ki', f_t, np.atleast_2d(x))
    return x_out


def h(x, t=None, u=None, sensor_list=Default_Sensor_List):
    """

    """
    all_ys = [x[:, 2] - sensor.y for sensor in sensor_list]
    all_xs = [x[:, 0] - sensor.x for sensor in sensor_list]
    return np.sqrt(np.array(all_xs) ** 2 + np.array(all_ys) **2 ).T
    # theta = np.arctan2(all_ys, all_xs)
    # return theta.T
    # for sensor in sensor_list:
    #     # theta = np.arctan((x[1] - sensor.y) / (x[0] - sensor.x))
    #     theta = np.arctan2((x[1] - sensor.y) , (x[0] - sensor.x))
    #
    #     # if theta < 0:
    #     #     theta += 2*np.pi
    #
    #     measurements.append(theta)
    #
    # return np.array(measurements)


class BearingsOnlyTrackingTurn(DynamicSystemModel):
    """

    """

    def __init__(self, Q_sigma=1e-3, R_sigma=1, sensor_list=Default_Sensor_List, delta_t=1):

        init_dist = Gaussian(mean_vec=np.array([0.0, 0.0, 1.0, 0.0]),
                             cov_mat=np.eye(4) * [0.1, 0.1, 10, 10])
        if sensor_list is None:
            meas_dim = len(Default_Sensor_List)
        else:
            meas_dim = len(sensor_list)

        transition = partial(f, delta_t=delta_t)
        measurement = partial(h, sensor_list=sensor_list)
        dt = delta_t
        F2 = np.array([[(dt**4)/4,       (dt**3)/2],
                       [(dt**3)/2,       (dt**4)/4]
                       ])
        system_noise_cov = np.kron(np.eye(2), F2) * Q_sigma

        # system_noise_cov = np.array([
        #                              [(dt**4)/4,       (dt**3)/2,       0.0,         0.0],
        #                              [dt**3)/2,            dt**2,       0.0,         0.0],
        #                              [0,                       0,  dt**4)/4,   (dt**3)/2],
        #                              [0,                       0, (dt**3)/2,       dt**2]
        #                             ]) * Q_sigma

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

    def simulate(self, N, x_zero=None, t_zero=0.0, seed=None):
        if seed is None:
            np.random.seed(seed=7952)
        return super().simulate(N=N, x_zero=x_zero, t_zero=t_zero)

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
    np.random.seed(seed=7952)
    data = system.simulate(N)
    x_true, x_noisy, y_true, y_noisy = zip(*data)

    x_true = np.asanyarray(x_true)
    x_noisy = np.asanyarray(x_noisy)


    # plt.plot(x_true[:, 0])
    # plt.scatter(list(range(N)), x_noisy[:, 0])
    plt.scatter(x_noisy[:, 0], x_noisy[:, 1])
    plt.plot(x_true[:, 0], x_true[:, 1])
    plt.show()

