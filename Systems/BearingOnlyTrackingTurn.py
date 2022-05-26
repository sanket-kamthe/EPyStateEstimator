# %%
from Systems import DynamicSystemModel, GaussianNoise
from StateModel import Gaussian
from collections import namedtuple
from functools import partial
import autograd.numpy as np
from autograd.numpy import sin, cos, sqrt
import numpy


Sensor = namedtuple('Sensor', ['x', 'y'])
Default_Sensor_List = [Sensor(x=-20_000, y=20_000),
                       Sensor(x=20_000, y=20_000),
                       Sensor(x=-20_000, y=-20_000),
                       Sensor(x=20_000, y=-20_000)]

# Default_Sensor_List = [Sensor(x=0, y=0)]

def f(x, t=None, u=None, delta_t=1):
    """
    We define constant velocity model
    x_t+1 = F_t * x_t + G_t*w_t
    x_t = [x_t, y_t, \dot x_t, \dot y_t]
    """
    dt = delta_t
    xs = np.atleast_2d(x)
    ones = np.ones(xs.shape[0])
    zros = np.zeros(xs.shape[0])
    w = xs[:,-1]
    f_t = np.array([[ones,      sin(w*dt)/w,       zros,      -(1-cos(w*dt))/w,       zros],
                    [zros,      cos(w*dt),         zros,      -sin(w*dt),             zros],
                    [zros,      (1-cos(w*dt))/w,   ones,       sin(w*dt)/w,           zros],
                    [zros,      sin(w*dt),         zros,       cos(w*dt),             zros],
                    [zros,      zros,              zros,       zros,                  ones]
                    ])
    f_t = np.moveaxis(f_t, -1, 0) # Shape (batch, 5, 5)
    x_out = np.einsum('ijk, ik -> ij', f_t, xs)
    # for i, x in enumerate(xs):
    #     w = x[-1]
    #     f_t = np.array([[1,      sin(w*dt)/w,       0,      -(1-cos(w*dt))/w,       0],
    #                     [0,      cos(w*dt),         0,      -sin(w*dt),             0],
    #                     [0,      (1-cos(w*dt))/w,   1,       sin(w*dt)/w,           0],
    #                     [0,      sin(w*dt),         0,       cos(w*dt),             0],
    #                     [0,      0,                 0,       0,                     1]
    #                 ])

    #     x_out[i] = f_t @ x
        # x_out[i] = np.einsum('ij, kj->ki', f_t, x)
    return x_out


def h(x, t=None, u=None, sensor_list=Default_Sensor_List):
    """

    """
    x = np.atleast_2d(x)
    all_ys = np.array([x[:, 2] - sensor.y for sensor in sensor_list])
    all_xs = np.array([x[:, 0] - sensor.x for sensor in sensor_list])
    theta = np.arctan2(all_ys, all_xs)
    r = np.sqrt(all_xs**2 + all_ys**2)
    # return np.vstack([r, theta]).T
    return theta.T


class BearingsOnlyTrackingTurn(DynamicSystemModel):
    """

    """

    def __init__(self,
                Q_sigma=[0.1, 1.75e-4],
                R_sigma = [sqrt(10)*1e-3], # R_sigma=[10, sqrt(10)*1e-3],
                sensor_list=Default_Sensor_List, delta_t=1):

        init_dist = Gaussian(mean_vec=np.array([1000, 300, 1000, 0, -np.deg2rad(3.0)]),
                             cov_mat=np.eye(5) * [100, 10, 100, 10, 1e-4])
        if sensor_list is None:
            meas_dim = len(Default_Sensor_List)
            # meas_dim = 2*len(Default_Sensor_List)
        else:
            meas_dim = len(sensor_list)
            # meas_dim = 2*len(sensor_list)

        transition = partial(f, delta_t=delta_t)
        measurement = partial(h, sensor_list=sensor_list)
        dt = delta_t
        M = np.array([[(dt**3)/3,       (dt**2)/2],
                     [(dt**2)/2,         dt]])

        system_noise_cov = np.eye(5)
        system_noise_cov[:4, :4] = np.kron(np.eye(2), M) * Q_sigma[0]
        system_noise_cov[4, 4] = dt*Q_sigma[1]


        system_noise = GaussianNoise(dimension=5, cov=system_noise_cov)
        meas_noise = GaussianNoise(dimension=meas_dim,
                                   cov=np.eye(meas_dim) * np.array(R_sigma)**2)

        super().__init__(system_dim=5,
                         measurement_dim=meas_dim,
                         transition=transition,
                         measurement=measurement,
                         system_noise=system_noise,
                         measurement_noise=meas_noise,
                         init_distribution=init_dist
                         )

    # def simulate(self, N, x_zero=None, t_zero=0.0, seed=None):
    #     if seed is None:
    #         np.random.seed(seed=7952)
    #     return super().simulate(N=N, x_zero=x_zero, t_zero=t_zero)

# def numpy_array(x)

#%%
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import os
    import sys

    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)

    N = 100

    system = BearingsOnlyTrackingTurn()
    seed = 101
    np.random.seed(seed)
    data = system.simulate(N)
    x_true, y_meas = zip(*data)

    x_true = np.asanyarray(x_true)

    Sensor_x = [sensor.x for sensor in Default_Sensor_List]
    Sensor_y = [sensor.y for sensor in Default_Sensor_List]
    plt.scatter(x_true[:, 0, 0], x_true[:, 0, 2], c='C0', label='ground truth')

    
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    from MomentMatching import UnscentedTransform
    from Filters.KalmanFilter import KalmanFilterSmoother
    from filterpy.kalman import unscented_transform

    # Implement UKF (filterpy)
    points = MerweScaledSigmaPoints(5, alpha=1, beta=0., kappa=-2)

    def f_(x, delta_t=1):
        dt = delta_t
        w = x[-1]
        f_t = np.array([[1,      sin(w*dt)/w,       0,      -(1-cos(w*dt))/w,       0],
                        [0,      cos(w*dt),         0,      -sin(w*dt),             0],
                        [0,      (1-cos(w*dt))/w,   1,       sin(w*dt)/w,           0],
                        [0,      sin(w*dt),         0,       cos(w*dt),             0],
                        [0,      0,                 0,       0,                     1]
                    ])

        return np.dot(f_t, x)

    def h_(x):
        all_ys = np.array([x[2] - sensor.y for sensor in Default_Sensor_List])
        all_xs = np.array([x[0] - sensor.x for sensor in Default_Sensor_List])
        theta = np.arctan2(all_ys, all_xs)
        r = np.sqrt(all_xs**2 + all_ys**2)
        return theta
        #return np.vstack([r, theta]).T[0]

    meas_dim = 4
    kf = UnscentedKalmanFilter(dim_x=5, dim_z=meas_dim, dt=1, fx=f_, hx=h_, points=points)
    x0 = np.array([1000, 300, 1000, 0, -np.deg2rad(3.0)])
    P0 = np.eye(5) * [100, 10, 100, 10, 1e-4]
    kf.x = x0
    kf.P = P0
    R_sigma= [sqrt(10)*1e-3] #[10, sqrt(10)*1e-3]
    kf.R = np.eye(meas_dim) * np.array(R_sigma)**2
    dt = 1
    Q_sigma=[0.1, 1.75e-4]
    M = np.array([[(dt**3)/3,       (dt**2)/2],
                    [(dt**2)/2,         dt]])

    system_noise_cov = np.eye(5)
    system_noise_cov[:4, :4] = np.kron(np.eye(2), M) * Q_sigma[0]
    system_noise_cov[4, 4] = dt*Q_sigma[1]
    kf.Q = system_noise_cov


    zs = [y[0] for y in y_meas]
    x_predict, P_predict = kf.batch_filter(zs)

    plt.scatter(x_predict[:, 0], x_predict[:, 2], c='C1', label='filterpy UKF')

    # Implement our version of UKF
    points = UnscentedTransform(5, alpha=1, beta=0., kappa=-2)
    f = KalmanFilterSmoother(points, system)
    filter_result = f.kalman_filter(y_meas)
    smoother_result = f.kalman_smoother(filter_result)

    mean_kf, std_kf = [], []
    mean_ks, std_ks = [], []
    for state_kf, state_ks in zip(filter_result, smoother_result):
        mean_kf.append(state_kf.mean)
        std_kf.append(np.sqrt(state_kf.cov))
        mean_ks.append(state_ks.mean)
        std_ks.append(np.sqrt(state_ks.cov))
    mean_kf = np.array(mean_kf).squeeze()
    std_kf = np.array(std_kf).squeeze()
    mean_ks = np.array(mean_ks).squeeze()
    std_ks = np.array(std_ks).squeeze()

    plt.scatter(mean_kf[:, 0], mean_kf[:, 2], c='C2', label='Our UKF')
    plt.legend()
    plt.show()

    # Evaulate

    # se_list = []
    # for i, x in enumerate(x_predict):
    #     se = np.square(np.linalg.norm(x_noisy[i,0,:] - x))
    #     se_list.append(se)
    # mse = np.array(se_list).mean()
    # rmse = np.sqrt(mse)
    # print(f"RMSE: {rmse}")


# %%
