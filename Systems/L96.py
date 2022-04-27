# %%
from Systems import DynamicSystemModel, GaussianNoise
from StateModel import Gaussian
from functools import partial
import autograd.numpy as np


def model(X, F):
        dXdt = -np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) - X + F
        return dXdt


def f(x, t=None, u=None, delta_t=0.05, F=8):
    model_ = partial(model, F=F)

    # Take timestep using fourth order Runge-Kutta scheme
    k1 = model_(x)
    k2 = model_(x + delta_t*k1/2)
    k3 = model_(x + delta_t*k2/2)
    k4 = model_(x + delta_t*k3)

    return x + (1/6) * delta_t * (k1 + 2*k2 + 2*k3 + k4)


def h(x, t=None, u=None, meas_dim=20):
    return x


class L96(DynamicSystemModel):
    def __init__(self, Q_sigma=0.1, R_sigma=1, delta_t=0.05, dim=40, F=8, init_cond_path='L96_initial_conditions.npy'):
        x0 = np.load(init_cond_path)
        assert x0.shape[0] == dim, f'Dimension of initial condition does not match the specified dimension dim={dim}'
        init_dist = Gaussian(mean_vec=x0,
                             cov_mat=np.eye(dim))

        meas_dim = 40
        transition = partial(f, delta_t=delta_t, F=F)
        measurement = partial(h, meas_dim=meas_dim)


        system_noise = GaussianNoise(dimension=dim, cov=Q_sigma*np.eye(dim))
        meas_noise = GaussianNoise(dimension=meas_dim, cov=R_sigma*np.eye(meas_dim))

        super().__init__(system_dim=dim,
                         measurement_dim=meas_dim,
                         transition=transition,
                         measurement=measurement,
                         system_noise=system_noise,
                         measurement_noise=meas_noise,
                         init_distribution=init_dist
                         )

# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x0 = np.load('L96_initial_conditions.npy')
    print(x0.shape)

    N = 200
    system = L96()
    data = system.simulate(N)
    x_true, x_noisy, y_true, y_noisy = zip(*data)

    x_true = np.asanyarray(x_noisy)

    plt.imshow(x_true.T)
# %%
