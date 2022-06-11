# %%
from Systems import DynamicSystemModel, GaussianNoise
from StateModel import Gaussian
from functools import partial
import autograd.numpy as np


def model(X, F):
    X = np.atleast_2d(X)
    dXdt = -np.roll(X, -1, 1) * (np.roll(X, -2, 1) - np.roll(X, 1, 1)) - X + F
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
    x = np.atleast_2d(x)
    return x**2


def generate_L96_initial_condition(dim=40, F=8):
    x0 = F * np.ones(dim)  # Initial state (equilibrium)
    x0[0] += 0.01  # Add small perturbation to the first variable

    x = [x0]
    for n in range(999):
        x.append(f(x[-1]))
    x = np.array(x)

    return x[-1]


class L96(DynamicSystemModel):
    def __init__(self, Q_sigma=0.1, R_sigma=1, delta_t=0.05, dim=40, F=8, init_cond_path=None):
        if init_cond_path is None:
            x0 = generate_L96_initial_condition(dim, F)
            np.save(f'../log/L96_initial_condition_F_{F}_dim_{dim}.npy', x0)
        else:
            try:
                x0 = np.load(init_cond_path)
            except FileNotFoundError:
                x0 = generate_L96_initial_condition(dim, F)
                np.save(f'../log/L96_initial_condition_F_{F}_dim_{dim}.npy', x0)
        
        assert x0.shape[0] == dim, f'Dimension of initial condition does not match the specified dimension dim={dim}'
        init_dist = Gaussian(mean_vec=x0,
                             cov_mat=np.eye(dim))

        meas_dim = dim
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
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    from MomentMatching import UnscentedTransform
    from Filters.KalmanFilter import KalmanFilterSmoother
    from filterpy.kalman import unscented_transform
    from Filters.KalmanFilter import KalmanFilterSmoother, IEKF

    F = 8
    dim = 200

    fname = f'../log/L96_initial_condition_F_{F}_dim_{dim}.npy'
    try:
        x0 = np.load(fname)
    except FileNotFoundError:
        x0 = generate_L96_initial_condition(dim, F)
        np.save(f'../log/L96_initial_condition_F_{F}_dim_{dim}.npy', x0)
        x0 = np.load(fname)

    N = 50
    system = L96(dim=dim, init_cond_path=fname)
    seed = 201
    np.random.seed(seed)
    data = system.simulate(N)
    x_true, y_meas = zip(*data)

    x_true = np.asanyarray(x_true)

    plt.imshow(x_true[:,0,:].T)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Components', fontsize=12)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('../figs/L96_ground_truth.pdf', bbox_inches='tight')


    # %%

    # Implement UKF (filterpy)
    points_1 = MerweScaledSigmaPoints(dim, alpha=2, beta=1., kappa=3)

    f_ = lambda x, dt: f(x, delta_t=dt)
    def h_(x):
        return x**2

    kf1 = UnscentedKalmanFilter(dim_x=dim, dim_z=dim, dt=0.05, fx=f_, hx=h_, points=points_1)
    kf1.x = x0
    P0 = np.eye(dim)
    kf1.P = P0
    kf1.Q = 0.1*np.eye(dim)
    kf1.R = np.eye(dim)

    zs = [y[0] for y in y_meas]
    x_predict, P_predict = kf1.batch_filter(zs)

    plt.imshow(x_predict.T)


    # %%
    # Implement UKF (ours)
    points_2 = UnscentedTransform(dim, alpha=2, beta=1, kappa=3)
    kf2 = KalmanFilterSmoother(points_2, system)
    filter_result = kf2.kalman_filter(y_meas)
    smoother_result = kf2.kalman_smoother(filter_result)

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

    plt.imshow(np.abs(mean_ks-x_true[:,0,:]).T, vmin=0, vmax=18)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.xlabel('Time', fontsize=12)
    plt.tight_layout()
    plt.savefig('../figs/L96_UKF_difference.pdf', bbox_inches='tight')


# %%
