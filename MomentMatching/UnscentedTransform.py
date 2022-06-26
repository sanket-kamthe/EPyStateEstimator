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

# %%
import numpy as np
import scipy as sp
from scipy.linalg import LinAlgWarning
from MomentMatching.MomentMatch import MappingTransform
from Utils.linalg import  jittered_chol
from Utils.linalg import symmetrize
import warnings

warnings.filterwarnings(action='error', category=LinAlgWarning)


class UnscentedTransform(MappingTransform):
    """

    """

    def __init__(self, dim=1, alpha=1, beta=2, kappa=1):
        self.w_m, self.W = self._weights(dim, alpha, beta, kappa)
        self.param_lambda = alpha * alpha * (dim + kappa) - dim
        super().__init__(approximation_method='Unscented Transform',
                         n=dim,
                         alpha=alpha,
                         beta=beta,
                         kappa= kappa)

    def _sigma_points(self, mean, cov, *args):
        sqrt_n_plus_lambda = np.sqrt(self.n + self.param_lambda)

        cov = symmetrize(cov)
        try:    
            L = sp.linalg.cholesky(cov,
                                   lower=False, # needs to be upper triangular because we work with row vectors (arrays)
                                   overwrite_a=False,
                                   check_finite=True)
        except:
            L = jittered_chol(cov, lower=False)

        scaledL = sqrt_n_plus_lambda * L
        mean_plus_L = mean + scaledL 
        mean_minus_L = mean - scaledL
        
        return np.vstack((mean, mean_plus_L, mean_minus_L)) # return row vectors

    @staticmethod
    def _weights(n, alpha, beta, kappa):
        param_lambda = alpha * alpha * (n + kappa) - n
        n_plus_lambda = n + param_lambda

        w_m = np.zeros([2 * n + 1], dtype=np.float)
        w_c = np.zeros([2 * n + 1], dtype=np.float)

        # weights w_m are for computing mean and weights w_c are
        # used for covarince calculation

        w_m = w_m + 1 / (2 * (n_plus_lambda))
        w_c = w_c + w_m
        w_m[0] = param_lambda / (n_plus_lambda)
        w_c[0] = w_m[0] + (1 - alpha * alpha + beta)

        w_left = np.eye(2 * n + 1) - np.array(w_m)
        W = w_left.T @ np.diag(w_c) @ w_left

        W = symmetrize(W)

        return w_m, W

    def _transform(self, func, state):

        sigma_pts = self._sigma_points(state.mean, state.cov)
        Y = func(sigma_pts)
        mean = self.w_m @ Y
        cov = symmetrize(Y.T @ self.W @ Y)
        cross_cov = np.asarray(sigma_pts).T @ self.W @ Y

        return mean, cov, cross_cov

# %%
if __name__ == "__main__":
    from Systems import BearingsOnlyTrackingTurn
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    from Filters.KalmanFilter import KalmanFilterSmoother
    from StateModel import Gaussian

    system = BearingsOnlyTrackingTurn()
    sys_dim = 5

    points_baseline = MerweScaledSigmaPoints(sys_dim, alpha=1, beta=0., kappa=-2)
    points = UnscentedTransform(sys_dim, alpha=1, beta=0., kappa=-2)

    x0 = np.array([1000, 300, 1000, 0, -np.deg2rad(3.0)])
    P0 = np.eye(sys_dim) * [100, 10, 100, 10, 1e-4]
    init_state = Gaussian(x0, P0)

    sigma_base = points_baseline.sigma_points(x0, P0)
    sigma = points._sigma_points(x0, P0)

    # Implement Unscented Kalman filter
    SEED = 101
    timesteps = 50
    np.random.seed(seed=SEED)
    data = system.simulate(timesteps)
    x_true, x_noisy, y_true, y_noisy = zip(*data)
    f = KalmanFilterSmoother(points, system)
    filter_result = f.kalman_filter(y_noisy)
    smoother_result = f.kalman_smoother(filter_result)
    mean_kf, std_kf = [], []
    mean_ks, std_ks = [], []
    for state in smoother_result:
        mean_ks.append(state.mean)
        std_ks.append(np.sqrt(state.cov))
    mean_ks = np.array(mean_ks).squeeze()
    std_ks = np.array(std_ks).squeeze()

    se_list = []
    x_noisy = np.asanyarray(x_noisy)
    for i, x in enumerate(mean_ks):
        se = np.square(np.linalg.norm(x_noisy[i,0,:] - x))
        se_list.append(se)
    mse = np.array(se_list).mean()
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")


    

# %%
