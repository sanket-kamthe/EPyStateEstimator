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
from .MomentMatch import MappingTransform
from Utils.linalg import  jittered_chol
from functools import partial


class UnscentedTransform(MappingTransform):
    """

    """

    def __init__(self, dim=1, alpha=1, beta=0, kappa=1):
        self.w_m, self.W = self._weights(dim, alpha, beta, kappa)
        self.param_lambda = alpha * alpha * (dim + kappa) - dim
        super().__init__(approximation_method='Unscented Transform',
                         n=dim,
                         alpha=alpha,
                         beta=beta,
                         kappa=kappa)

    def _sigma_points(self, mean, cov, *args):
        sqrt_n_plus_lambda = np.sqrt(self.n + self.param_lambda)

        # jittered_cov = cov + 1e-4 * np.eye(self.n)
        L = jittered_chol(cov)

        scaledL = sqrt_n_plus_lambda * L
        mean_plus_L = mean + scaledL
        mean_minus_L = mean - scaledL
        # list_sigma_points = [mean.tolist()] + mean_plus_L.tolist() + mean_minus_L.tolist()

        # return list_sigma_points
        return np.vstack((mean, mean_plus_L, mean_minus_L))

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

        return w_m, W

    def _transform(self, func, state):
        # frozen_func = partial(func, t=t, u=u, *args, **kwargs)

        sigma_pts = self._sigma_points(state.mean, state.cov)
        # Xi = []
        # for x in sigma_pts:
        #     x = np.asanyarray(x)
        #     Xi.append(func(x))

        # Y = np.asarray(Xi)
        Y = func(np.asanyarray(sigma_pts))
        mean = self.w_m @ Y
        cov = Y.T @ self.W @ Y
        cross_cov = np.asarray(sigma_pts).T @ self.W @ Y

        return mean, cov, cross_cov
