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


# Moment matching function should return integral of the form \int f(x) q (x) dx

import autograd.numpy as np
from MomentMatching.StateModels import GaussianState
from autograd import jacobian
from functools import partial
from MomentMatching.auto_grad import logpdf
from collections import namedtuple
import logging
FORMAT = "[ %(funcName)10s() ] %(message)s"

# from scipy.stats import multivariate_normal
EPS = 1e-4


logging.basicConfig(filename='Expectation_Propagation.log', level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)


class MomentMatching:

    def __init__(self, approximation_method=None, **kwargs):
        self.params = {}  # Dictionary containing parameters for the moment matching approximation

        self.method = approximation_method
        self.params.update(kwargs)
        # print(self.params)
        for key in self.params:
            self.__setattr__(key, self.params[key])

    def _transform(self, func, state, t=None, u=None, *args, **kwargs):
        """
        Returns the gaussian approximation the integral

        \int f(x) q (x) dx, where q(x) is the distribution and f(x) is the non-linear function
        The result is exact when f(x) is a linear function.
        :param nonlinear_func:
        :param distribution: object of type GaussianState for example
        :return: distribution of y =  \int f(x) q (x) dx in the form of a tuple
        mean (y), variance (yyT), cross_covariance(xyT)
        """
        return NotImplementedError

    def __call__(self, func, state, t=None, u=None, *args, **kwargs):
        return self._transform(func=func, state=state, t=t, u=u, *args, **kwargs)


class UnscentedTransform(MomentMatching):
    """
    
    """
    def __init__(self, n=1, alpha=1, beta=0, kappa=1):

        self.w_m, self.W = self._weights(n, alpha, beta, kappa)
        self.param_lambda = alpha * alpha * (n + kappa) - n
        super().__init__(approximation_method='Unscented Transform',
                         n=n,
                         alpha=alpha,
                         beta=beta,
                         kappa=kappa)

    def _sigma_points(self, mean, cov, *args):

        sqrt_n_plus_lambda = np.sqrt(self.n + self.param_lambda)
        jittered_cov = cov + 1e-4*np.eye(self.n)
        L = np.linalg.cholesky(jittered_cov)

        scaledL = sqrt_n_plus_lambda * L
        mean_plus_L = mean + scaledL
        mean_minus_L = mean - scaledL
        list_sigma_points = [mean.tolist()] + mean_plus_L.tolist() + mean_minus_L.tolist()

        return list_sigma_points

    @staticmethod
    def _weights(n, alpha, beta, kappa):

        param_lambda = alpha * alpha * (n + kappa) - n
        n_plus_lambda = n + param_lambda

        w_m = np.zeros([2 * n + 1], dtype=np.float)
        w_c = np.zeros([2 * n + 1], dtype=np.float)

        # weights w_m are for computing mean and weights w_c are
        # used for covarince calculation

        w_m = w_m + 1 / (2 * (n + param_lambda))
        w_c = w_c + w_m
        w_m[0] = param_lambda / (n + param_lambda)
        w_c[0] = w_m[0] + (1 - alpha * alpha + beta)

        w_left = np.eye(2 * n +1) - np.array(w_m)
        W = w_left.T @ np.diag(w_c) @ w_left

        return w_m, W

    def _transform(self, func, state, t=None, u=None, *args, **kwargs):

        frozen_func = partial(func, t=t, u=u, *args, **kwargs)

        sigma_pts = self._sigma_points(state.mean, state.cov)
        Xi = []
        for x in sigma_pts:
            x = np.asanyarray(x)
            Xi.append(frozen_func(x))

        Y = np.asarray(Xi)

        mean = self.w_m @ Y
        cov = Y.T @ self.W @ Y
        cross_cov = np.asarray(sigma_pts).T @ self.W @ Y

        return mean, cov, cross_cov


class MonteCarloTransform(MomentMatching):
    def __init__(self, dimension_of_state=1, number_of_samples=None):
        super().__init__(approximation_method='Monte Carlo Sampling',
                         dimension_of_state=dimension_of_state,
                         number_of_samples=number_of_samples)

    def _transform(self, func, state, t=None, u=None, *args, **kwargs):
        # (self, nonlinear_func, distribution, fargs=None):

        assert isinstance(state, GaussianState)
        frozen_func = partial(func, t=t, u=u, *args, **kwargs)

        samples = state.sample(self.number_of_samples)

        propagated_samples = frozen_func(samples)
        mean = np.mean(propagated_samples, axis=0)
        cov = np.cov(propagated_samples.T)
        cross_cov = np.cov(samples.T, propagated_samples.T)

        return mean, cov, cross_cov


class TaylorTransform(MomentMatching):
    def __init__(self, dimension_of_state=1, eps=EPS):
        super().__init__(approximation_method='Taylor 1st Order',
                         dimension_of_state=dimension_of_state,
                         eps=eps)

    @staticmethod
    def numerical_jacobian(f, x, eps=EPS):
        z = f(x)
        #     jacobian  = np.zeros((m,n), dtype=float)
        jacobian = []
        x_list = x.tolist()
        for i, data in enumerate(x_list):
            x1 = x.tolist()
            x1[i] = np.array(data) + eps
            x1 = np.array(x1)
            jacobian.append((f(x1) - z) / eps)

        return np.array(jacobian).T

    def _transform(self, func, state, t=None, u=None, *args, **kwargs):
        # (self, nonlinear_func, distribution, fargs=None, y_observation=None):
        assert isinstance(state, GaussianState)
        frozen_func = partial(func, t=t, u=u, *args, **kwargs)
        J_t = self.numerical_jacobian(frozen_func, state.mean)
        # J_t = jacobian(frozen_nonlinear_func)(distribution.mean)
        mean = frozen_func(state.mean)
        cov = J_t @ state.cov @ J_t.T
        cross_cov = state.cov @ J_t.T
        return mean, cov, cross_cov

    def _data_likelihood(self, nonlinear_func, distribution, data=None):
        meanz = nonlinear_func(distribution.mean)  # z = f(x)
        linear_C = jacobian(nonlinear_func)(distribution.mean)  # linear factor A
        covz = linear_C @ distribution.cov @ linear_C.T  # predictive covariance sz = var(f(x)) = A * Sigma * A^T
        logZi = logpdf(data, meanz, covz)

        return logZi

    def project(self, nonlinear_func, distribution, data=None):

        logZi = self._data_likelihood(nonlinear_func, distribution, data)
        dlogZidMz = jacobian(self._data_likelihood, argnum=1)(nonlinear_func, distribution, data)
        dlogZidSz = jacobian(self._data_likelihood, argnum=2)(nonlinear_func, distribution, data)

        return logZi, dlogZidMz, dlogZidSz


if __name__ == '__main__':

    import os
    import sys

    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)

    import numpy as np
    from MomentMatching.StateModels import GaussianState
    unscented_transform = UnscentedTransform()

    xx_mean = np.array([0.0, 0.0], dtype=float)
    xx_sigma = np.array([[1, 0], [0, 1]], dtype=float)
    distribution = GaussianState(xx_mean, xx_sigma)
    a = 5
    b = 3

    def f(x, a=a, b=b):
        return a * x + b


    print(unscented_transform.method)

    res = unscented_transform.project(f, distribution)

    print(f"The transformed mean is {res.mean} and the expected mean is {a * xx_mean + b}")

    print(f"The transformed cov is {res.cov} and the expected cov is {a * xx_sigma * a}")

    mct = MonteCarloTransform(1, number_of_samples=1000)

    res2 = mct.predict(f, distribution)
    print(f"The MCT transformed mean is {res2} and the expected mean is {a * xx_mean + b}")

    print(f"The transformed cov is {res2} and the expected cov is {a * xx_sigma * a}")


