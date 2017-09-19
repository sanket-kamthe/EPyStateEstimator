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

import numpy as np
from MomentMatching.StateModels import GaussianState
from autograd import jacobian
from functools import partial
from .auto_grad import logpdf

# from scipy.stats import multivariate_normal
EPS = 1e-4


class MomentMatching:


    def __init__(self, approximation_method=None, **kwargs):
        self.params = {}  # Dictionary containing parameters for the moment matching approximation

        self.method = approximation_method
        self.params.update(kwargs)
        # print(self.params)
        for key in self.params:
            self.__setattr__(key, self.params[key])

    def project(self, nonlinear_func, distribution):
        """
        Returns the approximate projection of distribution through a nonlinear function.

        ..math \\int f(x) q (x) dx, where q(x) is the distribution and f(x) is the non-linear function
        The result is exact when f(x) is a linear function.

        Parameters
        -----------------------------------------------------
        :param nonlinear_func: a nonlinear function in x
        :param distribution: distribution of the variable x
        :type nonlinear_func: any python function
        :type distribution: GaussianState object

        Returns
        ---------------------------------------------

        :return: approximate distribution in the same family as input distribution
        :rtype: object of the type 'distribution'
        """
        return NotImplementedError

    def predict(self, nonlinear_func, distribution, *args, y_observation = None):
        """
        Mainly to be used with Kalman Filtering
        Returns the gaussian approximation the integral
        \int f(x) q (x) dx, where q(x) is the distribution and f(x) is the non-linear function
        The result is exact when f(x) is a linear function.
        :param nonlinear_func:
        :param distribution: object of type GaussianState for example
        :return: distribution of y =  \int f(x) q (x) dx in the form of a tuple
        mean (y), variance (yyT), cross_covariance(xyT)
        """
        return NotImplementedError


class UnscentedTransform(MomentMatching):
    """
    
    """
    def __init__(self, n=2, alpha=0.5, beta=2, kappa=10):
        #
        # default_params = {
        #     'n' : 2,
        #     'alpha':0.5,
        #     'beta': 2,
        #     'kappa': 10
        # }
        super().__init__(approximation_method='Unscented Transform',
                         n=n,
                         alpha=alpha,
                         beta=beta,
                         kappa=kappa)

    def _get_sigma_points(self, x_mean, x_cov, n):
        """

        Args:
            x_mean:
            x_cov:
            n:

        Returns:

        """


        # n = self.n  # TODO: check whether we actually need this or we use distribution.dim
        alpha = self.alpha
        kappa = self.kappa
        beta = self.beta

        sigma_points = np.zeros([n, 2 * n + 1], dtype=np.float)

        # Use Cholesky as proxy for square root of the matrix
        L = np.linalg.cholesky(x_cov)

        par_lambda = alpha * alpha * (n + kappa) - n
        sqrt_n_plus_lambda = np.sqrt(n + par_lambda)

        sigma_points[:, 0:1] = x_mean[:, np.newaxis]
        sigma_points[:, 1:n + 1] = x_mean + sqrt_n_plus_lambda * L
        sigma_points[:, n + 1:2 * n + 1] = x_mean - sqrt_n_plus_lambda * L

        w_m = np.zeros([1, 2 * n + 1], dtype=np.float)
        w_c = np.zeros([1, 2 * n + 1], dtype=np.float)

        # print(par_lambda)

        # weights w_m are for computing mean and weights w_c are
        # used for covarince calculation

        w_m = w_m + 1 / (2 * (n + par_lambda))
        w_c = w_c + w_m
        w_m[0, 0] = par_lambda / (n + par_lambda)
        w_c[0, 0] = w_m[0, 0] + (1 - alpha * alpha + beta)

        return sigma_points, w_m, w_c

    def project(self, nonlinear_func, distribution, *args):
        """

        Args:
            nonlinear_func:
            distribution:
            *args:

        Returns:

        """
        assert isinstance(distribution, GaussianState)

        sigma_points, w_m, w_c = self._get_sigma_points(distribution.mean, distribution.cov, n=distribution.dim)

        transformed_points = nonlinear_func(sigma_points, *args)
        pred_mean = np.sum(np.multiply(transformed_points, w_m), axis=1)
        pred_mean = pred_mean.reshape([distribution.dim, 1])
        gofx_minus_mean = transformed_points - pred_mean
        p_s = np.matmul(gofx_minus_mean, np.transpose(gofx_minus_mean))
        res = np.einsum('ij,jk->ikj', gofx_minus_mean, gofx_minus_mean.T)
        res_mul = res * w_c[np.newaxis, :, :]
        pred_cov = np.einsum('ijk->ij', res_mul)

        return GaussianState(pred_mean, pred_cov)

    def predict(self, nonlinear_func, distribution, *args):
        assert isinstance(distribution, GaussianState)

        # if args is not tuple:
        #     args = (args, )
        frozen_nonlinear_func = partial(nonlinear_func, *args)

        sigma_points, w_m, w_c = self._get_sigma_points(distribution.mean, distribution.cov, n=distribution.dim)

        transformed_points = frozen_nonlinear_func(sigma_points)
        pred_mean = np.sum(np.multiply(transformed_points, w_m), axis=1)
        # pred_mean = pred_mean.reshape([distribution.dim, 1])
        gofx_minus_mean = transformed_points - pred_mean[:, np.newaxis]
        scaled_gofx_minus_mean = w_c * gofx_minus_mean

        res = np.einsum('ij,jk->ikj', gofx_minus_mean, scaled_gofx_minus_mean.T)
        # res_mul = res * w_c[np.newaxis, :, :]
        pred_cov = np.einsum('ijk->ij', res)

        gofy_minus_mean = sigma_points - distribution.mean[:, np.newaxis]
        res_cross = np.einsum('ij,jk->ikj', gofy_minus_mean, scaled_gofx_minus_mean.T)
        # res_mul = res * w_c[np.newaxis, :, :]
        pred_cross_cov = np.einsum('ijk->ij', res_cross)

        return pred_mean, pred_cov, pred_cross_cov
        # pass

class MonteCarloTransform(MomentMatching):
    def __init__(self, dimension_of_state=1, number_of_samples=None):
        super().__init__(approximation_method='Monte Carlo Sampling',
                         dimension_of_state=dimension_of_state,
                         number_of_samples=number_of_samples)

    def predict(self, nonlinear_func, distribution, *args, y_observation=None):

        assert isinstance(distribution, GaussianState)
        frozen_nonlinear_func = partial(nonlinear_func, *args)

        samples = distribution.sample(self.number_of_samples)

        propagated_samples = frozen_nonlinear_func(samples)
        pred_mean = np.mean(propagated_samples, axis=0)
        pred_cov = np.cov(propagated_samples.T)
        pred_cross_cov = np.cov(samples.T, propagated_samples.T )
        return pred_mean, pred_cov, pred_cross_cov


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
    # def _jacobian(self, ):

    def predict(self, nonlinear_func, distribution, *args, y_observation = None):
        assert isinstance(distribution, GaussianState)
        frozen_nonlinear_func = partial(nonlinear_func, *args)
        J_t = self.numerical_jacobian(frozen_nonlinear_func, distribution.mean)
        pred_mean = frozen_nonlinear_func(distribution.mean)
        pred_cov = J_t @ distribution.cov @ J_t.T
        pred_cross_cov = distribution.cov @ J_t.T
        return pred_mean, pred_cov, pred_cross_cov

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


