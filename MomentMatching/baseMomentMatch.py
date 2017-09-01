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

# import numpy as np
# from .StateModels import GaussianState

class MomentMatching:
    params = {}  # Dictionary containing parameters for the moment matching approximation

    def __init__(self, approximation_method=None, **kwargs):

        self.method = approximation_method
        self.params.update(kwargs)
        print(self.params)
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

    def predict(self, nonlinear_func, distribution):
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

    def __init__(self, n = 2, alpha= 0.5, beta = 2, kappa = 10 ):
        #
        # default_params = {
        #     'n' : 2,
        #     'alpha':0.5,
        #     'beta': 2,
        #     'kappa': 10
        # }
        super().__init__(approximation_method='Unscented Transform',
                         n=2,
                         alpha=0.5,
                         beta=2,
                         kappa=10)

    def _get_sigma_points(self, x_mean, x_cov, n):

        # n = self.n  # TODO: check whether we actually need this or we use distribution.dim
        alpha = self.alpha
        kappa = self.kappa
        beta = self.beta

        sigma_points = np.zeros([n, 2 * n + 1], dtype=np.float)

        # Use Cholesky as proxy for square root of the matrix
        L = np.linalg.cholesky(x_cov)

        par_lambda = alpha * alpha * (n + kappa) - n
        sqrt_n_plus_lambda = np.sqrt(n + par_lambda)

        sigma_points[:, 0:1] = x_mean
        sigma_points[:, 1:n + 1] = x_mean + sqrt_n_plus_lambda * L
        sigma_points[:, n + 1:2 * n + 1] = x_mean - sqrt_n_plus_lambda * L

        w_m = np.zeros([1, 2 * n + 1], dtype=np.float)
        w_c = np.zeros([1, 2 * n + 1], dtype=np.float)

        print(par_lambda)

        # weights w_m are for computing mean and weights w_c are
        # used for covarince calculation

        w_m = w_m + 1 / (2 * (n + par_lambda))
        w_c = w_c + w_m
        w_m[0, 0] = par_lambda / (n + par_lambda)
        w_c[0, 0] = w_m[0, 0] + (1 - alpha * alpha + beta)

        return sigma_points, w_m, w_c

    def project(self, nonlinear_func, distribution):
        assert isinstance(distribution, GaussianState)

        sigma_points, w_m, w_c = self._get_sigma_points(distribution.mean, distribution.cov, n=distribution.dim)

        transformed_points = nonlinear_func(sigma_points)
        pred_mean = np.sum(np.multiply(transformed_points, w_m), axis=1)
        pred_mean = pred_mean.reshape([distribution.dim, 1])
        gofx_minus_mean = transformed_points - pred_mean
        p_s = np.matmul(gofx_minus_mean, np.transpose(gofx_minus_mean))
        res = np.einsum('ij,jk->ikj', gofx_minus_mean, gofx_minus_mean.T)
        # pred_sigma = np.sum(np.multiply( np.matmul( gofx_minus_mean, gofx_minus_mean.transpose()) , w_m ))
        return pred_mean, p_s
        # pass


if __name__ == '__main__':

    import os
    import sys

    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)

    import numpy as np
    from MomentMatching.StateModels import GaussianState
    unscented_transform = UnscentedTransform()

    xx_mean = np.array([[0.0], [0]])
    xx_sigma = np.array([[1, 0], [0, 1]])
    distribution = GaussianState(xx_mean, xx_sigma)

    def f(x):
        return 2 * x + 1


    print(unscented_transform.method)

    unscented_transform.project(f, distribution)
