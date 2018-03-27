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
from numpy.linalg import LinAlgError
from StateModel import State
from Utils import cholesky
from scipy.stats import multivariate_normal
from Utils.linalg import jittered_solve
import scipy as sp

RTOL, ATOL = 1e-3, 1e-5
INF = 1000
JIT = 1e-12
LARGE_NUM = 99999


def natural_to_moment(precision, shift):
    dim = precision.shape[0]

    if np.trace(precision) < 1e-6:
        # almost zero precision
        dim = dim
        mean = np.zeros((dim,), dtype=float)
        cov = LARGE_NUM * np.eye(dim)
        return mean, cov

    # cov = np.linalg.pinv(precision)

    try:
        cov = jittered_solve(precision,
                             np.eye(N=dim),
                             assume_a='pos')

    except LinAlgError:
        print('possible bad precision matrix {}'.format(precision))
        raise LinAlgError
    mean = np.dot(cov, shift)
    return mean, cov


def moment_to_natural(mean, cov):
    dim = cov.shape[0]
    try:
        precision = jittered_solve(cov, np.eye(N=dim), assume_a='pos')
    except LinAlgError:
        print('possible bad covariance matrix {}'.format(cov))
        raise LinAlgError

    # precision = np.linalg.pinv(cov)
    shift = precision @ mean
    return precision, shift


class Gaussian(State):
    """

    Define a Gaussian state model for EP operations in
    the standard parameters.

    Parameters

    --------

    mean_vec: numpy array_like

    """

    def __init__(self,
                 mean_vec=None,
                 cov_mat=None,
                 precision_mat=None,
                 shift_vec=None):
        """

        :param mean_vec:
        :param cov_mat:
        :param precision_mat:
        :param shift_vec:
        """
        #         dim = mean_vec.shape[0]
        self._mean = None
        self._cov = None
        self._dim = None
        self._precision = None
        self._shift = None
        self._mode = None  # 'natural', 'moment'
        self._chol = None
        #         self.dim = dim

        if mean_vec is not None:
            self.mean = mean_vec
            self.cov = cov_mat
            self._mode = 'moment'

        if shift_vec is not None:
            self.precision = precision_mat
            self.shift = shift_vec
            self._mode = 'natural'

        # super(multivariate_normal, self).__init__(mean=mean_vec,
        #                                           cov_mat=cov)

            # TODO: Add type checks and asserts for mean and covariance

    @property
    def mean(self):
        if self._mode == 'natural' and self._mean is None:
            self.mean, self.cov = natural_to_moment(self.precision, self.shift)

        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean

    @property
    def dim(self):
        if self._dim is None:
            if self._mode == 'natural':
                self._dim = np.shape(self.shift)[0]
            else:
                self._dim = np.shape(self.mean)[0]
        return self._dim

    @property
    def cov(self):
        if self._mode == 'natural' and self._cov is None:
            self.mean, self.cov = natural_to_moment(self.precision, self.shift)

        return self._cov

    @cov.setter
    def cov(self, cov):
        self._cov = cov

    @property
    def precision(self):
        if self._mode == 'moment' and self._precision is None:
            self.precision, self.shift = moment_to_natural(mean=self.mean, cov=self.cov)

        return self._precision

    @precision.setter
    def precision(self, value):
        self._precision = value

    @property
    def shift(self):
        if self._mode == 'moment' and self._shift is None:
            self.precision, self.shift = moment_to_natural(mean=self.mean, cov=self.cov)

        return self._shift

    @shift.setter
    def shift(self, value):
        self._shift = value

    @property
    def chol_cov(self):
        if self._chol is None:
            self._chol = cholesky(self.cov, lower=True)

        return self._chol

    def sampler(self):
        multivariate_normal()

    def __mul__(self, other):
        # Make sure that other is also a GaussianState class
        assert isinstance(other, Gaussian)
        precision = self.precision + other.precision
        shift = self.shift + other.shift
        # mean, cov = natural_to_moment(precision, shift)
        # cov = (cov.T + cov) / 2
        return Gaussian(precision_mat=precision,
                        shift_vec=shift)

    def __truediv__(self, other):
        # Make sure that 'other' is also a GaussianState class
        # TODO: Replace assert with a custom Error
        # assert isinstance(other, GaussianState)
        precision = self.precision - other.precision
        # if precision < 0:
        #     warnings.warn('Negative Precision!!!')
        # print(precision)
        # precision + 1e-6

        shift = self.shift - other.shift
        return Gaussian(precision_mat=precision,
                        shift_vec=shift)

    def __pow__(self, power, modulo=None):
        # if (self.cov[0, 0]) > INF:
            # return Gaussian(self.mean, self.cov)

        if self._mode == 'natural':
            precision = power * self.precision
            shift = power * self.shift
            return Gaussian(precision_mat=precision,
                            shift_vec=shift)
        else:
            cov = self.cov / power
            cov = (cov.T + cov) / 2
        return Gaussian(self.mean, cov)

    def __eq__(self, other):
        # Make sure that 'other' is also a GaussianState class
        # TODO: Replace assert with a custom Error
        assert isinstance(other, Gaussian)
        mean_equal = np.allclose(self.mean, other.mean, rtol=RTOL, atol=RTOL)
        cov_equal = np.allclose(self.cov, other.cov, rtol=RTOL, atol=RTOL)

        return mean_equal and cov_equal

    def nll(self, x):
        """
        Find the negative log likelihood of x
        :param x:
        :return: -ve of logpdf (x, mean=self.mean, cov=self.cov)
        """
        # from scipy.stats import multivariate_normal
        # if np.isinf(self.cov[0, 0]):
        #     return np.nan
        #
        # diff = x - self.mean
        # logdet = np.log(2 * np.pi) + np.log(np.linalg.det(self.cov))
        # NLL = 0.5 * (logdet + diff.T @ self.precision @ diff)
        loglikelihood = multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov, allow_singular=True)
        return -loglikelihood
        # return -multivariate_normal(mean=self.mean, cov=self.cov).logpdf(x, cond=1e-6)

    def rmse(self, x):
        """
        Squared Error
        :param x:
        :return:
        """
        return np.square(np.linalg.norm(self.mean - x))

    def sample(self, number_of_samples):

        # from scipy.stats import multivariate_normal

        # return multivariate_normal(mean=self.mean, cov=self.cov).rvs(number_of_samples)

        samples = np.random.multivariate_normal(mean=self.mean,
                                                cov=self.cov,
                                                size=number_of_samples)
        return samples

    def __repr__(self):
        if self._mode == 'natural':
            return str.format('GaussianState \n shift=\n {}, \n precis=\n{})', self.shift, self.precision)
        else:
            return str.format('GaussianState \n mean=\n {}, \n cov=\n{})', self.mean, self.cov)

    def __str__(self):
        return str.format('mean={},cov={}', self.mean, self.cov)

    def copy(self):
        if self._mode == 'natural':
            return Gaussian.from_natural(precision=self.precision, shift=self.shift)
        else:
            return Gaussian(self.mean, self.cov)

    @classmethod
    def as_factor(cls, dim):

        shift = np.zeros((dim,), dtype=float)
        precision = np.zeros((dim, dim), dtype=float)
        return cls(shift_vec=shift, precision_mat=precision)

    @classmethod
    def from_natural(cls, precision, shift):
        return cls(precision_mat=precision, shift_vec=shift)

    @classmethod
    def as_marginal(cls, dim):
        mean = np.zeros((dim,), dtype=float)
        cov = LARGE_NUM * np.eye(dim)
        return cls(mean_vec=mean, cov_mat=cov)


class GaussianFactor(Gaussian):
    def __init__(self, dim):
        mean = np.zeros((dim,), dtype=float)
        diag_cov = np.inf * np.ones((dim,), dtype=float)
        cov = np.diag(diag_cov)
        super().__init__(mean_vec=mean,
                         cov_mat=cov)