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
import scipy as sp

RTOL, ATOL = 1e-3, 1e-5
INF = 1000
JIT = 1e-12

def natural_to_moment(precision, shift):
    dim = precision.shape[0]
    cov = sp.linalg.solve(precision,
                          np.eye(N=dim),
                          assume_a='pos')
    # cov = np.linalg.pinv(precision)
    mean = np.dot(cov, shift)
    return mean, cov


def moment_to_natural(mean, cov):
    dim = cov.shape[0]
    precision = sp.linalg.solve(cov, np.eye(N=dim),
                                assume_a='pos')
    # precision = np.linalg.pinv(cov)
    shift = precision @ mean
    return precision, shift


class GaussianState(State):
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
        #         self.dim = dim

        if mean_vec is not None:
            self.mean = mean_vec
            self.cov = cov_mat


        if shift_vec is not None:
            self.precision = precision_mat
            self.shift = shift_vec

        # super(multivariate_normal, self).__init__(mean=mean_vec,
        #                                           cov_mat=cov)

            # TODO: Add type checks and asserts for mean and covariance

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.dot(self.cov, self.shift)
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean
        # we have changed mean so shift and precision are no longer valid, so we set them to None for lazy computation
        #  if needed
        # self._shift = None
        # self._precision = None

    @property
    def dim(self):
        if self._dim is None:
            if self._mean is None:
                self._dim = np.shape(self.shift)[0]
            else:
                self._dim = np.shape(self.mean)[0]
        return self._dim

    @property
    def cov(self):
        if self._cov is None:
            try:
                self.precision += np.eye(self.dim) * JIT
                self._cov = np.linalg.solve(self.precision, np.eye(self.dim))
            except LinAlgError:
                print('bad covariance {}'.format(self.cov))
        return self._cov

    @cov.setter
    def cov(self, cov):
        self._cov = cov
        # we have changed the covariance so shift and precision are no longer valid,
        #  so we set them to None for lazy computation, if needed.
        self._shift = None
        self._precision = None

    @property
    def precision(self):
        if self._precision is None:
            try:
                self.cov += np.eye(self.dim) * JIT
                self._precision = np.linalg.solve(self.cov, np.eye(self.dim))
            except LinAlgError:
                print('bad covariance {}'.format(self.cov))

        return self._precision

    @precision.setter
    def precision(self, value):
        self._precision = value

    @property
    def shift(self):
        if self._shift is None:
            self._shift = np.dot(self.precision, self.mean)

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
        assert isinstance(other, GaussianState)
        precision = self.precision + other.precision
        shift = self.shift + other.shift
        # mean, cov = natural_to_moment(precision, shift)
        # cov = (cov.T + cov) / 2
        return GaussianState(precision_mat=precision,
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
        # mean, cov = natural_to_moment(precision, shift)
        # cov = (cov.T + cov) / 2
        return GaussianState(precision_mat=precision,
                             shift_vec=shift)

    def __pow__(self, power, modulo=None):
        if (self.cov[0, 0]) > INF:
            return GaussianState(self.mean, self.cov)

        # precision = power * self.precision
        # shift = power * self.shift
        # mean, cov = natural_to_moment(precision, shift)
        cov = self.cov / power
        cov = (cov.T + cov) / 2
        return GaussianState(self.mean, cov)

    def __eq__(self, other):
        # Make sure that 'other' is also a GaussianState class
        # TODO: Replace assert with a custom Error
        assert isinstance(other, GaussianState)
        mean_equal = np.allclose(self.mean, other.mean, rtol=RTOL, atol=RTOL)
        cov_equal = np.allclose(self.cov, other.cov, rtol=RTOL, atol=RTOL)

        return mean_equal and cov_equal

    def nll(self, x):
        """
        Find the negative log likelihood of x
        :param x:
        :return: -ve of logpdf (x, mean=self.mean, cov=self.cov)
        """
        from scipy.stats import multivariate_normal
        if np.isinf(self.cov[0, 0]):
            return np.nan

        diff = x - self.mean
        logdet = np.log(2 * np.pi) + np.log(np.linalg.det(self.cov))
        NLL = 0.5 * (logdet + diff.T @ self.precision @ diff)
        return NLL
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

        return str.format('GaussianState \n mean=\n {}, \n cov=\n{})', self.mean, self.cov)

    def __str__(self):
        return str.format('mean={},cov={}', self.mean, self.cov)

    def copy(self):
        return GaussianState(self.mean, self.cov)

    @classmethod
    def as_factor(cls, dim):
        mean = np.zeros((dim,), dtype=float)
        diag_cov = (np.inf) * np.ones((dim,), dtype=float)
        cov = np.diag(diag_cov)
        return cls(mean_vec=mean, cov_mat=cov)

    @classmethod
    def from_natural(cls, precision, shift):
        c = cls()

    @classmethod
    def as_marginal(cls, dim):
        mean = np.zeros((dim,), dtype=float)
        # diag_cov = (np.inf) * np.ones((dim,), dtype=float)
        # cov = np.diag(diag_cov)
        cov = 99999 * np.eye(dim)
        return cls(mean_vec=mean, cov_mat=cov)


class GaussianFactor(GaussianState):
    def __init__(self, dim):
        mean = np.zeros((dim,), dtype=float)
        diag_cov = np.inf * np.ones((dim,), dtype=float)
        cov = np.diag(diag_cov)
        super().__init__(mean_vec=mean,
                         cov_mat=cov)
