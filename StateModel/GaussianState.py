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

RTOL, ATOL = 1e-3, 1e-5
INF = 1000
JIT = 1e-6

def natural_to_moment(precision, shift):
    # dim = precision.shape[0]
    # cov = np.linalg.solve(precision, np.eye(N=dim))
    cov = np.linalg.pinv(precision)
    mean = np.dot(cov, shift)
    return mean, cov


def moment_to_natural(mean, cov):
    # dim = cov.shape[0]
    # precision = np.linalg.solve(cov, np.eye(N=dim))
    precision = np.linalg.pinv(cov)
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
    def __init__(self, mean_vec, cov_matrix):
        """

        :param mean_vec:
        :param cov_matrix:
        """
        dim = mean_vec.shape[0]
        self._mean = None
        self._cov = None
        self.dim = dim
        self.mean = mean_vec
        self.cov = cov_matrix

        # Lazy computation of precision and shift
        self._precision = None
        self._shift = None


        # TODO: Add type checks and asserts for mean and covariance

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean
        # we have changed mean so shift and precision are no longer valid, so we set them to None for lazy computation
        #  if needed
        self._shift = None
        self._precision = None

    @property
    def cov(self):
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
                print(f'bad covariance{self.cov}')

        return self._precision

    @property
    def shift(self):
        if self._shift is None:
            self._shift = np.dot(self.precision, self.mean)

        return self._shift

    def __mul__(self, other):
        # Make sure that other is also a GaussianState class
        assert isinstance(other, GaussianState)
        precision = self.precision + other.precision
        shift = self.shift + other.shift
        mean, cov = natural_to_moment(precision, shift)
        return GaussianState(mean, cov)

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
        mean, cov = natural_to_moment(precision, shift)
        return GaussianState(mean, cov)

    def __pow__(self, power, modulo=None):

        # precision = power * self.precision
        # shift = power * self.shift
        # mean, cov = natural_to_moment(precision, shift)
        return GaussianState(self.mean, self.cov/power)

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
        if np.isinf(self.cov[0,0]):
            return np.nan

        diff = x - self.mean
        logdet = np.log(2 * np.pi) + np.log(np.linalg.det(self.cov))
        NLL= 0.5 * (logdet + diff.T @ self.precision @ diff)
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
        return cls(mean_vec=mean, cov_matrix=cov)

    @classmethod
    def as_marginal(cls, dim):
        mean = np.zeros((dim,), dtype=float)
        #diag_cov = (np.inf) * np.ones((dim,), dtype=float)
        #cov = np.diag(diag_cov)
	cov = 99999 * np.eye(dim)
        return cls(mean_vec=mean, cov_matrix=cov)


class GaussianFactor(GaussianState):
    def __init__(self, dim):
        mean = np.zeros((dim,), dtype=float)
        diag_cov = np.inf * np.ones((dim,), dtype=float)
        cov = np.diag(diag_cov)
        super().__init__(mean_vec=mean,
                         cov_matrix=cov)
