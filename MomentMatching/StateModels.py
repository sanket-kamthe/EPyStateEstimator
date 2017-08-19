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

#
# from abc import ABC, abstractmethod
#
# class StateModel(ABC):
#
#     @abstractmethod
#     def


# class StateModel:
#     def __init__(self, *parameters):
#         self.parameters = parameters

# Import Statements

import numpy as np

def natural_to_moment(precision, shift):
    cov = np.linalg.inv(precision)
    mean = np.dot(cov, shift)
    return mean, cov

def moment_to_natural(mean, cov):
    precision = np.linalg.inv(cov)
    shift = np.dot(precision, mean)
    return precision, shift

class GaussianState:
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
        self.dim = dim
        self.mean = mean_vec
        self.cov = cov_matrix

        # Kill the warning for instance attributes
        # self._mean = None
        # self._cov = None

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
            self._precision = np.linalg.inv(self.cov)  # TODO: Change to more stable solve later
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
        mean, cov  = natural_to_moment(precision, shift)
        return GaussianState(mean, cov)

    def __truediv__(self, other):
        # Make sure that 'other' is also a GaussianState class
        # TODO: Replace assert with a custom Error
        assert isinstance(other, GaussianState)
        precision = self.precision - other.precision
        shift = self.shift - other.shift
        mean, cov = natural_to_moment(precision, shift)
        return GaussianState(mean, cov)

    def __pow__(self, power, modulo=None):
        precision = power * self.precision
        shift = power * self.shift
        mean, cov = natural_to_moment(precision, shift)
        return GaussianState(mean, cov)

    def __eq__(self, other):
        # Make sure that 'other' is also a GaussianState class
        # TODO: Replace assert with a custom Error
        assert isinstance(other, GaussianState)
        mean_equal = np.allclose(self.mean, other.mean)
        cov_equal = np.allclose(self.cov, other.cov)

        return mean_equal and cov_equal
