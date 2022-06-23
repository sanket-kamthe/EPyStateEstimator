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

from abc import abstractmethod


class MomentMatching():
    _params = {}

    @abstractmethod
    def project(self, func, distribution, noise=None, meas=None):
        """
        Returns the approximate projection of distribution through a nonlinear function.

        ..math \\int f(x) q (x) dx, where q(x) is the distribution and f(x) is the non-linear function
        The result is exact when f(x) is a linear function.

        Parameters
        -----------------------------------------------------
        :param func: a nonlinear function in x
        :param distribution: distribution of the variable x
        :param meas: either a distribution or measurement (observation)
        :param noise: noise parameter
        :type func: any python function
        :type distribution: GaussianState object

        Returns
        ---------------------------------------------

        :return: approximate distribution in the same family as input distribution
        :rtype: object of the type 'distribution'
        """
        pass


class MappingTransform:
    """
    Parent class for moment matching methods (e.g. Taylor transform, Unscented transform)
    """
    def __init__(self, approximation_method=None, **kwargs):
        self.params = {}  # Dictionary containing parameters for the moment matching approximation

        self.method = approximation_method
        self.params.update(kwargs)
        # print(self.params)
        for key in self.params:
            self.__setattr__(key, self.params[key])

    def _transform(self, func, state):
        """
        Returns the gaussian approximation the integral

        :math: \int f(x) q (x) dx, where q(x) is the distribution and f(x) is the non-linear function
        The result is exact when f(x) is a linear function.
        :param nonlinear_func:
        :param distribution: object of type GaussianState for example
        :return: distribution of y = :math: \int f(x) q (x) dx in the form of a tuple
        mean (y), variance (yyT), cross_covariance(xyT)
        """
        return NotImplementedError

    def __call__(self, func, state):
        return self._transform(func=func, state=state)

