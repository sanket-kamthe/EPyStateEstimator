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


class MomentMatching:
    params = {}  # Dictionary containing parameters for the moment matching approximation

    def __init__(self, approximation_method=None):
        self.method = approximation_method

    def project(self, nonlinear_func, distribution):
        """
        Returns the gaussian approximation the integral
        \int f(x) q (x) dx, where q(x) is the distribution and f(x) is the non-linear function
        The result is exact when f(x) is a linear function.
        :param nonlinear_func:
        :param distribution: object of type GaussianState for example
        :return:
        same object as the type of distribution.
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
