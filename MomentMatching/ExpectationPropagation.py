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
from .StateModels import GaussianState
from .baseMomentMatch import MomentMatching

# class GaussianState(object):
#     def __init__(self, mean, variance):
#         self.mean = mean
#         self.variance = variance
#         self.precision = np.linalg.solve(variance, np.eye(3, dtype=float))
# #         TODO: Add checks for the variance and mean shapes
#
#
# def gauss_divide(numerator, denominator, n_alpha=None, damping_factor=None):
#     variance = np.linalg.inv(numerator.precision - denominator.precision)
#     scaled_mean = np.dot(numerator.precision, numerator.mean) - np.dot(denominator.precision, denominator.mean)
#     mean = np.dot(variance, scaled_mean)
#     return GaussianState(mean, variance)

class BaseNode(object):
    """
    Factory to create nodes
    """
    FACTORS = ['Fwd', 'Measurement', 'Back']

    def __init__(self, state, factors=None):
        self.state = state
        if factors is None:
            factors = self.FACTORS
        self._factors = [factor for factor in factors]

    def update(self, factor=None):
        return NotImplementedError

    # def


class BeginNode(BaseNode):
    def __init__(self, state, factors):
        super(BeginNode, self).__init__(state, factors)

    def ep_update(self, factor, power=None, damping=None):
        """
        Psuedo code


        :param factor:
        :return:
        """

        #  First calculate the cavity distribution q_back

        # compute projection

        #
    #def
    # def _forward_update(self, mean, variance):
        marginal = self.state
        tilted_marginal = marginal / factor   # PowerEP equation 20
        projected_marginal = project(f(x), tilted_marginal)  # PowerEP equation 21
        new_factor = factor * ((projected_marginal / marginal) ** damping)  # PowerEP equation 22
        new_marginal = marginal * ((projected_marginal/marginal) ** (power * damping))  # PowerEP equation 23



