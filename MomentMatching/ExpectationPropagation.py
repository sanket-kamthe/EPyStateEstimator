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
from collections import  namedtuple
import itertools
from collections.abc import MutableSequence

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


#
# class TimeSeriesNode:
#     def __init__()
#         self.marginal = None  # This is a Gaussian density for example
#         self.factors =[]
#         self.t = 0
#
#     def forward_update(self):

class BeginNode(BaseNode):
    def __init__(self, state, factors):
        super(BeginNode, self).__init__(state, factors)

    def ep_update(self, factor, project=None, f=None, power=None, damping=None):
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
        projected_marginal = project(f, tilted_marginal)  # PowerEP equation 21
        new_factor = factor * ((projected_marginal / marginal) ** damping)  # PowerEP equation 22
        new_marginal = marginal * ((projected_marginal/marginal) ** (power * damping))  # PowerEP equation 23



class TimeSeriesNodeForEP:
    def __init__(self, t, marginal=None, factors=None):
        self.t = t
        self.marginal = marginal
        self.factors = factors


class EPNodes(MutableSequence):
    def __init__(self, N, marginal_init=None, factors_init=None):
        self._Node = []
        for i in range(N):
            self._Node.append(TimeSeriesNodeForEP(i, marginal=marginal_init,
                                                  factors=factors_init)
                              )
        # self._Node = list(itertools.repeat(init_node, N))
        self.mode_select = [1, 1, 1]

    def validate_input(self, value):
        pass

    def __len__(self):
        return len(self._Node)

    def __getitem__(self, key):
        return self._Node[key]

    def __delitem__(self, key):
        del self._Node[key]

    def __setitem__(self, key, value):
        self.validate_input(value)
        self._Node[key] = value

    def insert(self, index, value):

        self.validate_input(value)
        self._Node.insert(index, value)

    def __str__(self):
        return f'EP Nodes with {len(self._Node)} items in list ' + str(self._Node[0])

    def filter_mode(self):
        mode_select = [1, 1, 0]
        for node in self._Node:
            mode = itertools.compress(node.factors, mode_select)
            for factor in mode:
                print(f'In Node{node.t} {factor} factor')

    def smoother_mode(self):
        self.filter_mode()
        mode_select = [0, 0, 1]
        for node in reversed(self._Node):
            mode = itertools.compress(node.factors, mode_select)
            for factor in mode:
                print(f'In Node{node.t} {factor} factor')



