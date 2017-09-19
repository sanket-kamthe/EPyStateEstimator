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
    def __init__(self, t, state_dim=1, marginal_init=None, factor_init=None):

        self.t = t

        if marginal_init is None:
            self.marginal = self.marginal_init(state_dim)
        else:
            self.marginal = marginal_init

        if factor_init is None:
            self.measurement_factor, self.back_factor, self.forward_factor = (self.initialise_factors(state_dim))
        else:
            self.measurement_factor, self.back_factor, self.forward_factor = factor_init



    # @property
    # def back_factor(self):
    #     return self._back_factor
    #
    # @back_factor.setter
    # def back_factor(self, value):
    #     assert isinstance(value, GaussianState)
    #     self._back_factor = value
    #
    # @property
    # def forward_factor(self):
    #     return self._forward_factor
    #
    # @forward_factor.setter
    # def forward_factor(self, value):
    #     assert isinstance(value, GaussianState)
    #     self._forward_factor = value
    #
    # @property
    # def measurement_factor(self):
    #     return self._measurement_factor
    #
    # @measurement_factor.setter
    # def measurement_factor(self, value):
    #     assert isinstance(value, GaussianState)
    #     self._measurement_factor = value
    #
    #
    @staticmethod
    def marginal_init(state_dim):
        mean = np.zeros((state_dim,), dtype=float)
        cov = 100000 * np.eye(state_dim, dtype=float)
        return GaussianState(mean_vec=mean, cov_matrix=cov)

    # @staticmethod
    # def factor_init(state_dim):
    #
    #     return GaussianState(mean_vec=mean, cov_matrix=cov)

    @staticmethod
    def initialise_factors(state_dim):
        mean = np.zeros((state_dim,), dtype=float)
        cov = 9999 * np.eye(state_dim, dtype=float)
        # self.measurement_factor = self.factor_init(state_dim)
        # self.back_factor = self.factor_init(state_dim)
        # self.forward_factor = self.factor_init(state_dim)
        return (GaussianState(mean_vec=mean, cov_matrix=cov),
                     GaussianState(mean_vec=mean, cov_matrix=cov),
                     GaussianState(mean_vec=mean, cov_matrix=cov))


class EPbase:

    def __init__(self):
        self.me = 'cool'

    def forward_update(self, previous_node, node):
        """
        forward_cavity_distribution = node_marginal / forward_factor  # q(x_t) / q_fwd(x_t)
        back_cavity_distribution = previous_node_marginal / previous_back_factor # q(x_t-1) / q_back(x_t-1)

        new_fwd_factor = moment matching with back_cavity_distribution, transition function and noise Q  #proj op

        new_node_marginal = forward_cavity_distribution * new_fwd_factor

        return new_fwd_factor
        """
        assert isinstance(node, TimeSeriesNodeForEP)
        assert isinstance(previous_node, TimeSeriesNodeForEP)

        forward_cavity_distribution = node.marginal / node.forward_factor  # q(x_t) / q_fwd(x_t)
        back_cavity_distribution = previous_node.marginal / previous_node.back_factor  # q(x_t-1) / q_back(x_t-1)

        new_node = node

        new_node.fwd_factor = self.moment_matching(self.transition, back_cavity_distribution, self.Q)  # proj op

        new_node.marginal = forward_cavity_distribution * new_node.fwd_factor

        return new_node

    def measurement_update(self, node, measurement):
        """
        measurement cavity distribution = node_marginal / measurement_factor  # q(x_t) / q_up(x_t)

        new marginal = moment matching with measurement cavity distribution

        new_measurment factor = new_marginal / measurement cavity distribution

        return new_measurement_factor
        """
        assert isinstance(node, TimeSeriesNodeForEP)

        measurement_cavity_distribution = node.marginal / node.measurement_factor  # q(x_t) / q_up(x_t)

        new_node = node

        new_node.marginal = self.moment_matching(self.measurement, measurement_cavity_distribution, self.R, measurement)

        new_node.measurement_factor = new_node.marginal / measurement_cavity_distribution

        return new_node

    def backward_update(momentmatching, transition_function):
        """
        back_cavity_distribution = node_marginal / back_factor # q(x_t) / q_back(x_t)
        forward_cavity_distribution = next_node_marginal / next_forward_factor  # q(x_t) / q_fwd(x_t)

        new_marginal = moment matching with back cavity distribution

        return new_back_factor
        """


class EPNodes(MutableSequence):
    def __init__(self, dimension_of_state=1, N=None, marginal_init=None, factors_init=None):
        self._Node = []
        if N is not None:
            for t in range(N):
                self._Node.append(TimeSeriesNodeForEP(t=t,
                                                      state_dim=dimension_of_state,
                                                      marginal_init=marginal_init,
                                                      factor_init=factors_init)
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


    def filter_iter(self):

        previous_node, next_node = itertools.tee(self._Node)
        next(next_node, None)
        return zip(previous_node, next_node)



