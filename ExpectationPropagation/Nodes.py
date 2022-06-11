# Copyright 2018 Sanket Kamthe
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
from StateModel import Gaussian
from numpy.linalg import LinAlgError
from Utils import validate_covariance
from functools import partial
from MomentMatching import Estimator
from Systems import DynamicSystemModel
from typing import List, Union


class OverLoadError(Exception):
    pass


class Node:
    """
    A single node of the factor graph. Consists of the marginal distribution and methods to update the marginal.
    """
    def __init__(self, dim, index=0, prev_node=None, next_node=None, factor_init=None, marginal_init=None):
        self.next_node = next_node
        self.prev_node = prev_node
        self.index = index

        self.dim = dim
        if marginal_init is None:
            self.marginal = Gaussian.as_marginal(dim)
        else:
            self.marginal = marginal_init

        if factor_init is None:
            self.measurement_factor = Gaussian.as_factor(dim)
            self.back_factor = Gaussian.as_factor(dim)
            self.forward_factor = Gaussian.as_factor(dim)
        else:
            self.measurement_factor, self.back_factor, self.forward_factor = factor_init

        self.converged = False

        self.factors = ['forward_update', 'measurement_update', 'backward_update']

    def copy(self):
        cls = self.__class__
        newone = cls.__new__(cls)
        newone.__dict__.update(self.__dict__)
        return newone

    def fwd_update(self, proj_trans):
        # fwd_cavity = self.marginal / self.forward_factor
        old_forward_factor = self.forward_factor
        try:
            prev_node = self.prev_node.copy()
            back_cavity = prev_node.marginal / (prev_node.back_factor ** (1 / self.power))
        except AttributeError:
            back_cavity = self.prior

        try:
            state = proj_trans(self.trans_func, back_cavity)
            validate_covariance(state)

            fwd_factor = (self.forward_factor ** (1 - self.damping)) * (state ** self.damping)
            marginal = self.marginal * (fwd_factor / old_forward_factor) ** (1 / self.power)
            validate_covariance(marginal)
        except LinAlgError:
            return

        self.forward_factor, self.marginal = fwd_factor, marginal

    def meas_update(self, proj_meas):
        measurement_cavity = self.marginal / self.measurement_factor

        try:
            state = proj_meas(self.meas_func, measurement_cavity, self.meas)
            validate_covariance(state)

            self.measurement_factor, self.marginal = \
                self.power_update(projected_marginal=state,
                                  factor=self.measurement_factor,
                                  marginal=self.marginal,
                                  cavity=measurement_cavity)
        except LinAlgError:
            return

    def back_update(self, proj_back):

        back_cavity = self.marginal / self.back_factor

        try:
            next_node = self.next_node.copy()
            next_fwd_cavity = next_node.marginal / (next_node.forward_factor ** (1 / self.power))
        except AttributeError:
            return
        # forward_cavity = next_node.marginal / next_node.forward_factor

        try:
            state = proj_back(next_node.trans_func,
                              back_cavity,
                              next_fwd_cavity,
                              next_state=next_node.marginal)
            validate_covariance(state)

            self.back_factor, self.marginal = \
                self.power_update(projected_marginal=state,
                                  factor=self.back_factor,
                                  marginal=self.marginal,
                                  cavity=back_cavity)
        except LinAlgError:
            return

    def power_update(self, projected_marginal, factor, marginal, cavity):
        damping = self.damping
        power = self.power
        # if power == 1
        # projected_marginal = project(f, tilted_marginal)  # PowerEP equation 21
        marginal_ratio = (projected_marginal / marginal)
        new_factor = factor * (marginal_ratio ** damping)  # PowerEP equation 22
        new_marginal = marginal * (marginal_ratio ** (damping / power))  # PowerEP equation 23

        validate_covariance(new_marginal)

        return new_factor, new_marginal

    def trans_func(self):
        '''
        This function should be overloaded. Failure to do so raises error.
        :return:
        '''
        raise OverLoadError

    def meas_func(self):
        '''
        This function should be overloaded. Failure raises error.
        :return:

        '''
        raise OverLoadError


def node_estimator(nodes: List[Node], estimator: Estimator) -> List[Node]:
    """
    This function adds the moment matching method as attributes of the factor graph.
    :nodes: Factor graph (built using build_nodes)
    :estimator: Moment matching method (e.g. Taylor transformation)
    :return: Updated factor graph
    """
    out_nodes = []

    power = getattr(estimator, 'power', 1)
    damping = getattr(estimator, 'damping', 1)

    for node in nodes:
        setattr(node, 'fwd_update', partial(node.fwd_update, proj_trans=estimator.proj_trans))
        setattr(node, 'meas_update', partial(node.meas_update, proj_meas=estimator.proj_meas))
        setattr(node, 'back_update', partial(node.back_update, proj_back=estimator.proj_back))

        setattr(node, 'power', power)
        setattr(node, 'damping', damping)

        out_nodes.append(node)

    return out_nodes


def node_system(
    nodes: List[Node],
    system_model: DynamicSystemModel,
    measurements: Union[np.ndarray, list],
    farg_list: List[dict] = None
    ) -> List[Node]:
    """
    This function adds the transition function, measurement function and observations 
    as attributes of the factor graph.
    :nodes: Factor graph (built using build_nodes)
    :system_model: Dynamical system
    :measurements: Observations y
    :farg_list: kwargs for transition function
    :return: Updated factor graph
    """
    out_nodes = []
    N = len(measurements)

    if farg_list is None:
        farg_list = []
        t = 0.0
        for i in range(N):
            f_kwargs = {'t': t, 'u': 0.0}
            t += system_model.dt
            farg_list.append(f_kwargs)

    for node, measurement, f_kwarg in zip(nodes, measurements, farg_list):
        setattr(node, 'trans_func', partial(system_model.transition, **f_kwarg))
        setattr(node, 'meas_func', system_model.measurement)
        setattr(node, 'meas', np.squeeze(measurement))

        out_nodes.append(node)

    setattr(out_nodes[0], 'prior', system_model.init_state)
    return out_nodes


def build_nodes(
    N: int,
    dim: int,
    estimator: Estimator = None,
    system: DynamicSystemModel = None
    ) -> List[Node]:
    """
    This function builds the factor graph representation of a dynamical system (Figure 2).
    :N: Number of timesteps
    :dim: State dimension
    :estimator: Method of moment matching (e.g. Taylor transformation)
    :system: Dynamical system
    :return: Factor graph representation of a dynamical system
    """
    nodes = [Node(dim, index=i) for i in range(N)]

    for i, node in enumerate(nodes):
        if i == 0:
            node.prev_node = None
            node.next_node = nodes[i + 1]
            continue
        if i == (N - 1):
            node.prev_node = nodes[i - 1]
            node.next_node = None
            continue

        node.next_node = nodes[i + 1]
        node.prev_node = nodes[i - 1]

    if estimator is not None:
        nodes = node_estimator(nodes=nodes, estimator=estimator)

    if system is not None:
        nodes = node_system(nodes=nodes, system_model=system)

    return nodes
