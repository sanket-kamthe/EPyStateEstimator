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
from .ExpectationPropagation import TopEP
from numpy.linalg import LinAlgError
from functools import partial, partialmethod

class OverLoadError(Exception):
    pass

class TimeSeriesNodeForEP:
    def __init__(self, t, state_dim=1, marginal_init=None, factor_init=None):

        self.t = t
        self.state_dimension=state_dim
        if marginal_init is None:
            self.marginal = Gaussian.as_marginal(dim=state_dim)
        else:
            self.marginal = marginal_init

        if factor_init is None:
            self.measurement_factor = Gaussian.as_factor(dim=state_dim)
            self.back_factor = Gaussian.as_factor(dim=state_dim)
            self.forward_factor = Gaussian.as_factor(dim=state_dim)
        else:
            self.measurement_factor, self.back_factor, self.forward_factor = factor_init

        self.factors = ['forward_update', 'measurement_update', 'backward_update']

        self.converged = False

    def copy(self):
        marginal_init = self.marginal.copy()
        factor_init = self.measurement_factor.copy(), self.back_factor.copy(), self.forward_factor.copy()
        return TimeSeriesNodeForEP(t=self.t,
                                   state_dim=self.state_dimension,
                                   marginal_init=marginal_init,
                                   factor_init=factor_init)

    def __repr__(self):
        str_rep = '''{}.t={}, state_dim={}, marginal_init={},
         factor_init={})'''.format(self.__class__,
                                   self.t,
                                   self.marginal,
                                   (self.measurement_factor, self.back_factor, self.forward_factor))
        return str_rep

class TimeSeriesNodeForEP:
    def __init__(self, t, state_dim=1,
                 marginal_init=None,
                 factor_init=None, transition=None, measurement=None):

        self.t = t
        self.state_dimension=state_dim
        if marginal_init is None:
            self.marginal = Gaussian.as_marginal(dim=state_dim)
        else:
            self.marginal = marginal_init

        if factor_init is None:
            self.measurement_factor = Gaussian.as_factor(dim=state_dim)
            self.back_factor = Gaussian.as_factor(dim=state_dim)
            self.forward_factor = Gaussian.as_factor(dim=state_dim)
        else:
            self.measurement_factor, self.back_factor, self.forward_factor = factor_init

        self.factors = ['forward_update', 'measurement_update', 'backward_update']

        self.converged = False

    def copy(self):
        marginal_init = self.marginal.copy()
        factor_init = self.measurement_factor.copy(), self.back_factor.copy(), self.forward_factor.copy()
        return TimeSeriesNodeForEP(t=self.t,
                                   state_dim=self.state_dimension,
                                   marginal_init=marginal_init,
                                   factor_init=factor_init)

    def __repr__(self):
        str_rep = '''{}.t={}, state_dim={}, marginal_init={},
         factor_init={})'''.format(self.__class__,
                                   self.t,
                                   self.marginal,
                                   (self.measurement_factor, self.back_factor, self.forward_factor))
        return str_rep

class Nodes():
    def __init__(self, measurements, base_node, init_state):
        self.base_node = base_node
        self._nodes = []
        for i, measurement in enumerate(measurements):

            self._nodes.append()



######################################################
# estimator
#     pred
#     correct
#     smooth
#     fwd
#     back
#     meas
# Top nodes

def node_estimator(nodes, estimator):

    out_nodes = []

    power = getattr(estimator, 'power', 1)
    damping = getattr(estimator, 'damping', 1)

    for node in nodes:
        setattr(node, 'fwd_update', partial(node.fwd_update, proj_trans=estimator.proj_trans))
        setattr(node, 'meas_update', partial(node.meas_update, proj_meas=estimator.proj_meas))
        setattr(node, 'back_update', partial(node.back_update, proj_trans=estimator.proj_back))

        setattr(node, 'power', power)
        setattr(node, 'damping', damping)

        out_nodes.append(node)

    return out_nodes


def node_system(nodes, system_model, measurements, farg_list=None):
    out_nodes = []
    N = len(measurements)

    if farg_list is None:
        farg_list = []
        t = 0
        for i in range(N):
            f_kwargs = {'t': t, 'u': None}
            t += system_model.dt
            farg_list.append(f_kwargs)

    for node, f_kwarg in zip(nodes, farg_list):
        setattr(node, 'trans_func', partial(system_model.transition, **f_kwarg))
        setattr(node, 'meas_func', system_model.measurement)

        out_nodes.append(node)

    return out_nodes


class Node:
    def __init__(self, dim, prev_node=None, next_node=None, factor_init=None, marginal_init=None):
        self.next_node = next_node
        self.prev_node = prev_node

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

        self.factors = ['forward_update', 'measurement_update', 'backward_update']

        self.converged = False

    def __copy__(self):
        cls = self.__class__
        newone = cls.__new__(cls)
        newone.__dict__.update(self.__dict__)
        return newone

    def fwd_update(self, proj_trans):
        fwd_cavity = self.marginal / self.forward_factor
        prev_node = self.prev_node.copy()
        back_cavity = prev_node.marginal / prev_node.back_factor

        try:
            state = proj_trans(self.trans_func, back_cavity)
        except LinAlgError:
            return

        self.forward_factor = (self.forward_factor ** (1 - self.damping)) * (state ** (self.damping))
        self.marginal = self.marginal * (self.forward_factor / self.forward_factor)

    def meas_update(self, proj_meas):
        measurement_cavity = self.marginal / self.measurement_factor

        try:
            state = proj_meas(self.meas_func, measurement_cavity, self.meas)
        except LinAlgError:
            return

        self.measurement_factor, self.marginal = \
            self.power_update(projected_marginal=state,
                              factor=self.measurement_factor,
                              marginal=self.marginal,
                              cavity=measurement_cavity)

    def back_update(self, proj_back):

        back_cavity = node.marginal / node.back_factor
        next_node = self.next_node.copy()
        # forward_cavity = next_node.marginal / next_node.forward_factor

        try:
            state = proj_back(self.trans_func,
                              back_cavity,
                              next_node.marginal)
        except LinAlgError:
            return


        self.back_factor, self.marginal = \
            self.power_update(projected_marginal=state,
                              factor=self.back_factor,
                              marginal=self.marginal,
                              cavity=back_cavity)


    def power_update(self, projected_marginal, factor, marginal, cavity):
        damping = self.damping
        power = self.power
        # projected_marginal = project(f, tilted_marginal)  # PowerEP equation 21
        new_factor = (factor ** (1 - damping)) * ((projected_marginal / cavity) ** damping)  # PowerEP equation 22
        new_marginal = marginal * ((projected_marginal / marginal) ** (damping / power))  # PowerEP equation 23

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


def build_nodes(N, dim, prior=None):
    nodes = [Node(dim) for _ in range(N)]

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
    return nodes