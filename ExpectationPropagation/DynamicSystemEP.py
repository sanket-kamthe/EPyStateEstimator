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

from abc import abstractmethod, ABCMeta
from numpy.linalg import LinAlgError
from MomentMatching import MomentMatching
from Systems import DynamicSystem, DynamicSystemModel
import numpy as np


class TimeSeriesEP(metaclass=ABCMeta):

    @abstractmethod
    def fwd_update(self, node, prev_node):
        pass

    @abstractmethod
    def meas_update(self, node, meas):
        pass

    @abstractmethod
    def back_update(self, node, next_node):
        pass

    # @abstractmethod
    # def project_transition(self, distribution):
    #     pass
    #
    # @abstractmethod
    # def project_measurement(self, distribution):
    #     pass


class DynamicSystemEP(TimeSeriesEP):

    def __init__(self, system, ep_project, measurements):

        assert isinstance(ep_project, MomentMatching)
        assert isinstance(system, DynamicSystemModel)

        self._system = system
        self.project = ep_project.project
        self._measurements = measurements

    @property
    def transition(self):
        return self._system.transition

    @property
    def measure(self):
        return self._system.measure

    @property
    def transition_noise(self):
        return self._system.transition_noise

    @property
    def measurement_noise(self):
        return self._system.measurement_noise

    @property
    def prior(self):
        return self._system.init_state

    @staticmethod
    def enumerate_reversed(L):
        """
        Traverse in reverse, from one but last without making any copies

        https://stackoverflow.com/a/529466

        Parameters
        ----------
        L: input list

        Returns
        -------
        generator with index, reversed_list
        """

        for index in reversed(range(len(L) - 1)):
            yield index, L[index]

    @property
    def time_index(self):
        n = len(self._measurements)
        time_index = np.arange(start=0.0, stop=n, step=1) * self._system.dt
        return time_index

    def fwd_update(self, node, prev_node, t=None, u=None):
        forward_cavity = node.marginal / node.forward_factor
        back_cavity = prev_node.marginal / prev_node.back_factor

        try:
            state = self.project(self.transition,
                                 distribution=back_cavity,
                                 noise=self.transition_noise,
                                 t=t, u=u)
        except LinAlgError:
            return node.copy()
        else:
            result_node = node.copy()

        result_node.forward_factor = state.copy()
        result_node.marginal = forward_cavity * result_node.forward_factor

        return result_node

    def meas_update(self, node, meas, t=None, u=None):

        measurement_cavity = node.marginal / node.measurement_factor

        try:
            state = self.project(self.measure,
                                 distribution=measurement_cavity,
                                 meas=meas,
                                 noise=self.measurement_noise,
                                 t=t, u=u)
        except LinAlgError:
            return node.copy()
        else:
            result_node = node.copy()

        result_node.marginal = state.copy()
        result_node.measurement_factor = result_node.marginal / measurement_cavity

        return result_node

    def back_update(self, node, next_node, t=None, u=None):
        back_cavity = node.marginal / node.back_factor
        forward_cavity = next_node.marginal / next_node.forward_factor

        try:
            state = self.project(self.transition,
                                 distribution=back_cavity,
                                 meas=next_node.marginal,
                                 noise=self.transition_noise,
                                 t=t, u=u)
        except LinAlgError:
            return node.copy()
        else:
            result_node = node.copy()

        result_node.marginal = state.copy()
        result_node.back_factor = result_node.marginal / back_cavity

        return result_node

    def kalman_filter(self, nodes):
        prior = nodes[0].copy()
        prior.marginal = self.prior

        for i, (node, meas, t) in enumerate(zip(nodes,
                                                self._measurements,
                                                self.time_index)):

            pred_state = self.fwd_update(node=node,
                                         prev_node=prior,
                                         t=t)

            corr_state = self.meas_update(node=pred_state,
                                          meas=meas,
                                          t=t)
            nodes[i] = corr_state.copy()
            prior = corr_state.copy()

        return nodes

    def kalman_smoother(self, nodes):
        next_node = nodes[-1]
        time_index = self.time_index
        for i, node in self.enumerate_reversed(nodes):
            smoothed_node = self.back_update(node=nodes[i],
                                             next_node=next_node,
                                             t=time_index[i])
            next_node = smoothed_node.copy()
            nodes[i] = next_node

        return nodes

    def ep_update(self, nodes, max_iter=10):

        self.kalman_filter(nodes=nodes)
        self.kalman_smoother(nodes=nodes)
        max_iter -= 1

        prior = nodes[0].copy()
        prior.marginal = self.prior

        for i, (node, meas, t) in enumerate(zip(nodes,
                                                self._measurements,
                                                self.time_index)):

            pred_state = self.fwd_update(node=node,
                                         prev_node=prior,
                                         t=t)

            corr_state = self.meas_update(node=pred_state,
                                          meas=meas,
                                          t=t)

            if (i+1) < len(nodes):
                smthd_node = self.back_update(node=corr_state,
                                              next_node=nodes[i+1],
                                              t=t)













