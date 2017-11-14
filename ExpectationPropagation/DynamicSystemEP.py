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
from MomentMatching import ProjectTransition, ProjectMeasurement


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

    @abstractmethod
    def project_transition(self, distribution):
        pass

    @abstractmethod
    def project_measurement(self, distribution):
        pass


class DynamicSystemEP(TimeSeriesEP):

    def __init__(self, project_transition, project_measurement):

        assert isinstance(project_measurement, ProjectMeasurement)
        assert isinstance(project_transition, ProjectTransition)

        self._transition = project_transition
        self._measurement = project_measurement

    def fwd_update(self, node, prev_node):
        forward_cavity = node.marginal / node.forward_factor
        back_cavity = prev_node.marginal / prev_node.back_factor

        try:
            state = self.project_transition(distribution=back_cavity)
        except LinAlgError:
            return node.copy()
        else:
            result_node = node.copy()

        result_node.forward_factor = state.copy()
        result_node.marginal = forward_cavity * result_node.forward_factor

        return result_node

    def meas_update(self, node, meas):

        measurement_cavity = node.marginal / node.measurement_factor

        try:
            state = self.project_measurement(distribution=measurement_cavity)
        except LinAlgError:
            return node.copy()
        else:
            result_node = node.copy()

        result_node.marginal = state.copy()
        result_node.measurement_factor = result_node.marginal / measurement_cavity

        return result_node

    def back_update(self, node, next_node):
        back_cavity = node.marginal / node.back_factor
        forward_cavity = next_node.marginal / next_node.forward_factor

        try:
            state = self.project_transition(distribution=back_cavity, state=next_node.marginal)
        except LinAlgError:
            return node.copy()
        else:
            result_node = node.copy()

        result_node.marginal = state.copy()
        result_node.back_factor = result_node.marginal / back_cavity

        return result_node

    def project_measurement(self, distribution):
        return self._measurement.project(distribution)

    def project_transition(self, distribution):
        return self._transition.project(distribution)
