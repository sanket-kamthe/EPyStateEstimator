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

from .DynamicSystemEP import DynamicSystemEP
from numpy.linalg import LinAlgError
from functools import partial



class DynamicSystemPowerEP(DynamicSystemEP):

    def __init__(self, system, ep_project, measurements,
                 power=1, damping=1):
        self.power = power
        self.damping = damping

        super().__init__(system=system,
                         ep_project=ep_project,
                         measurements=measurements)

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

        result_node.marginal = (node.forward_factor ** (1 - self.damping)) * (state ** (self.damping))
        result_node.forward_factor = node.marginal * (result_node.forward_factor / node.forward_factor)
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



        result_node.measurement_factor, result_node.marginal = \
            self.power_update(projected_marginal=state,
                              factor=node.measurement_factor,
                              marginal=node.marginal,
                              cavity=measurement_cavity)
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

        result_node.back_factor, result_node.marginal = \
            self.power_update(projected_marginal=state,
                              factor=node.back_factor,
                              marginal=node.marginal,
                              cavity=back_cavity)

        return result_node

    def power_update(self, projected_marginal, factor, marginal, cavity):
        damping = self.damping
        power = self.power
        # projected_marginal = project(f, tilted_marginal)  # PowerEP equation 21
        new_factor = (factor ** (1 - damping)) * ((projected_marginal / cavity) ** damping)  # PowerEP equation 22
        new_marginal = marginal * ((projected_marginal/marginal) ** (damping/power))   # PowerEP equation 23

        return new_factor, new_marginal

