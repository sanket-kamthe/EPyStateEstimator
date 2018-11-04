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

import logging
import numpy as np

FORMAT = "[ %(funcName)10s() ] %(message)s"


logging.basicConfig(filename='Expectation_Propagation.log', level=logging.FATAL, format=FORMAT)
logger = logging.getLogger(__name__)
logger.propagate = False


def power_update(self, projected_marginal, factor, marginal, cavity):
    damping = self.damping
    power = self.power
    # projected_marginal = project(f, tilted_marginal)  # PowerEP equation 21
    new_factor = (factor ** (1 - damping)) * ((projected_marginal / cavity) ** damping)  # PowerEP equation 22
    new_marginal = marginal * ((projected_marginal / marginal) ** (damping / power))  # PowerEP equation 23

    return new_factor, new_marginal

def forward_update(self, node, prev_node, fargs):

    forward_cavity = node.marginal / node.forward_factor
    back_cavity = prev_node.marginal / prev_node.back_factor

    logger.debug('[forward_cavity:: t={} mean={} cov={}]]'.format(node.t,
                                                                  forward_cavity.mean,
                                                                  forward_cavity.cov))
    logger.debug('[back_cavity:: t={} mean={} cov={}]]'.format(node.t,
                                                               back_cavity.mean,
                                                               back_cavity.cov))
    if np.linalg.det(back_cavity.cov) < 0:
        return node.copy()
    result_node = node.copy()

    state = self.kf.predict(prior_state=back_cavity, t=fargs)
    # logger.debug('time {} mean={}, cov={}'.format(node.t, node.marginal.mean, node.marginal.cov))
    # print(state.cov)
    if np.linalg.det(state.cov) > 0:
        result_node.forward_factor = (node.forward_factor ** (1-self.damping)) * (state ** (self.damping))
        result_node.marginal = node.marginal * (result_node.forward_factor / node.forward_factor) ** (1/1)



    return result_node


def measurement_update(self, node, measurement):
    """
    measurement cavity distribution = node_marginal / measurement_factor  # q(x_t) / q_up(x_t)

    new marginal = moment matching with measurement cavity distribution

    new_measurment factor = new_marginal / measurement cavity distribution

    return new_measurement_factor
    """
    assert isinstance(node, TimeSeriesNodeForEP)

    measurement_cavity_distribution = node.marginal / node.measurement_factor  # q(x_t) / q_up(x_t)

    new_node = node.copy()

    new_node.marginal = self.moment_matching(self.measurement, measurement_cavity_distribution, self.R, measurement)

    new_node.measurement_factor = new_node.marginal / measurement_cavity_distribution

    return new_node


def backward_update(self, node, next_node):
    """
    back_cavity_distribution =  q(x_t) / q_back(x_t)
    forward_cavity_distribution =  q(x_t) / q_fwd(x_t)

    new_marginal = moment matching with back cavity distribution

    return new_back_factor
    """

    back_cavity_distribution = node.marginal / node.back_factor  # q(x_t) / q_back(x_t)
    forward_cavity_distribution = next_node.marginal / next_node.forward_factor  # q(x_t+1) / q_fwd(x_t+1)

    new_node = node.copy()

    new_node.marginal = self.moment_matching(self.transition, back_cavity_distribution, forward_cavity_distribution)

    new_node.measurement_factor = new_node.marginal / back_cavity_distribution

    return new_node