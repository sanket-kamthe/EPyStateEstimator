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
from StateModel import Gaussian
from itertools import tee
from functools import partial
import logging

FORMAT = "[ %(funcName)10s() ] %(message)s"

logging.basicConfig(filename='kalman_filter.log', level=logging.FATAL, format=FORMAT)
logger = logging.getLogger(__name__)


def pairwise(x):
    node, next_node = tee(x)
    next(node, None)
    yield zip(node, next_node)


class KalmanFilterSmoother:
    def __init__(self, moment_matching, system_model, meas_moment_matching=None):

        self.transform = moment_matching

        if meas_moment_matching is None:
            self.meas_transform = moment_matching
        else:
            self.meas_transform = meas_moment_matching
        self.transition = system_model.transition
        self.measurement = system_model.measurement
        self.transition_noise = system_model.system_noise.cov
        self.measurement_noise = system_model.measurement_noise.cov
        self.init_state = system_model.init_state
        self.dt =system_model.dt

    def predict(self, prior_state, t=None, u=None, *args, **kwargs):
        func = partial(self.transition, t=t, u=u)
        xx_mean, xx_cov, _ = self.transform(func, prior_state)
        xx_cov += self.transition_noise
        return Gaussian(xx_mean, xx_cov)

    def correct(self, state, meas, t=None, u=None, *args, **kwargs):
        func = partial(self.measurement, t=t, u=u)
        z_mean, z_cov, xz_cross_cov = \
            self.meas_transform(func, state)

        z_cov += self.measurement_noise

        kalman_gain = np.linalg.solve(z_cov, xz_cross_cov.T).T
        mean = state.mean + np.dot(kalman_gain, (meas - z_mean)) # equation 15  in Marc's ACC paper
        cov = state.cov - np.dot(kalman_gain, np.transpose(xz_cross_cov))

        return Gaussian(mean, cov)

    def smooth(self, state, next_state, t=None, u=None, *args, **kwargs):
        func = partial(self.transition, t=t, u=u)
        xx_mean, xx_cov, xx_cross_cov = self.transform(func, state)

        xx_cov += self.transition_noise

        J = np.linalg.solve(xx_cov, xx_cross_cov.T).T
        mean = state.mean + np.dot(J, (next_state.mean - xx_mean))
        cov = state.cov + J @ (next_state.cov - xx_cov) @ J.T

        return Gaussian(mean, cov)

    def kalman_filter(self, measurements, prior_state=None, t_zero=0.0, u=None, *args, **kwargs):

        if prior_state is None:
            prior_state = self.init_state

        state = prior_state
        t = t_zero

        result_filter = []

        for i, measurement in enumerate(measurements):
            pred_state = self.predict(prior_state, t=t, u=u, *args, **kwargs)
            logger.debug('{},{},{}'.format(prior_state, t, pred_state))
            corrected_state = self.correct(pred_state, measurement.squeeze(), t=t, u=u, *args, **kwargs)
            result_filter.append(corrected_state)
            t += self.dt
            prior_state = corrected_state

        return result_filter

    def kalman_smoother(self, filtered_list, u=None, *args, **kwargs):
        reversed_filtered = reversed(filtered_list)
        N = len(filtered_list)
        t = (N-1) * self.dt

        result = []

        next_state = next(reversed_filtered).copy()
        result.append(next_state)
        for state in reversed_filtered:
            smoothed_state = self.smooth(state, next_state, t=t, u=u, *args, **kwargs)
            result.append(smoothed_state)
            next_state = smoothed_state.copy()
            t -= self.dt

        return list(reversed(result))


