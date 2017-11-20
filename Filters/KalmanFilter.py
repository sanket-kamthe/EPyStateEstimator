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
from MomentMatching.newMomentMatch import MomentMatching
from MomentMatching.TimeSeriesModel import TimeSeriesModel, DynamicSystemModel
# from MomentMatching.StateModels import GaussianState
from StateModel import GaussianState
from itertools import tee
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
        # assert isinstance(system_model, DynamicSystemModel)  # make sure we are working with the time series model.
        # assert isinstance(moment_matching, MomentMatching)

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
        xx_mean, xx_cov, _ = self.transform(self.transition,
                                            prior_state,
                                            t=t, u=u,
                                            *args, **kwargs)
        xx_cov += self.transition_noise
        return GaussianState(xx_mean, xx_cov)

    def correct(self, state, meas, t=None, u=None, *args, **kwargs):

        z_mean, z_cov, xz_cross_cov = \
            self.meas_transform(self.measurement,
                                state,
                                t=t, u=u,
                                *args, **kwargs)

        z_cov += self.measurement_noise

        # kalman_gain = np.matmul(xz_cross_cov, np.linalg.pinv(z_cov))
        kalman_gain = np.linalg.solve(z_cov, xz_cross_cov.T).T
        mean = state.mean + np.dot(kalman_gain, (meas - z_mean)) # equation 15  in Marc's ACC paper
        cov = state.cov - np.dot(kalman_gain, np.transpose(xz_cross_cov))

        return GaussianState(mean, cov)

    def smooth(self, state, next_state, t=None, u=None, *args, **kwargs):

        xx_mean, xx_cov, xx_cross_cov = \
            self.transform(self.transition,
                           state,
                           t=t, u=u,
                           *args, **kwargs)

        xx_cov += self.transition_noise

        # J = xx_cross_cov @ np.linalg.pinv(xx_cov)
        J = np.linalg.solve(xx_cov, xx_cross_cov.T).T
        mean = state.mean + np.dot(J, (next_state.mean - xx_mean))
        cov = state.cov + J @ (next_state.cov - xx_cov) @ J.T

        return GaussianState(mean, cov)

    def kalman_filter(self, measurements, prior_state=None, t_zero=0.0, u=None, *args, **kwargs):

        if prior_state is None:
            prior_state = self.init_state

        state = prior_state
        t = t_zero

        result_filter = []

        for i, measurement in enumerate(measurements):
            pred_state = self.predict(prior_state, t=t, u=u, *args, **kwargs)
            logger.debug('{},{},{}'.format(prior_state, t, pred_state))
            corrected_state = self.correct(pred_state, measurement, t=t, u=u, *args, **kwargs)
            result_filter.append(corrected_state)
            t += self.dt
            prior_state = corrected_state

        return result_filter

    def kalman_smoother(self, filtered_list, u=None, *args, **kwargs):
        reversed_filtered = reversed(filtered_list)
        N = len(filtered_list)
        t = (N-1) * self.dt

        # data = pairwise(reversed_filtered_list)
        result = []

        next_state = next(reversed_filtered).copy()
        result.append(next_state)
        for state in reversed_filtered:
            smoothed_state = self.smooth(state, next_state, t=t, u=u, *args, **kwargs)
            result.append(smoothed_state)
            next_state = smoothed_state.copy()
            t -= self.dt

        return list(reversed(result))


class PowerKalmanFilterSmoother(KalmanFilterSmoother):

    def __init__(self, moment_matching, system_model, power=1, meas_moment_matching=None):
        self.power = power
        # if meas_moment_matching is None:
        #     self.meas_transform = moment_matching
        # else:
        #     self.meas_transform = meas_moment_matching
        super().__init__(moment_matching=moment_matching,
                         system_model=system_model,
                         meas_moment_matching=meas_moment_matching)


    def predict(self, prior_state, t=None, u=None, *args, **kwargs):
        xx_mean, xx_cov, _ = self.transform(self.transition,
                                            prior_state,
                                            t=t, u=u,
                                            *args, **kwargs)
        xx_cov += self.transition_noise
        xx_cov /= self.power
        return GaussianState(xx_mean, xx_cov)

    def correct(self, state, meas, t=None, u=None, *args, **kwargs):

        z_mean, z_cov, xz_cross_cov = \
            self.meas_transform(self.measurement,
                                state,
                                t=t, u=u,
                                *args, **kwargs)

        z_cov += self.measurement_noise
        z_cov /= self.power

        # kalman_gain = np.matmul(xz_cross_cov, np.linalg.pinv(z_cov))
        kalman_gain = np.linalg.solve(z_cov, xz_cross_cov.T).T
        mean = state.mean + kalman_gain @ (meas - z_mean)  # equation 15  in Marc's ACC paper
        cov = state.cov - kalman_gain @ xz_cross_cov.T

        return GaussianState(mean, cov)

    def smooth(self, state, next_state, t=None, u=None, *args, **kwargs):

        xx_mean, xx_cov, xx_cross_cov = \
            self.transform(self.transition,
                           state,
                           t=t, u=u,
                           *args, **kwargs)

        xx_cov += self.transition_noise
        xx_cov /= self.power

        # J = xx_cross_cov @ np.linalg.pinv(xx_cov)
        J = np.linalg.solve(xx_cov, xx_cross_cov.T).T
        mean = state.mean + np.dot(J, (next_state.mean - xx_mean))
        cov = state.cov + J @ (next_state.cov - xx_cov) @ J.T

        return GaussianState(mean, cov)

