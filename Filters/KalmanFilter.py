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
from autograd import jacobian
from MomentMatching import TaylorTransform
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
        self.dt = system_model.dt

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
        meas = np.atleast_1d(meas)
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


class IEKF(KalmanFilterSmoother):
    def __init__(self, system_model, sys_dim):
        transform = TaylorTransform(dim=sys_dim)
        super().__init__(transform, system_model)

    def _iterated_update(self, state, meas, t=None, u=None, *args, **kwargs):
        x = state.mean
        P = state.cov
        for i in range(self.num_iter):
            h_i = self.measurement(state.mean, t, u)[0]
            H_i = jacobian(self.measurement, argnum=0)(state.mean, t, u)[0]
            z_mean = h_i - H_i @ (x - state.mean)
            z_cov = H_i @ P @ H_i.T + self.measurement_noise
            xz_cross_cov = P @ H_i.T

            kalman_gain = np.linalg.solve(z_cov, xz_cross_cov.T).T
            meas = np.atleast_1d(meas)
            mean = x + np.dot(kalman_gain, (meas - z_mean)) # equation 15  in Marc's ACC paper
            cov = P - np.dot(kalman_gain, np.transpose(xz_cross_cov))

            state = Gaussian(mean, cov)

        return state

    def __call__(self, measurements, num_iter=5, prior_state=None, t_zero=0.0, u=None, *args, **kwargs):
        self.num_iter = num_iter
        if prior_state is None:
            prior_state = self.init_state

        state = prior_state
        t = t_zero

        result_filter = []

        for i, measurement in enumerate(measurements):
            pred_state = self.predict(prior_state, t=t, u=u, *args, **kwargs)
            logger.debug('{},{},{}'.format(prior_state, t, pred_state))
            corrected_state = self._iterated_update(pred_state, measurement.squeeze(), t=t, u=u, *args, **kwargs)
            result_filter.append(corrected_state)
            t += self.dt
            prior_state = corrected_state

        return result_filter


class IEKS(KalmanFilterSmoother):
    def __init__(self, system_model, sys_dim):
        transform = TaylorTransform(dim=sys_dim)
        super().__init__(transform, system_model)
        self.f = system_model.transition
        self.h = system_model.measurement
        self.Df = jacobian(system_model.transition, argnum=0)
        self.Dh = jacobian(system_model.measurement, argnum=0)

    @staticmethod
    def _linearise(func, Df, x0):
        return lambda x, t, u: func(x0, t=t, u=u) + Df(x0, t=t, u=u) @ (x - x0)

    def _kalman_filter(self, measurements, prior_state=None, t_zero=0.0, u=None, *args, **kwargs):

        if prior_state is None:
            prior_state = self.init_state

        state = prior_state
        t = t_zero

        result_filter = []

        for i, measurement in enumerate(measurements):
            self.transition = self.f_list[i]
            self.measurement = self.h_list[i]
            pred_state = self.predict(prior_state, t=t, u=u, *args, **kwargs)
            logger.debug('{},{},{}'.format(prior_state, t, pred_state))
            corrected_state = self.correct(pred_state, measurement.squeeze(), t=t, u=u, *args, **kwargs)
            result_filter.append(corrected_state)
            t += self.dt
            prior_state = corrected_state

        return result_filter

    def _kalman_smoother(self, filtered_list, u=None, *args, **kwargs):
        reversed_filtered = reversed(filtered_list)
        N = len(filtered_list)
        t = (N-1) * self.dt

        result = []

        next_state = next(reversed_filtered).copy()
        result.append(next_state)
        i = len(filtered_list)-1
        for state in reversed_filtered:
            self.transition = self.f_list[i]
            smoothed_state = self.smooth(state, next_state, t=t, u=u, *args, **kwargs)
            result.append(smoothed_state)
            next_state = smoothed_state.copy()
            t -= self.dt
            i -= 1

        return list(reversed(result))

    def __call__(self, measurements, num_iter=5):
    #     self.f_list = [self._linearise(self.f, self.Df, self.init_state.mean) for _ in measurements]
    #     self.h_list = [self._linearise(self.h, self.Dh, self.init_state.mean) for _ in measurements]

        # First iteration
        filter_results = self.kalman_filter(measurements)
        smoother_results = self.kalman_smoother(filter_results)

        self.f_list = [self._linearise(self.f, self.Df, state.mean) for state in smoother_results]
        self.h_list = [self._linearise(self.h, self.Dh, state.mean) for state in smoother_results]

        for k in range(num_iter-1):

            # print(f'iteration: {k+1}/{num_iter}')
            filter_results = self._kalman_filter(measurements)
            smoother_results = self._kalman_smoother(filter_results)

            # Update linearisation points
            self.f_list = [self._linearise(self.f, self.Df, state.mean) for state in smoother_results]
            self.h_list = [self._linearise(self.h, self.Dh, state.mean) for state in smoother_results]
        
        return smoother_results