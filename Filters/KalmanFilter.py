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
from MomentMatching.StateModels import GaussianState


class KalmanFilterSmoother:
    def __init__(self, moment_matching, system_model):
        assert isinstance(system_model, DynamicSystemModel)  # make sure we are working with the time series model.
        assert isinstance(moment_matching, MomentMatching)

        self.transform = moment_matching
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
            self.transform(self.measurement,
                           state,
                           t=t, u=u,
                           *args, **kwargs)

        z_cov += self.measurement_noise

        kalman_gain = np.matmul(xz_cross_cov, np.linalg.pinv(z_cov))

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

        J = xx_cross_cov @ np.linalg.pinv(xx_cov)

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
            corrected_state = self.correct(pred_state, measurement, t=t, u=u, *args, **kwargs)
            result_filter.append(corrected_state)
            t += self.dt
            prior_state = corrected_state

        return result_filter


class KalmanFilterSmootherOld:
    def __init__(self, moment_matching, system_model):
        """
        moment matching implements the transfer function needed.

        """
        assert isinstance(system_model, TimeSeriesModel)  # make sure we are working with the time series model.
        assert isinstance(moment_matching, MomentMatching)

        self.transition = self.system_model.transition_function
        self.observation_model = self.system_model.measurement_function
        self.init_dist = system_model.init_dist
        self.moment_matching = moment_matching
        self.predict = moment_matching.predict
        self.system_model = system_model

    #         self.init_dist = system_model.init_dist

    def kalman_filter(self, noisy_observations, prior_distribution=None):

        if prior_distribution is None:
            x_distribution = self.system_model.init_distribution
        else:
            x_distribution = prior_distribution

        for observation in noisy_observations:
            x_filtered_distribution = self._step_kalman_filter(x_distribution, observation)
            yield x_filtered_distribution
            x_distribution = x_filtered_distribution

    def _filter(self, x_distribution, t, z_measurement):
        pred_mean, pred_cov, _ = self.predict(self.transition, x_distribution, t)

        # Add transition Noise Q_t
        pred_cov = pred_cov + self.system_model.Q.cov
        pred_distribution = GaussianState(pred_mean, pred_cov)

        z_mean, z_cov, xz_cross_cov = self.predict(self.measurement, pred_distribution)
        # Add measurement Noise R_t
        z_cov = z_cov + self.system_model.R.cov

        K = np.matmul(xz_cross_cov, np.linalg.inv(z_cov))
        corrected_mean = pred_mean + np.dot(K, (z_measurement - z_mean))  # equation 15  in Marc's ACC paper
        corrected_cov = pred_cov - np.dot(K, np.transpose(xz_cross_cov))
        filtered_distribution = GaussianState(corrected_mean, corrected_cov)
        return filtered_distribution

    def _smoother(self, x_distribution, x_next_distribution, t):
        xx_mean, xx_cov, xx_cross_cov, = self.predict(self.transition, x_distribution, t)
        # Add transition Noise Q_t
        xx_cov = xx_cov + self.system_model.Q.cov

        # calculate smoother gain J_t
        J = np.dot(xx_cross_cov, np.linalg.inv(xx_cov))

        smoothed_mean = x_distribution.mean + np.dot(J, (x_next_distribution.mean - xx_mean))
        smoothed_cov = x_distribution.cov + np.dot(np.dot(J, x_next_distribution.cov - xx_cov), J.T)
        smoothed_distribution = GaussianState(smoothed_mean, smoothed_cov)
        return smoothed_distribution

    def _step_kalman_filter(self, x_distribution, z_observation):
        """
        This is a single step in kalman filter

        1) Given x_mean and x_sigma predict the t+1 value using transistion
            function and moment matching method

        2) Project latent state through  measurement function to obtain predcitive
           observation density

        3) correct the density using Kalman gain


        We use x as latent state distribution
        """

        x_t_plus_1 = self.moment_matching.predict(self.system_model.transition_function, x_distribution)

        pred_mean, pred_cov = self.predict(self.transition, x_distribution)
        pred_distribution = GaussianState(pred_mean, pred_cov)

        z_mean, xz_sigma, z_sigma = self.predict(self.measurement, pred_distribution)

        # calculate Kalman Gain K
        K = np.matmul(xz_sigma, np.linalg.inv(z_sigma))
        corrected_mean = x_mean + np.dot(K, (z_observation - z_mean))  # equation 15  in Marc's ACC paper
        corrected_cov = x_sigma - np.dot(K, np.transpose(xz_sigma))
        x_distribution_new = GaussianState(corrected_mean, corrected_cov)
        yield x_distribution_new

    def _predict(self, x_mean, xx_sigma):
        # we don't need cross covariance here, ignore it
        pred_mean, _, pred_sigma = self.transition_model.predict_transition(x_mean, xx_sigma)

        return pred_mean, pred_sigma

    def correct(self, z_observation, x_mean, x_sigma):
        # Todo : Check if it is better to pass inverse always or to use Cholesky for stability
        # (Numpy lstq or solve may work)
        z_mean, xz_sigma, z_sigma = self.observation_model.predict_observation(x_mean, x_sigma)

        # calculate Kalman Gain K
        K = np.matmul(xz_sigma, np.linalg.inv(z_sigma))
        corrected_mean = x_mean + np.dot(K, (z_observation - z_mean))  # equation 15  in Marc's ACC paper
        corrected_sigma = x_sigma - np.dot(K, np.transpose(xz_sigma))
        return corrected_mean, corrected_sigma

    def smooth(self, x_mean, x_sigma, x_next_mean, x_next_sigma):
        xx_mean, xx_cross_sigma, xx_sigma = self.transition_model.predict_transition(x_mean, x_sigma)

        # calculate smoother gain J_t
        J = np.dot(xx_cross_sigma, np.linalg.inv(xx_sigma))

        smoothed_mean = x_mean + np.dot(J, (x_next_mean - xx_mean))
        smoothed_sigma = x_sigma + np.dot(np.dot(J, x_next_sigma - xx_sigma), np.transpose(J))
        return smoothed_mean, smoothed_sigma