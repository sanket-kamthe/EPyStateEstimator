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


class KalmanFilterSmoother:

    def __init__(self, transition_model, observation_model, moment_matching):
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.moment_matching = moment_matching

    def predict(self, x_mean, xx_sigma):
        # we don't need cross covariance here, ignore it
        pred_mean, _, pred_sigma = self.transition_model.predict_transition(x_mean, xx_sigma)

        return pred_mean, pred_sigma

    def correct(self, z_observation, x_mean, x_sigma):
        # Todo : Check if it is better to pass inverse always or to use Cholesky for stability
        # (Numpy lstq or solve may work)
        z_mean, xz_sigma, z_sigma = self.observation_model.predict_observation(x_mean, x_sigma)

        # calculate Kalman Gain K
        K = np.matmul(xz_sigma, np.linalg.inv(z_sigma))
        corrected_mean = x_mean + np.dot(K, (z_observation - z_mean)) # equation 15  in Marc's ACC paper
        corrected_sigma = x_sigma - np.dot(K, np.transpose(xz_sigma))
        return corrected_mean, corrected_sigma

    def smooth(self, x_mean, x_sigma, x_next_mean, x_next_sigma):
        xx_mean, xx_cross_sigma, xx_sigma = self.transition_model.predict_transition(x_mean, x_sigma)

        # calculate smoother gain J_t
        J = np.dot(xx_cross_sigma, np.linalg.inv(xx_sigma))

        smoothed_mean = x_mean + np.dot(J, (x_next_mean - xx_mean))
        smoothed_sigma = x_sigma + np.dot (np.dot(J, x_next_sigma - xx_sigma), np.transpose(J))
        return smoothed_mean, smoothed_sigma


class TransitionModel:

    def __init__(self, ndims, transition_function, moment_matching, noise_covariance):
        # Todo : Add check functions to make sure that transition function is correct
        self.ndims = ndims
        assert (self._check_tranistion_function(transition_function))
        self.transition_function = transition_function
        self.moment_matching = moment_matching
        self.Q = noise_covariance


    def _check_tranistion_function(self, transition_function):
        """
        This function checks whether the transition function is valid

        :param transition_function: a function handle or a matrix
        :return: Boolean True if transition function is invalid

        """

    def predict_transition(self, mean, covariance):

        # Only to show the interface
        pred_mean = None
        pred_cross_sigma = None
        pred_sigma = None
        return pred_mean, pred_cross_sigma, pred_sigma


class MomentMatching:

    def __init__(self, given_function, noise_covariance):
        self.given_function = given_function
        self.noise = noise_covariance

    def _mean(self, mean, covariance):
        self.given_function(mean, covariance)
        return approximate_mean

    def _covariance(self, approximate_mean, mean, covariance):
        return approximate_cross_covariance, approximate_covariance

    def get_moments(self, mean, covariance):
        return NotImplementedError
        #return pred_mean, pred_cross_covariance, pred_covariance
