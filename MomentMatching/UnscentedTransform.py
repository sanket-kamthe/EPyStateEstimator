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

    def predict(self, mean, covariance):

        return pred_mean, pred_covariance

    def correct(self, observation):
        return corrected_mean, corrected_covariance

    def smooth(self, mean, covariance):
        return smoothed_mean, smoothed_covariance


class TransitionModel:

    def __init__(self, transition_function, moment_matching, noise_covariance):
        self.transition_function = transition_function
        self.moment_matching = moment_matching
        self.Q = noise_covariance

    def predict_gaussian_and_cross_covariance(self, mean, covariance):

        return pred_mean, pred_cross_covariance, pred_covariance


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
