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

######################################################
# estimator
#     pred
#     correct
#     smooth
#     fwd
#     back
#     meas
from StateModel import Gaussian

# Kalman estimator

class estimator:
    def __init__(self, trans_map, meas_map,
                 trans_noise=None, meas_noise=None,
                 power=1):
        self.trans_map = trans_map
        self.meas_map = meas_map
        self.transition_noise = trans_noise
        self.measurement_noise = meas_noise
        self.power = power

    def proj_trans(self, func, state):
        xx_mean, xx_cov, _ = self.trans_map(func, state)
        xx_cov += self.transition_noise
        xx_cov /= self.power

        pred_state = Gaussian(xx_mean, xx_cov)
        return pred_state

    def proj_meas(self, func, state, meas):
        z_mean, z_cov, xz_cross_cov = \
            self.meas_map(func, state)

        z_cov += self.measurement_noise
        z_cov /= self.power

        # kalman_gain = np.matmul(xz_cross_cov, np.linalg.pinv(z_cov))
        kalman_gain = np.linalg.solve(z_cov, xz_cross_cov.T).T
        mean = state.mean + kalman_gain @ (meas - z_mean)  # equation 15  in Marc's ACC paper
        cov = state.cov - kalman_gain @ xz_cross_cov.T

        corrected_state = Gaussian(mean, cov)
        return corrected_state

    def proj_back(self, func, state, next_state):
        xx_mean, xx_cov, xx_cross_cov = \
            self.trans_map(func, state)

        xx_cov += self.transition_noise
        xx_cov /= self.power

        # J = xx_cross_cov @ np.linalg.pinv(xx_cov)
        J = np.linalg.solve(xx_cov, xx_cross_cov.T).T
        mean = state.mean + np.dot(J, (next_state.mean - xx_mean))
        cov = state.cov + J @ (next_state.cov - xx_cov) @ J.T
        smoothed_state = Gaussian(mean, cov)
        return smoothed_state
