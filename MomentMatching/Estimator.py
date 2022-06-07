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
import numpy as np
from Utils.linalg import symmetrize
from MomentMatching import MappingTransform


class Estimator:
    '''
    This class contains methods to compute the approximate time/measurement/backward update
    for a given linearisation method (such as Taylor transform, unscented transform, etc).
    '''
    def __init__(self,
                 trans_map: MappingTransform,
                 meas_map: MappingTransform,
                 trans_noise: np.ndarray=None,
                 meas_noise: np.ndarray=None,
                 power: int=1,
                 damping: int=1):
        self.trans_map = trans_map
        self.meas_map = meas_map
        self.transition_noise = trans_noise
        self.measurement_noise = meas_noise
        self.power = power
        self.damping = damping

    def proj_trans(self, func, state):
        """
        Time update.
        Eqs (52)-(53) for standard EP.
        Eqs (140)-(141) for power EP.
        """
        xx_mean, xx_cov, _ = self.trans_map(func, state)
        xx_cov += self.transition_noise
        xx_cov /= self.power
        xx_cov = symmetrize(xx_cov)
        np.linalg.cholesky(xx_cov)
        pred_state = Gaussian(xx_mean, xx_cov)
        return pred_state

    def proj_meas(self, func, state, meas):
        """
        Measurement update.
        Eqs (56)-(57) for standard EP.
        Eqs (148)-(151) for power EP.
        """
        meas = np.atleast_1d(meas)
        np.linalg.cholesky(state.cov)
        z_mean, z_cov, xz_cross_cov = \
            self.meas_map(func, state)

        z_cov += self.measurement_noise / self.power
        z_cov = symmetrize(z_cov)
        np.linalg.cholesky(z_cov)
        kalman_gain = np.linalg.solve(z_cov, xz_cross_cov.T).T
        mean = state.mean + kalman_gain @ (meas - z_mean)  # equation 15  in Marc's ACC paper
        cov = state.cov - kalman_gain @ xz_cross_cov.T
        cov = symmetrize(cov)
        corrected_state = Gaussian(mean, cov)
        return corrected_state

    def proj_back(self, func, state, next_fwd_cavity, next_state=None):
        """
        Backward update.
        Eqs (78)-(79) for standard EP.
        Eqs (165)-(167) for power EP.
        """
        xx_mean, xx_cov, xx_cross_cov = \
            self.trans_map(func, state)

        xx_cov += self.transition_noise / self.power
        xx_cov = symmetrize(xx_cov)
        J = np.linalg.solve(xx_cov, xx_cross_cov.T).T
        if self.power == 1 and next_state is not None:
            mu = next_state.mean
            Sigma = next_state.cov
        else:
            q = Gaussian(xx_mean, xx_cov) * (next_fwd_cavity ** self.power)
            mu = q.mean
            Sigma = q.cov
        mean = state.mean + np.dot(J, (mu - xx_mean))
        cov = state.cov + J @ (Sigma - xx_cov) @ J.T
        cov = symmetrize(cov)
        smoothed_state = Gaussian(mean, cov)
        return smoothed_state

