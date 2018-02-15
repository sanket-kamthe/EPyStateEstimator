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

import numpy as np
from StateModel import GaussianState
from MomentMatching import MomentMatching as transform


def predict(self, prior_state, t=None, u=None, *args, **kwargs):
    xx_mean, xx_cov, _ = transform(self.transition,
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
        transform(self.transition,
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