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
import unittest
from MomentMatching.TimeSeriesModel import UniformNonlinearGrowthModel
from Filters.KalmanFilter import KalmanFilterSmoother
from MomentMatching.newMomentMatch import UnscentedTransform

SEED = 120816


class TestKalmanFilter(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=SEED)
        N = 10
        self.system = UniformNonlinearGrowthModel()
        self.transform = UnscentedTransform(n=1)
        self.data = self.system.simulate(N=N) #system_simulation(N)
        x_true, x_noisy, y_true, y_noisy = zip(*self.data)
        self.kf = KalmanFilterSmoother(moment_matching=self.transform,
                                  system_model=self.system)

    def test_kf(self):
        x_true, x_noisy, y_true, y_noisy = zip(*self.data)
        self.kf.kalman_filter(y_noisy, prior_state=self.system.init_state)
        self.assertIsInstance(self.kf, KalmanFilterSmoother)


if __name__ == '__main__':
    unittest.main()