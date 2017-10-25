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
from MomentMatching.newMomentMatch import UnscentedTransform, TaylorTransform
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-poster')
np.set_printoptions(precision=4)
from Utils.Plot_Helper import plot_gaussian
from Utils.Metrics import nll, rmse

SEED = 125


class TestKalmanFilter(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=SEED)
        N = 50
        self.system = UniformNonlinearGrowthModel()
        self.transform = UnscentedTransform(n=1, alpha=1, beta=0, kappa=1)
        # self.transform = TaylorTransform()
        self.data = self.system.simulate(N=N) #system_simulation(N)
        x_true, x_noisy, y_true, y_noisy = zip(*self.data)
        self.kf = KalmanFilterSmoother(moment_matching=self.transform,
                                  system_model=self.system)
        self.N = N

    def test_kf(self):
        x_true, x_noisy, y_true, y_noisy = zip(*self.data)
        result = \
            self.kf.kalman_filter(y_noisy,
                                  prior_state=self.system.init_state)
        self.assertIsInstance(self.kf, KalmanFilterSmoother)

        smoothed = self.kf.kalman_smoother(result)
        plt.plot(x_true, 'r--', label='True value')
        plt.scatter(list(range(self.N)), y_noisy)
        # plt.plot(y_noisy)
        # plt.plot(y_true, 'g--')
        plt.plot([x.mean for x in result], label='Filtered')
        # plot_gaussian(result, label='Filtered')
        # plt.figure()
        # plt.plot(x_true, 'r--')
        # plt.plot([x.mean for x in smoothed], 'g-', label='Smoothed')
        plot_gaussian(smoothed, label='Smoothed')

        print('\n Filtered NLL = {}, RMSE = {}'.format(nll(result, x_true),
                                                       rmse(result, x_true)))

        print('\n Smoothed NLL = {}, RMSE ={}'.format(nll(smoothed, x_true),
                                                      rmse(smoothed, x_true)))
        #print('Filtered RMSE ={},  NLL{}'.format(nll(result, x_true)))
        print(nll(smoothed, x_true))
        # print()
        plt.legend()
        # plt.show()



if __name__ == '__main__':
    unittest.main()