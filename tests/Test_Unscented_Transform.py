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
import random
from MomentMatching.baseMomentMatch import UnscentedTransform

SEED = 106

class TestSigmaPoints(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=SEED)
        self.dim = np.random.randint(2, 5)
        self.dim = 1
        # self.dim = random.randint(2, 5)
        self.mean = np.random.randn(self.dim) * 0.0
        self.cov = 1 * np.eye(self.dim)  # square root is 0.3
        self.transform = UnscentedTransform()

    def test_sigma_points(self):
        print(self.dim)
        # L = self.transform._sigma_points(self.mean, self.cov, 1)
        # sigma_plus_L = self.mean + L
        # sigma_minus_L = self.mean - L
        # print(L)
        # print('#'*40)
        # list_val = [self.mean.tolist()] + sigma_plus_L.tolist() + sigma_minus_L.tolist()
        # print(list_val)
        # list_val = list_val.insert(0, 0.0)
        list_val = self.transform._sigma_points(self.mean, self.cov, 2)
        print(list_val)
        print(len(list_val))
        pts = np.asarray(list_val, dtype=float)
        print(pts.shape)

    def test_weights(self):
        n, alpha, beta, kappa = self.transform.n, self.transform.alpha, self.transform.beta, self.transform.kappa
        print(f'weights = {self.transform._weights(n, alpha, beta, kappa)}')

        # np.testing.assert_allclose((self.transform.w_m, self.transform.W), self.transform._weights(n, alpha, beta, kappa))

if __name__ == '__main__':
    unittest.main()

