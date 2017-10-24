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
from MomentMatching.newMomentMatch import UnscentedTransform, MonteCarloTransform, TaylorTransform
from MomentMatching.newMomentMatch import EPS
from MomentMatching.StateModels import GaussianState

SEED = 120816


class TestFunctions:
    def __init__(self, dim=1):
        self.dim = dim
        A = np.random.randn(self.dim, self.dim)
        self.A = A @ A.T
        self.B = np.random.randn(self.dim, )

    def linear(self, x, t=None, u=None, *args, **kwargs):
        y = self.A @ x + self.B
        return y

    def sinusoidal(self, x, t=None, u=None, *args, **kwargs):
        y = np.sin(0.5 * x)
        return 2 * y

    def softplus(self, x, t=None, u=None, *args, **kwargs):
        y = np.log(1 + np.exp(x)) - np.log(2)
        return 2 * y


class TestTaylorTransform(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=SEED)
        self.dim = 1
        self.mean = np.random.randn(self.dim) * 0.0
        self.cov = 1 * np.eye(self.dim)
        self.transform = TaylorTransform(dimension_of_state=self.dim)
        self.test_functions= TestFunctions(self.dim)
        self.x_state = GaussianState(self.mean, self.cov)

    def test_taylor_transform_init(self):
        TT = self.transform
        np.testing.assert_allclose(self.dim, TT.dimension_of_state)
        np.testing.assert_allclose(TT.eps, EPS)

    def test_predict_linear(self):
        result_under_test = self.transform(self.test_functions.linear, self.x_state )

        pred_mean, pred_cov, pred_cross_cov = result_under_test
        np.testing.assert_allclose(pred_mean, self.test_functions.B)
        np.testing.assert_allclose(pred_cov, self.test_functions.A @ self.test_functions.A.T )

    def test_predict_sinusoidal(self):
        result_under_test = self.transform(self.test_functions.sinusoidal, self.x_state)
        pred_mean, pred_cov, pred_cross_cov = result_under_test
        np.testing.assert_allclose(pred_mean, self.mean)
        np.testing.assert_array_almost_equal(pred_cov, self.cov)

    def test_predict_softplus(self):
        result_under_test = self.transform(self.test_functions.softplus, self.x_state)
        pred_mean, pred_cov, pred_cross_cov = result_under_test
        np.testing.assert_allclose(pred_mean, self.mean)
        np.testing.assert_allclose(pred_cov, self.cov, rtol=EPS)


class TestUnscentedTransform(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=SEED)
        self.dim = 1
        self.mean = np.random.randn(self.dim) * 0.0
        self.cov = 0.25 * np.eye(self.dim)
        self.transform = UnscentedTransform(n=self.dim)
        self.test_functions= TestFunctions(self.dim)
        self.x_state = GaussianState(self.mean, self.cov)

    def test_unscented_transform_init(self):
        transform = self.transform
        np.testing.assert_allclose(self.dim, transform.n)

    def test_predict_linear(self):
        result_under_test = self.transform(self.test_functions.linear, self.x_state, fargs=None)

        pred_mean, pred_cov, pred_cross_cov = result_under_test
        np.testing.assert_allclose(pred_mean, self.test_functions.B)
        np.testing.assert_allclose(pred_cov, self.test_functions.A @ self.test_functions.A.T, atol=1e-3 )

    def test_predict_sinusoidal(self):
        result_under_test = self.transform(self.test_functions.sinusoidal, self.x_state, fargs=None)
        pred_mean, pred_cov, pred_cross_cov = result_under_test
        np.testing.assert_allclose(pred_mean, self.mean)
        np.testing.assert_allclose(pred_cov, self.cov, rtol=EPS, atol=1e-2)


    def test_predict_softplus(self):
        result_under_test = self.transform(self.test_functions.softplus, self.x_state, fargs=None)
        pred_mean, pred_cov, pred_cross_cov = result_under_test
        np.testing.assert_allclose(pred_mean, self.mean, rtol=EPS, atol=1e-1)
        np.testing.assert_allclose(pred_cov, self.cov, rtol=EPS,  atol=1e-1)


if __name__ == '__main__':
    unittest.main()