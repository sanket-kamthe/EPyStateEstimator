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

import unittest
import random
import numpy as np
from ..MomentMatching.StateModels import GaussianState, moment_to_natural, natural_to_moment


class Test1DGaussianStateModel(unittest.TestCase):
    def setUp(self):
        self.dim = 1
        self.mean = np.array([0.1])
        self.cov = 0.5 * np.eye(self.dim)
        self.state = GaussianState(mean_vec=self.mean, cov_matrix=self.cov)

    def test_1d_instantiation_dim(self):

        self.assertEqual(self.state.dim, self.dim, 'Dimension value read incorrectly')
        np.testing.assert_allclose(self.state.mean, self.mean, err_msg='mean value read incorrectly')
        np.testing.assert_allclose(self.state.cov, self.cov, err_msg='Cov value read incorrectly')

    def test_1d_precision_shift_areNone(self):
        self.assertIsNone(self.state._precision)
        self.assertIsNone(self.state._shift)

    def test_1d_precision(self):
        self.assertAlmostEqual(self.state.precision, 1/self.cov)

    def test_1d_shift(self):
        self.assertAlmostEqual(self.state.shift, self.mean/self.cov)


class TestDiagGaussianStateModel(unittest.TestCase):
    def setUp(self):
        self.dim = random.randint(2, 5)
        self.mean = np.random.randn(self.dim, 1)
        self.cov = 0.5 * np.eye(self.dim)
        self.state = GaussianState(mean_vec=self.mean, cov_matrix=self.cov)

    def test_precision(self):
        np.testing.assert_allclose(4 * self.cov, self.state.precision)

    def test_shift(self):
        shift = np.dot(np.linalg.inv(self.cov), self.mean)
        np.testing.assert_allclose(self.state.shift, shift)

class TestGaussianParamsTransform(unittest.TestCase):
    def setUp(self):
        self.dim = random.randint(2, 5)
        self.mean = np.random.randn(self.dim, 1)
        temp = np.random.randn(self.dim, self.dim)
        self.cov = temp.T @ temp
        self.state = GaussianState(mean_vec=self.mean, cov_matrix=self.cov)

    def test_precision_function(self):
        test_precision, test_shift = moment_to_natural(self.mean, self.cov)
        np.testing.assert_allclose(test_precision, self.state.precision)

    def test_shift_function(self):
        test_precision, test_shift = moment_to_natural(self.mean, self.cov)
        np.testing.assert_allclose(test_shift, self.state.shift)

    def test_mean_function(self):
        test_mean, test_cov = natural_to_moment(self.state.precision, self.state.shift)
        np.testing.assert_allclose(test_mean, self.state.mean)
        np.testing.assert_allclose(test_mean, self.mean)

    def test_cov_function(self):
        test_mean, test_cov = natural_to_moment(self.state.precision, self.state.shift)
        np.testing.assert_allclose(test_cov, self.state.cov)
        np.testing.assert_allclose(test_cov, self.cov)


class TestGaussianStateModelArithmetic(unittest.TestCase):
    def setUp(self):
        self.dim = random.randint(3, 7)
        self.mean = np.random.randn(self.dim, 1)
        temp = np.random.randn(self.dim, self.dim)
        self.cov = temp.T @ temp
        self.state1 = GaussianState(mean_vec=self.mean, cov_matrix=self.cov)
        self.mean = np.random.randn(self.dim, 1)
        temp = np.random.randn(self.dim, self.dim)
        self.cov = temp.T @ temp
        self.state2 = GaussianState(mean_vec=self.mean, cov_matrix=self.cov)

    def test_Gauss_multiplication_division(self):
        self.state3 = self.state1 * self.state2
        self.assertIsInstance(self.state3, GaussianState)
        # print(self.state3.cov)
        #print(self.state3.precision)
        self.state4 = self.state3 / self.state1
        self.assertIsInstance(self.state4, GaussianState)
        np.testing.assert_allclose(self.state4.cov, self.state2.cov)
        np.testing.assert_allclose(self.state4.mean, self.state2.mean)
        #np.testing.assert_allclose(self.state2, self.state3/self.state1)

        self.state5 = self.state3 / self.state2
        self.assertIsInstance(self.state5, GaussianState)
        np.testing.assert_allclose(self.state5.cov, self.state1.cov)
        np.testing.assert_allclose(self.state5.mean, self.state1.mean)

class TestGaussianStatePower(unittest.TestCase):
    def setUp(self):
        self.dim = random.randint(3, 7)
        self.mean = np.random.randn(self.dim, 1)
        temp = np.random.randn(self.dim, self.dim)
        self.cov = temp.T @ temp
        self.state1 = GaussianState(mean_vec=self.mean, cov_matrix=self.cov)

    def test_power_unity(self):
        self.state2 = self.state1 ** 1
        np.testing.assert_allclose(self.state2.cov, self.state1.cov)
        np.testing.assert_allclose(self.state2.mean, self.state1.mean)
        assert (self.state2 == self.state1)

    def test_power_integer(self):
        self.pow = random.randint(2, 7)
        self.state2 = self.state1
        for _ in range(self.pow-1):
            self.state2 = self.state2 * self.state1

        self.state_pow = self.state1 ** self.pow
        assert isinstance(self.state_pow, GaussianState)
        np.testing.assert_allclose(self.state_pow.cov, self.state2.cov)
        np.testing.assert_allclose(self.state_pow.mean, self.state2.mean)
        # assert (self.state2 == self.state_pow)





if __name__ == '__main__':
    unittest.main()