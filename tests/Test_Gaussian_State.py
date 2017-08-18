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
from ..MomentMatching.StateModels import GaussianState


class Test1DGaussianStateModel(unittest.TestCase):
    def setUp(self):
        self.dim = 1
        self.mean = np.array([0.1])
        self.cov = 0.5 * np.eye(self.dim)
        self.state = GaussianState(mean_vec=self.mean, cov_matrix=self.cov)

    def test_1d_instantiation_dim(self):
        self.assertEqual(self.state.dim, self.dim, 'Dimension value read incorrectly')
        self.assertEqual(self.state.mean, self.mean, 'mean value read incorrectly')
        self.assertEqual(self.state.cov, self.cov, 'Cov value read incorrectly')

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


class TestGaussianStateModelArithmetic(unittest.TestCase):
    def setUp(self):
        self.dim = random.randint(2, 5)
        self.mean = np.random.randn(self.dim, 1)
        self.cov = np.random.randn(self.dim, self.dim)
        self.cov = self.cov.T @ self.cov
        self.state1 = GaussianState(mean_vec=self.mean, cov_matrix=self.cov)
        self.mean = np.random.randn(self.dim, 1)
        self.cov = np.random.randn(self.dim, self.dim)
        self.cov = self.cov.T @ self.cov
        self.state2 = GaussianState(mean_vec=self.mean, cov_matrix=self.cov)

    def test_Gauss_multiplication(self):
        self.state3 = self.state1 * self.state2
        self.assertIsInstance(self.state3, GaussianState)
        print(self.state3.cov)
        print(self.state3.precision)
        # np.testing.assert_allclose(self.state2, self.state3/self.state1)



if __name__ == '__main__':
    unittest.main()