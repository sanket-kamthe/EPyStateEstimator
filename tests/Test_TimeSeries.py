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
import numpy as np
from Systems.DynamicSystem import GaussianNoise, DynamicSystemModel
SEED = 12345


def dummy_transition(x, u=None, t=None, *args, **kwargs):
    result = dict()
    result['x'] = x
    result['u'] = u
    result['t'] = t
    # result['args']
    return result


def dummy_measurement(x, *args, **kwargs):
    result = dict()
    result['x'] = x
    # result['args']
    return result


class CreateGaussianLinearLatent(DynamicSystemModel):
    def __init__(self, sys_dim, meas_dim, seed=SEED):
        np.random.seed(seed=seed)
        system_noise = GaussianNoise(sys_dim)
        measurement_noise = GaussianNoise(meas_dim)

        A = np.random.randn(sys_dim, sys_dim)
        # self.B = np.random.randn(sys_dim, sys_dim)
        C = np.random.randn(meas_dim, sys_dim)
        # def transition(x, )

    def transition(self):
        pass


class TestNoiseModel(unittest.TestCase):
    def test_defaults(self):
        eta = GaussianNoise()
        np.testing.assert_equal(eta.dimension, 1)
        np.testing.assert_allclose(eta.params.Q, np.eye(1))
        np.testing.assert_allclose(eta.mean, 0.0)
        np.testing.assert_allclose(eta.cov, np.eye(1))

    def test_1drandom_sample(self):
        eta = GaussianNoise()

        np.random.seed(seed=SEED)
        actual = eta.sample()

        np.random.seed(seed=SEED)
        desired = np.random.multivariate_normal(mean=np.zeros((1,), dtype=float),
                                                cov=np.eye(1))

        np.testing.assert_allclose(actual, desired)


class TestDynamicSystem(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=SEED)
        sys_dim = 3
        meas_dim = 2
        dyn_sys = DynamicSystemModel(system_dim=sys_dim,
                                     measurement_dim=meas_dim,
                                     transition=dummy_transition,
                                     measurement=dummy_measurement,
                                     system_noise=GaussianNoise(dimension=sys_dim),
                                     measurement_noise=GaussianNoise(dimension=meas_dim))
        self.system = dyn_sys
        self.x = np.random.randn(sys_dim)
        self.y = np.random.randn(meas_dim)
        self.t_zero = 0.0
        self.t = 1.1

    def test_transition(self):
        desired = {'x': self.x, 'u': None, 't': 0.0}
        actual_dict = self.system.transition(self.x, None, self.t_zero)
        self.assertDictEqual(actual_dict, desired)






if __name__ == '__main__':
    unittest.main()
